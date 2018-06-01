#!/usr/bin/env

import logging
_log = logging.getLogger(__name__)

import time
import sys
import os
import re
import json
import signal
import subprocess as sp
import cothread
import cothread.catools as ca

# commands tokens for event queue
sigchld = "SIGCHLD"
start = "START"
stop = "STOP"
abort = "ABORT"
join = "JOIN"

# helper to ignore current value of subscription which is the first
# update sent after (re)connect.
class Delta(object):
    def __init__(self, action):
        self.connected, self.action = False, action
    def __call__(self, val):
        if not val.ok:
            _log.debug("%s disconnect", val.name)
            self.connected = False
        elif self.connected:
            _log.debug("%s update %s", val.name, val)
            self.action(val)
        else:
            _log.debug("%s (re-)connect %s", val.name, val)
            self.connected = True

class ProcControl(object):
    all_procs = set()
    def __init__(self, pref, args, logto=None, **kws):
        self.pref = bytes(pref)
        self.args, self.logto, self.launch_args = args, logto, kws

        _log.debug("%s Setup", self.pref)

        # wakeup with one of the command tokens
        self.evt = cothread.EventQueue()

        self.cmd_start = ca.camonitor(b'%sSTRT_'%self.pref, Delta(lambda _val:self.evt.Signal(start)), notify_disconnect=True)
        self.cmd_stop  = ca.camonitor(b'%sSTOP_'%self.pref, Delta(lambda _val:self.evt.Signal(stop)),  notify_disconnect=True)
        self.cmd_abort = ca.camonitor(b'%sABRT_'%self.pref, Delta(lambda _val:self.evt.Signal(abort)), notify_disconnect=True)

        ca.caput(b'%sSTS'%self.pref, 0, wait=True)

        self.child = None

        # our long running task
        self.ca_task = cothread.Spawn(self.loop)

        self.all_procs.add(self)
        _log.info("%s Ready", self.pref)

    def close(self):
        self.evt.Signal(abort)
        self.evt.Signal(join)
        try:
            self.ca_task.Wait()
        except:
            _log.exception("%s close() join"%self.pref)

        assert self.child is None or self.child.poll() is not None, (self.child and self.child.poll())

    def sigchld(self):
        self.evt.Signal(sigchld)

    def loop(self):
        try:
            while True:
                val = self.evt.Wait()
                _log.debug("Wakeup with %s", val)

                self.check_status()

                if val == start:
                    self.handle_start()
                elif val == stop:
                    self.handle_stop()
                elif val == abort:
                    self.handle_abort()
                elif val == sigchld:
                    pass
                elif val == join:
                    break
                else:
                    _log.warn("Spurious wakeup with %s", val)
        except:
            _log.exception("%s error in runner", self.pref)
            raise

    def check_status(self):
        if self.child is None:
            sts = 0 # Crash (really Init)
            ret = -1
        else:
            ret = self.child.poll()
            _log.debug("%s Child status %s", self.pref, ret)

            if ret is None:
                sts = 2 # Running
            elif ret==0:
                sts = 1 # Complete
            else:
                sts = 0 # Crash

        ca.caput(b'%sSTS'%self.pref, sts, wait=True)

        self.current_status = sts

    def handle_start(self):
        if self.current_status != 2:
            _log.debug("%s starting %s %s", self.pref, self.args, self.launch_args)

            args = {'stderr':sp.STDOUT}
            fp = None

            try:
                if self.logto:
                    _log.debug("Log to '%s'", self.logto)
                    fp = open(self.logto, 'w')
                    fp.write('# %s\n# %s\n# --- begin output ---\n'%(self.args, time.ctime()))
                    args['stdout'] = fp.fileno()

                args.update(self.launch_args)

                self.child = sp.Popen(self.args, **args)

            finally:
                if fp is not None:
                    fp.close()

            ca.caput(b'%sSTS'%self.pref, 2, wait=True)

    def handle_stop(self):
        if self.current_status == 2:
            _log.debug("%s SIGINT", self.pref)
            self.child.send_signal(signal.SIGINT)

    def handle_abort(self):
        if self.current_status == 2:
            _log.debug("%s kill", self.pref)
            self.child.kill()

class KeepAlive(object):
    def __init__(self, conf):
        self.conf = conf
        self.pv = conf.get('pv')
        if self.pv is None:
            return
        self.max = conf.get('max', 2**31 -1) # set default rollover for 32-bit signed integer
        self.done = cothread.Event()
        self.val = 0
        self.task = cothread.Spawn(self._task)
    def close(self):
        if self.pv is None:
            return
        self.done.Signal()
        self.task.Wait() # join
        self.pv = None
        self.val = 0
    def incr(self):
            if self.val < self.max:
                self.val += 1
            else:
                self.val = 0
        
    def _task(self):
        while True:
            try:
                self.done.Wait(self.conf.get('interval', 1.0))
                return # wakeup from close()
            except cothread.Timedout:
                pass

            try:
                ca.caput(str(self.pv), self.val, timeout=self.conf.get('timeout', 3))
            except cothread.Timedout:
                _log.debug("Timeout writing heartbeat "+self.pv)
            except ca.ca_nothing:
                _log.debug("Timeout writing heartbeat "+self.pv)
            except:
                _log.exception("Error writing heartbeat "+self.pv)

            self.incr()

def poke_all():
    _log.debug("SIGCHLD 2")
    [proc.sigchld() for proc in ProcControl.all_procs]

def handle_child(sig, frame):
    _log.debug("SIGCHLD 1")
    cothread.Callback(poke_all)

def getargs():
    from argparse import ArgumentParser
    P=ArgumentParser()
    P.add_argument('config', help='JSON config file')
    P.add_argument('-d','--debug', action='store_const', const=logging.DEBUG, default=logging.INFO)
    return P.parse_args()

def main(args):
    logging.basicConfig(level=args.debug, format='%(asctime)s %(levelname)s %(message)s')

    signal.signal(signal.SIGCHLD, handle_child)

    confdir = os.path.dirname(args.config)

    with open(args.config, 'r') as F:
        conf = F.read()

    # we work from the directory containing the config file so that
    # any relative paths it contains will be relative to the
    # location of the config file
    os.chdir(confdir or '.')

    conf = re.sub('#[^\n\r]*\r?\n', '', conf) # strip '#' comments
    try:
        conf = json.loads(conf)
    except ValueError as e:
        print conf
        print 'Syntax Error:', e
        sys.exit(1)

    for name, sect in conf['procs'].items():
        ProcControl(name, sect['command'], logto=sect.get('logfile'))

    kp = KeepAlive(conf.get('keepalive', {}))

    try:
        cothread.WaitForQuit()
    except KeyboardInterrupt:
        pass

    for proc in ProcControl.all_procs:
        proc.close()

    kp.close()

if __name__=='__main__':
    main(getargs())
