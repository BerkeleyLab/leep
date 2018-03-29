#!/usr/bin/env

import logging
_log = logging.getLogger(__name__)

import signal
import subprocess as sp
import cothread
import cothread.catools as ca

sigchld = object()

class ProcControl(object):
    all_procs = set()
    def __init__(self, pref, args, **kws):
        self.pref = pref
        self.args, self.launch_args = args, kws

        # wakeup with: sigchld or a number
        self.evt = cothread.Event()

        # commands are updates w/ a value of 0 or 1
        self.cmd_sub = ca.camonitor('%sRUN'%self.pref, self.evt.Signal)
        # wait for, and ignore, initial value
        self.evt.Wait()

        ca.caput('%sSTS'%self.pref, 0, wait=True)

        self.child = None

        self.ca_task = cothread.Spawn(self.run)

        self.all_procs.add(self)

    def close(self):
        if self.child is None or self.child.poll() is not None:
            return
        try:
            self.child.kill()
        except OSError:
            _log.exception("%s close()"%self.pref)
        self.child.wait()

    def run(self):
        while True:
            # wakup on event (SIGCHLD or command), also every 10 seconds for a paranoia status check
            try:
                val = self.evt.Wait(10.0)
            except cothread.Timedout:
                val = None

            _log.debug("%s wakeup with %s", self.pref, val)

            # test/update child status on any wakeup

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

            ca.caput('%sSTS'%self.pref, sts, wait=True)
                
            if val==1:
                _log.debug("%s request start", self.pref)

                if ret is not None:
                    _log.debug("%s starting %s %s", self.pref, self.args, self.launch_args)

                    self.child = sp.Popen(self.args, **self.launch_args)

                    ca.caput('%sSTS'%self.pref, 2, wait=True)

            elif val==0:
                _log.debug("%s request stop", self.pref)

                if ret is None:
                    _log.debug("%s SIGINT", self.pref)
                    self.child.send_signal(signal.SIGINT)

            elif val==2:
                _log.debug("%s request abort", self.pref)

                if ret is None:
                    _log.debug("%s kill", self.pref)
                    self.child.kill()

def poke_all():
    _log.debug("SIGCHLD 2")
    for proc in ProcControl.all_procs:
        proc.evt.Signal(sigchld)

def handle_child(sig, frame):
    _log.debug("SIGCHLD 1")
    cothread.Callback(poke_all)

signal.signal(signal.SIGCHLD, handle_child)


ProcControl('SLP10:', ['/bin/sleep', '10'])

logging.basicConfig(level=logging.DEBUG)

try:
    cothread.WaitForQuit()
except KeyboardInterrupt:
    pass

for proc in ProcControl.all_procs:
    proc.close()
