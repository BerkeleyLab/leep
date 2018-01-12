
import logging
_log = logging.getLogger(__name__)

import json, zlib, re

import numpy

caget = caput = camonitor = None
try:
    from cothread.catools import caget, caput, camonitor, FORMAT_TIME
except ImportError:
    pass
else:
    from cothread import Event


# flags for _wait_acq
IGNORE="IGNORE"
WARN="WARN"
ERROR="ERROR"

def open(addr, **kws):
    """Access to a single LEEP Device.
    
    :param str addr: Device Address.  prefix with "ca://<prefix>" or "leep://<ip>[:<port>]"
    """
    if addr.startswith('ca://'):
        if caget is None:
            raise RuntimeError('ca:// not available, cothread module not in PYTHONPATH')
        return CADevice(addr[5:], **kws)
    elif addr.startswith('leep://'):
        return LEEPDevice(addr[7:], **kws)
    else:
        raise ValueError("Unknown '%s' must begin with ca:// or leep://"%addr)

class AcquireBase(object):
    def close(self):
        pass
    def inc_tag(self):
        pass
    def wait(self, tag=WARN):
        pass

    def __enter__(self):
        return self
    def __exit__(self, A, B, C):
        self.close()

class DeviceBase(object):
    def __init__(self, instance=[]):
        self.instance = instance[:] # shallow copy

    def expand_regname(self, name, instance=[]):
        """Return a full register name from the short name and optionally instance number(s)

        >>> D.expand_regname('XXX', instance=[0])
        'tget_0_delay_pc_XXX'
        """
        # build a regexp
        # TODO: cheating a little bit, should enforce a '_' between instance parts
        I = r'.*'.join(map(lambda x:re.escape(str(x)), self.instance + instance))
        R = re.compile('^.*%s.*%s$'%(I, name))

        ret = None
        for reg in self.regmap:
            if R.match(reg) is not None
                if ret is not None:
                    raise RuntimeError('%s %s Matched more than one register %s, %s'%(name, I, ret, reg))
                ret = reg
        return ret

    def reg_write(self, ops, instance=[]):
        """
        >>> D.reg_write([
            ('reg_a', 5),
            ('reg_b', 6),
        ])
        """
        raise NotImplementedError

    def reg_read(self, names, instance=[]):
        """
        >>> A, B = D.reg_read(['reg_a', 'reg_b'])
        """
        raise NotImplementedError

    def get_reg_info(self, name, instance=[]):
        """Return a dict describing the named register.
        """
        return self.regmap[self.expand_regname(name, instance=instance)]

    def set_tgen(self, tbl, instance=[]):
        """Load waveform to PRC tgen
        
        Provided tbl must be an Nx3 array
        of delay (in ticks) and 2x levels (I/Q in counts).
        First delay must be zero.

        >>> D.set_tgen([
            (0 , 100, 0), # time zero, I=100, Q=0
            (50, 0, 0)    # time 50, I=0, Q=0
        ])

        The XXX sequencer runs "programs" of writes.
        Each instruction is stored in 4 words/addresses.

        [0] Delay (applied _after_ write)
        [1] Address to write
        [2] value high word (16-bits)
        [3] value low word (16-bits)
        """
        tbl = numpy.asarray(tbl)
        _log.debug("tgen input table %s", tbl)
        assert tbl.shape[1]==3, tbl.shape

        assert tbl[0,0]==0, ('Table must start with delay 0', tbl[0,0])
        # delay is applied after the write
        tbl[:-1,0] = tbl[1:,0]
        tbl[-1,0] = 0

        info = self.get_reg_info('proc_lim')
        assert info['addr_width'] == 2, info
        # lim has 4 addresses to set bounds on X and Y (aka. I and Q)
        # when high==low the output will be forced to the limit
        # aka. feedforward

        lim_X_hi = info['base_addr']
        lim_Y_hi = lim_X_hi + 1
        lim_X_lo = lim_X_hi + 2
        lim_Y_lo = lim_X_hi + 3

        info = self.get_reg_info('XXX')

        prg = numpy.zeros(2**info['addr_width'], dtype='u4')

        T, idx = 0, 0
        for i in range(tbl.shape[0]):
            delay, X, Y = tbl[i,:]
            delay -= T
            assert delay>=0, (i, delay)
            # delay 0 is really 4 ticks
            # so delay is offset by 4.
            # Delay applied _after_ write

            prg[idx+0] = 0
            prg[idx+1] = lim_X_lo
            prg[idx+2] = X>>16
            prg[idx+3] = X&0xffff
            prg[idx+4] = 0
            prg[idx+5] = lim_X_hi
            prg[idx+6] = X>>16
            prg[idx+7] = X&0xffff
            idx += 8

            prg[idx+0] = 0
            prg[idx+1] = lim_Y_lo
            prg[idx+2] = Y>>16
            prg[idx+3] = Y&0xffff
            prg[idx+4] = delay
            prg[idx+5] = lim_Y_hi
            prg[idx+6] = Y>>16
            prg[idx+7] = Y&0xffff
            idx += 8

        # address zero ends sequence
        prg[idx+0] = 0
        prg[idx+1] = 0
        prg[idx+2] = 0
        prg[idx+3] = 0
        idx += 4

        #for i in range(0, idx, 4):
        #    print "INST", prg[i+0], prg[i+1], prg[i+2], prg[i+3]

        _log.debug("XXX program %s", prg[:idx])
        self.reg_write(('XXX', prg))

    def acquire(self):
        """Setup for acquisition.  Returns an AcquireBase
        which can be used to wait for an acquisition to complete.
        """
        raise NotImplementedError

class CADevice(DeviceBase):
    def __init__(self, addr, timeout=1.0, **kws):
        DeviceBase.__init__(self, **kws)
        self.timeout = timeout
        self.prefix = addr # PV prefix

        _log.debug('caget("%s")', addr+'ctrl:JInfo-I')
        # fetch mapping from register name to info dict
        # {'records':{'reg_name':{'<info>':'<value>'}}}
        # common info tags are:
        #  'input' PV to read register
        #  'output' PV to write register
        #  'increment' PV to +1 register
        info = json.loads(zlib.decompress(caget(addr+'ctrl:JInfo-I')))
        self._info = info['records']

        # raw JSON blob from device
        # {'reg_name:{'base_addr':0, ...}}
        _log.debug('caget("%s")', addr+'ctrl:JSON-I')
        self.regmap = json.loads(zlib.decompress(caget(addr+'ctrl:JSON-I')))

    def pv_write(self, suffix, value):
        """Shortcut for operations which don't map naturally to registers
        when going through an IOC
        """
        _log.debug('caput("%s", %s)', self.prefix+suffix, value)
        caput(self.prefix+suffix, value, wait=True, timeout=self.timeout)

    def reg_write(self, ops, instance=[]):
        for name, value in ops:
            name = self.expand_regname(name, instance=instance)
            pvname = str(info['output'])
            _log.debug('caput("%s", %s)', pvname, value)
            caput(pvname, value, wait=True, timeout=self.timeout)

    def reg_read(self, names, instance=[]):
        ret = [None]*len(names)
        for i, name in enumerate(names):
            name = self.expand_regname(name, instance=instance)
            pvname = str(info['input'])

            _log.debug('caput("%s.PROC", 1)', pvname)
            caput(pvname+'.PROC', 1, wait=True, timeout=self.timeout)
            ret[i] = caget(pvname, timeout=self.timeout)
            _log.debug('caget("%s") -> %s', pvname, value)

        return ret

    class CAAcq(AcquireBase):
        def __init__(self, dev, prefix):
            # prefix should be "shell_X"
            self._dev, self._prefix = dev, prefix
            self.tag = None
            self._wait = Event()

            circle_reg = str(prefix["circle_data"])
            slow_reg = str(prefix["slow_data"])

            circle_pv = str(self._dev._info[circle_reg]['rawinput'])
            slow_pv = str(self._dev._info[slow_reg]['rawinput'])

            # cache initially empty
            self._circle = self._slow = None

            _log.debug('camonitor("%s") start', circle_pv)
            self.S1 = camonitor(circle_pv, self._new_circle, format=FORMAT_TIME)
            _log.debug('camonitor("%s") start', slow_pv)
            self.S2 = camonitor(slow_pv, self._new_slow, format=FORMAT_TIME)

        def close(self):
            if self.S1 is not None:
                self.S1.close()
                self.S2.close()
                self.S1 = self.S2 = None

        def _new_circle(self, value):
            _log.debug("Update circle")
            self._circle = value
            self._check_done()

        def _new_slow(self, value):
            _log.debug("Update slow")
            self._slow = value
            self._check_done()

        def _check_done(self):
            if self._slow is None or self._circle is None:
                _log.debug('Acq %s cache not ready', self._prefix)
                return
            #elif self._slow.raw_stamp != self._circle.raw_stamp:
            elif abs(self._slow.timestamp - self._circle.timestamp)<0.001:
                _log.debug('Acq %s cache old %s != %s', self._prefix, self._slow.timestamp, self._circle.timestamp)
                if self._slow.raw_stamp < self._circle.raw_stamp:
                    self._slow = None
                else:
                    self._circle = None
                return
            else:
                _log.debug('Acq %s cache ready', self._prefix)
                self._wait.Signal({'circle':self._circle, 'slow':self._slow})
                self._circle = self._slow = None

        def wait(self, tag=WARN, tag_at_end=True, timeout=10.0):
            expect = self.tag
            _log.debug("Wait for acquisition w/ tag=%s check %s", expect, tag)
            while True:
                ret = self._wait.Wait(timeout=timeout)
                slow = ret['slow']
                Tb, Te = slow[33], slow[34]

                if Tb<expect:
                    continue

                if Te<expect and tag_at_end:
                    continue

                if Te!=expect:
                    if tag!=IGNORE:
                        _log.debug("End tag early %d < %d", Te, expect)
                    if tag==ERROR:
                        continue

                _log.debug("Acquisition complete start=%d end=%d", Tb, Te)
                return ret

        def inc_tag(self):
            tagreg = str(self._prefix['dsp']['tag'])
            info = self._dev._info[tagreg]
            incpv = str(info['increment'])
            _log.debug('caput("%s", 1)', incpv)
            caput(incpv, 1, wait=True, timeout=self._dev.timeout)
            self.tag = self._dev.reg_read(tagreg)
            _log.debug("Acquisition tag now %s", self.tag)
            return self.tag

        def __enter__(self):
            return self

        def __exit__(self, A, B, C):
            if A is None:
                self.inc_tag()

    def acquire(self, instance=0):
        """Setup for acquisition
        """
        return self.CAAcq(self, RegName('shell')[instance])



class LEEPDevice(DeviceBase):
    def __init__(self, addr, timeout=0, **kws):
        DeviceBase.__init__(self, **kws)
        host, _sep, port = addr.partition(':')
        self.dest = (host, int(port or '50006'))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        self.sock.settimeout(timeout)

        self._readrom()

    def _exchange(self, addrs, values=None):

        pad = None
        if len(addrs)<3:
            pad = 3-len(addrs)
            addrs.extend([0]*pad)
            values.extend([None]*pad)

        msg = numpy.zeros(2+2*len(addrs), dtype=be32)
        msg[0] = random.randint(0,0xffffffff)
        msg[1] = msg[0]&0xffffffff

        for i,(A, V) in enumerate(zip(addrs, values), 1):
            A &= 0x00ffffff
            if V is None:
                A |= 0x10000000
            msg[2*i] = A
            msg[2*i+1] = V or 0

        tosend = msg.tostring()
        _log.debug("%s Send (%d) %s", self.dest, len(tosend), repr(tosend))
        self.sock.sendto(tosend, self.dest)

        while True:
            reply, src = self.sock.recvfrom(1024)
            _log.debug("%s Recv (%d) %s", src, len(reply), repr(reply))

            if len(reply)%8:
                reply = reply[:-(len(reply)%8)]

            if len(tosend)!=len(reply):
                _log.error("Reply truncated %d %d", len(tosend), len(reply))
                continue

            reply = numpy.fromstring(reply, be32)
            if (msg[:2]!=reply[:2]).any():
                _log.error('Ignore reply w/o matching nonce %s %s', msg[:2], reply[:2])
                continue
            elif (msg[2::2]!=reply[2::2]).any():
                _log.error('reply addresses are out of order')
                continue

            break

        ret = reply[3::2]
        if pad:
            ret = ret[:-pad]
        return ret

    def exchange(self, addrs, values=None):
        """Accepts a list of address and values (None to read).
        Returns a numpy.ndarray in the same order.
        """

        if values is None:
            values = [None]*len(addrs)

        ret = numpy.zeros(len(addrs), be32)
        for i in range(0, len(addrs), 127):
            A, B = addrs[i:i+127], values[i:i+127]

            P = self._exchange(A, B)
            ret[i:i+127] = P

        return ret

    def _readrom(self):
        self.descript = None
        self.codehash = None
        self.jsonhash = None
        self.json = None

        values = self.exchange(range(0x800, 0x1000))

        values = numpy.frombuffer(values, be16)
        _log.debug("ROM[0] %08x", values[0])
        values = values[1::2] # discard upper bytes

        while len(values):
            type = values[0]>>14
            size = values[0]&0x3fff
            _log.debug("ROM Descriptor type=%d size=%d", type, size)

            if type==0:
                break

            blob, values = values[1:size+1], values[size+1:]
            if len(blob)!=size:
                raise ValueError("Truncated ROM Descriptor")

            if type==1:
                blob = blob.tostring()
                if self.descript is None:
                    self.descript = blob
                else:
                    _log.info("Extra ROM Text '%s'", blob)
            elif type==2:
                blob = ''.join(["%04x"%b for b in blob])
                if self.jsonhash is None:
                    self.jsonhash = blob
                elif self.codehash is None:
                    self.codehash = blob
                else:
                    _log.info("Extra ROM Hash %s", blob)

            elif type==3:
                if self.json is not None:
                    _log.error("Ignoring additional JSON blob in ROM")
                else:
                    self.json = json.loads(zlib.decompress(blob.tostring()).decode('ascii'))

        if self.json is None:
            raise RuntimeError('ROM contains no JSON')
