
import logging
_log = logging.getLogger(__name__)

import json, zlib

import numpy

caget = caput = camonitor = None
try:
    from cothread.catools import caget, caput, camonitor, FORMAT_TIME
    from cothread import Event
except ImportError:
    pass


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

class RegName(object):
    """Helper for building heirarchial names
    
    >>> reg = RegName('base')
    >>> str(reg[0]['foo'])
    'base_0_foo'
    """
    def __init__(self, name=None, parts=None):
        if parts is not None:
            self._parts = parts
        else:
            self._parts = [] if name is None else [name]
    def __str__(self):
        return '_'.join(map(str,self._parts))
    def __repr__(self):
        return 'RegName(%s)'%self._parts
    def __getitem__(self, part):
        return self.__class__(parts=self._parts+[part])

class Base(object):

    def reg_write(self, name, value):
        "Write single register"
        assert value is not None, value
        self.reg_batch([(name, value)])

    def reg_read(self, name):
        "Read single register"
        return self.reg_batch([(name, None)])[0]

    def reg_batch(self, cmds):
        """Batch read/write operations.
        
        :param list cmd: List of tuples [("regname", value|None)].  None triggers read.
        :returns list: List of value (for reads) or None (writes)
        """
        # squash RegName into string
        cmds = [(str(name), value) for name,value in cmds]
        return self._reg_batch(cmds)

    def set_tgen(self, tbl, instance=0):
        """Load waveform to PRC tgen
        
        Provided tbl must be an Nx3 array
        of delay (in ticks) and 2x levels (I/Q in counts).
        First delay must be zero.
        """
        _log.debug("tgen input table %s", tbl)
        assert tbl.shape[1]==3, tbl.shape

        assert tbl[0,0]==0, ('Table must start with delay 0', tbl[0,0])
        tbl[:-1,0] = tbl[1:,0]
        tbl[-1,0] = 0

        info = self.get_reg_info(RegName('shell')[instance]['dsp_fdbk_core_mp_proc_lim'])
        assert info['addr_width'] >= 2, info
        # lim has 4 addresses to set bounds on X and Y (aka. I and Q)
        # when high==low the output will be forced to the limit
        # aka. feedforward

        lim_X_hi = info['base_addr']
        lim_Y_hi = lim_X_hi + 1
        lim_X_lo = lim_X_hi + 2
        lim_Y_lo = lim_X_hi + 3

        XXX = RegName('tgen')[instance]['delay_pc_XXX']
        info = self.get_reg_info(XXX)

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

        for i in range(0, idx, 4):
            print "INST", prg[i+0], prg[i+1], prg[i+2], prg[i+3]

        _log.debug("XXX program %s", prg[:idx])
        self.reg_write(XXX, prg)

class CADevice(Base):
    def __init__(self, addr, timeout=1.0):
        Base.__init__(self)
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
        self._regs = json.loads(zlib.decompress(caget(addr+'ctrl:JSON-I')))

    def pv_write(self, suffix, value):
        _log.debug('caput("%s", %s)', self.prefix+suffix, value)
        caput(self.prefix+suffix, value, wait=True, timeout=self.timeout)

    def get_reg_info(self, name):
        return self._regs[str(name)]

    def _reg_batch(self, cmds):
        ret = []
        for name, value in cmds:
            try:
                info = self._info[name]
            except ValueError:
                raise ValueError("Unknown register '%s'"%name)

            if value is None:
                pvname = str(info['input'])
                _log.debug('caput("%s.PROC", 1)', pvname)
                caput(pvname+'.PROC', 1, wait=True, timeout=self.timeout)
                value = caget(pvname, timeout=self.timeout)
                _log.debug('caget("%s") -> %s', pvname, value)

            else:
                pvname = str(info['output'])
                _log.debug('caput("%s", %s)', pvname, value)
                caput(pvname, value, wait=True, timeout=self.timeout)

            ret.append(value)
        return ret

    class CAAcq(object):
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

        def __enter__(self):
            return self
        def __exit__(self, A, B, C):
            self.close()

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


class LEEPDevice(Base):
    pass
