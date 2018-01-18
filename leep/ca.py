
import logging
_log = logging.getLogger(__name__)

import json, zlib

import numpy

from .base import DeviceBase, AcquireBase, IGNORE, WARN, ERROR

caget = caput = camonitor = None
try:
    from cothread.catools import caget, caput, camonitor, FORMAT_TIME, DBR_CHAR_STR
except ImportError:
    pass
else:
    from cothread import Event


class CADevice(DeviceBase):
    backend = 'ca'

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
            if instance is not None:
                name = self.expand_regname(name, instance=instance)
            info = self._info[name]
            pvname = str(info['output'])
            _log.debug('caput("%s", %s)', pvname, value)
            caput(pvname, value, wait=True, timeout=self.timeout)

    def reg_read(self, names, instance=[]):
        ret = [None]*len(names)
        for i, name in enumerate(names):
            if instance is not None:
                name = self.expand_regname(name, instance=instance)
            info = self._info[name]
            pvname = str(info['input'])

            _log.debug('caput("%s.PROC", 1)', pvname)
            caput(pvname+'.PROC', 1, wait=True, timeout=self.timeout)
            # force as unsigned
            ret[i] = caget(pvname, timeout=self.timeout)
            # cope with lack of unsigned in CA
            info = self.regmap[name]
            if info.get('sign', 'unsigned')=='unsigned':
                ret[i] &= (2**info['data_width'])-1
            _log.debug('caget("%s") -> %s', pvname, ret[i])

        return ret

    @property
    def descript(self):
        return caget(str(self.prefix+'ctrl:Desc-I'), datatype=DBR_CHAR_STR)
    @property
    def codehash(self):
        return caget(str(self.prefix+'ctrl:CodeHash-I'), datatype=DBR_CHAR_STR)
    @property
    def jsonhash(self):
        return '<not implemented>'

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

    def acquire(self, instance=0):
        """Setup for acquisition
        """
        return self.CAAcq(self, 'shell_%d_'%instance)
