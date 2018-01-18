
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

        self._S = None
        self._S_pv = None
        self._E = Event()

    def close(self):
        if self._S is not None:
            self._S.close()
            self._S = None

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

    def set_channel_mask(self, chans=None, instance=[]):
        """Enabled specified channels.
        chans may be a bit mask or a list of channel numbers
        """
        I = self.instance + instance
        # assume that the shell_#_ number is the first

        chans = set(chans)
        caput(['%sacq:dev%d:ch%d:Ena-Sel'%(self.prefix, I[0], ch) for ch in range(12)],
              ['Enable' if ch in chans else 'Disable' for ch in range(12)],
              wait=True)

    def wait_for_acq(self, tag=False, timeout=5.0, instance=[]):
        """Wait for next waveform acquisition to complete.
        If tag=True, then wait for the next acquisition which includes the
        side-effects of all preceding register writes
        """
        I = self.instance + instance
        # assume that the shell_#_ number is the first

        if tag:
            # subscribe to last tag to get updates only when a new tag comes into effect
            pv = '%sacq:dev%d:Tag2-I'%(self.prefix, I[0])
            # increment tag
            caput('%sacq:dev%d:TagInc-Cmd'%(self.prefix, I[0]), 1, wait=True)
            old = caget('%sacq:dev%d:Tag-RB'%(self.prefix, I[0]))

        else:
            # subscribe to acquisition counter to get all updates
            pv = '%sacq:dev%d:AcqCnt-I'%(self.prefix, I[0])

        if self._S_pv != pv:
            self.close()

            self._S = camonitor(pv, self._E.Signal, format=FORMAT_TIME)
            # wait for initial update
            self._S.Wait(timeout=timeout)

            self._S_pv = pv

        else:
            self._S.Reset()

        while True:
            V = self._S.Wait(timeout=timeout)

            dT = (V - old)%0xffff

            if not tag or dT>=0:
                break

    def get_channels(self, chans=[], instance=[]):
        """:returns: a list of :py:class:`numpy.ndarray` with the numbered channels.
        chans may be a bit mask or a list of channel numbers
        """
        I = self.instance + instance
        return caget(['%sacq:dev%d:ch%d:I-I'%(self,prefix, I[0], ch) for ch in chans])
