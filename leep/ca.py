
import logging
_log = logging.getLogger(__name__)

import json
import zlib

import numpy

from .base import DeviceBase, IGNORE, WARN, ERROR

caget = caput = camonitor = None
try:
    from cothread.catools import caget as _caget, caput as _caput, camonitor, FORMAT_TIME, DBR_CHAR_STR
except ImportError:
    pass
else:
    from cothread import Event

def caget(*args, **kws):
    R = _caget(*args, **kws)
    _log.debug('caget(%s, %s) -> %s' % (args, kws, R))
    return R

def caput(*args, **kws):
    _log.debug('caput(%s, %s' % (args, kws))
    _caput(*args, **kws)

class CADevice(DeviceBase):
    backend = 'ca'

    def __init__(self, addr, timeout=5.0, **kws):
        DeviceBase.__init__(self, **kws)
        self.timeout = timeout
        assert self.timeout>0.0, self.timeout
        self.prefix = addr  # PV prefix

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
        # hack, detect how many format specs are present
        N, I = 0, 0
        while True:
            I = suffix.find('%', I)
            if I == -1:
                break
            I += 1 # skip past found charactor
            N += 1

        name = str(self.prefix + (suffix % tuple(self.instance[:N])))
        caput(name, value, wait=True, timeout=self.timeout)

    def reg_write(self, ops, instance=[]):
        for name, value in ops:
            if instance is not None:
                name = self.expand_regname(name, instance=instance)
            info = self._info[name]
            pvname = str(info['output'])
            caput(pvname, value, wait=True, timeout=self.timeout)

    def reg_read(self, names, instance=[]):
        ret = [None]*len(names)
        for i, name in enumerate(names):
            if instance is not None:
                name = self.expand_regname(name, instance=instance)
            info = self._info[name]
            pvname = str(info['input'])

            caput(pvname+'.PROC', 1, wait=True, timeout=self.timeout)
            # force as unsigned
            ret[i] = caget(pvname, timeout=self.timeout)
            # cope with lack of unsigned in CA
            info = self.regmap[name]
            if info.get('sign', 'unsigned') == 'unsigned':
                ret[i] &= (2**info['data_width'])-1

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

    def set_decimate(self, dec):
        assert dec>=1 and dec<=255
        self.pv_write('acq:dev%s:Dec-SP', dec)

    def set_channel_mask(self, chans=None, instance=[]):
        """Enabled specified channels.
        chans may be a bit mask or a list of channel numbers
        """
        I = self.instance + instance
        # assume that the shell_#_ number is the first

        chans = set(chans)
        disable = set(range(12)) - chans
        # enable/disable for even/odd channels are actually aliases
        # so disable first, then enable
        if disable:
            caput(['%sacq:dev%s:ch%d:Ena-Sel' % (self.prefix, I[0], ch) for ch in disable], 'Disable', wait=True)
        if chans:
            caput(['%sacq:dev%s:ch%d:Ena-Sel' % (self.prefix, I[0], ch) for ch in chans], 'Enable', wait=True)

    def wait_for_acq(self, tag=False, timeout=5.0, instance=[]):
        """Wait for next waveform acquisition to complete.
        If tag=True, then wait for the next acquisition which includes the
        side-effects of all preceding register writes
        """
        I = self.instance + instance
        # assume that the shell_#_ number is the first

        if tag:
            # subscribe to last tag to get updates only when a new tag comes into effect
            pv = '%sacq:dev%s:Tag2-I' % (self.prefix, I[0])
            # increment tag
            caput('%sacq:dev%s:TagInc-Cmd' % (self.prefix, I[0]), 1, wait=True)
            old = caget('%sacq:dev%s:Tag-RB' % (self.prefix, I[0]))

        else:
            # subscribe to acquisition counter to get all updates
            pv = '%sacq:dev%s:AcqCnt-I' % (self.prefix, I[0])

        if self._S_pv != pv:
            if self._S is not None:
                self._S.close()
            self._E.Reset()

            self._S = camonitor(pv, self._E.Signal, format=FORMAT_TIME)
            # wait for, and consume, initial update
            self._E.Wait(timeout=timeout)

            self._S_pv = pv

        else:
            self._E.Reset()

        while True:
            V = self._E.Wait(timeout=timeout)

            if not tag:
                break
            dT = (V - old) % 0xffff
            if dT >= 0:
                break

    def get_channels(self, chans=[], instance=[]):
        """:returns: a list of :py:class:`numpy.ndarray` with the numbered channels.
        chans may be a bit mask or a list of channel numbers
        """
        I = self.instance + instance
        ret = caget(['%sacq:dev%s:ch%d:I-I' % (self.prefix, I[0], ch) for ch in chans], format=FORMAT_TIME)
        if len(ret) >= 2 and not all([ret[0].raw_stamp == R.raw_stamp for R in ret[1:]]):
            raise RuntimeError("Inconsistent timestamps! %s" % [R.raw_stamp for R in ret])
        return ret

    def get_timebase(self, chans=[], instance=[]):
        I = self.instance + instance
        ret = caget(['%sacq:dev%s:ch%d:T-I' % (self.prefix, I[0], ch) for ch in chans], format=FORMAT_TIME)
        if len(ret) >= 2 and not all([ret[0].raw_stamp == R.raw_stamp for R in ret[1:]]):
            raise RuntimeError("Inconsistent timestamps! %s" % [R.raw_stamp for R in ret])
        return ret
