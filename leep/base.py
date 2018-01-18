
import logging
_log = logging.getLogger(__name__)

import json, zlib, re

import numpy

# flags for _wait_acq
IGNORE="IGNORE"
WARN="WARN"
ERROR="ERROR"

def open(addr, **kws):
    """Access to a single LEEP Device.
    
    :param str addr: Device Address.  prefix with "ca://<prefix>" or "leep://<ip>[:<port>]"
    """
    if addr.startswith('ca://'):
        from cothread.catools import caget
        if caget is None:
            raise RuntimeError('ca:// not available, cothread module not in PYTHONPATH')
        from .ca import CADevice
        return CADevice(addr[5:], **kws)

    elif addr.startswith('leep://'):
        from .raw import LEEPDevice
        return LEEPDevice(addr[7:], **kws)

    else:
        raise ValueError("Unknown '%s' must begin with ca:// or leep://"%addr)

class AcquireBase(object):
    def __init__(self, device):
        self.device = device
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
    backend = None # 'ca' or 'leep'

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
            if R.match(reg) is not None:
                if ret is not None:
                    raise RuntimeError('%s %s Matched more than one register %s, %s'%(name, I, ret, reg))
                ret = reg
        if ret is None:
            raise RuntimeError('No match for register pattern %s'%R.pattern)
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
        if instance is not None:
            name = self.expand_regname(name, instance=instance)
        return self.regmap[name]

    def set_tgen(self, tbl, instance=[]):
        """Load waveform to RFS tgen
        
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
