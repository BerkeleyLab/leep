
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

    Device addresses take the form of "ca://<prefix>" or "leep://<ip>[:<port>]".

    "ca://" addresses take a Process Variable (PV) name prefix string.

    "leep://" addresses are a hostname or IP address, with an optional port number.

    >>> from leep import open
    >>> dev = open('leep://localhost')

    or

    >>> dev = open('ca://TST:')

    :param str addr: Device Address.
    :param float timeout: Communications timeout.
    :param list instance: List of instance identifiers.
    :returns: :py:class:`base.DeviceBase`
    """
    if addr.startswith('ca://'):
        try:
            from cothread.catools import caget
        except ImportError:
            raise RuntimeError('ca:// not available, cothread module not found in PYTHONPATH')
        from .ca import CADevice
        return CADevice(addr[5:], **kws)

    elif addr.startswith('leep://'):
        from .raw import LEEPDevice
        return LEEPDevice(addr[7:], **kws)

    else:
        raise ValueError("Unknown '%s' must begin with ca:// or leep://"%addr)

class DeviceBase(object):
    backend = None # 'ca' or 'leep'

    def __init__(self, instance=[]):
        self.instance = instance[:] # shallow copy

    def close(self):
        pass

    def __enter__(self):
        return self
    def __exit__(self, A, B, C):
        self.close()


    def expand_regname(self, name, instance=[]):
        """Return a full register name from the short name and optionally instance number(s)

        >>> D.expand_regname('XXX', instance=[0])
        'tget_0_delay_pc_XXX'
        """
        # build a regexp
        I = self.instance + instance + [name]
        I = r'_(?:.*_)?'.join([re.escape(str(i)) for i in I])
        R = re.compile('^.*%s$'%I)

        if name in self.regmap:
            return name

        ret = filter(R.match, self.regmap)
        if len(ret)==1:
            return ret[0]
        elif len(ret)>1:
            raise RuntimeError('%s Matches more than one register: %s'%(R.pattern, ' '.join(ret)))
        else:
            raise RuntimeError('No match for register pattern %s'%R.pattern)

    def reg_write(self, ops, instance=[]):
        """Write to registers.

        :param list ops: A list of tuples of register name and value.
        :param list instance: List of instance identifiers.

        Register names are expanded using self.instance and the instance lists.

        >>> D.reg_write([
            ('reg_a', 5),
            ('reg_b', 6),
        ])
        """
        raise NotImplementedError

    def reg_read(self, names, instance=[]):
        """Read from registers.

        :param list names: A list of register names.
        :param list instance: List of instance identifiers.
        :returns: A :py:class:`numpy.ndarray` for each register name.

        >>> A, B = D.reg_read(['reg_a', 'reg_b'])
        """
        raise NotImplementedError

    def get_reg_info(self, name, instance=[]):
        """Return a dict describing the named register.
        This dictionary is passed through from the information read from the device ROM.

        Common dict keys are:

        * access
        * base_addr
        * addr_width
        * data_width
        * sign
        * description

        :param str name: A register name
        :param list instance: List of instance identifiers.
        :returns: A dict with string keys.
        """
        if instance is not None:
            name = self.expand_regname(name, instance=instance)
        return self.regmap[name]

    def set_channel_mask(self, chans=None, instance=[]):
        """Enabled specified channels.
        chans may be a bit mask or a list of channel numbers (zero indexed).

        :param list chans: A list of channel integer numbers (zero indexed).
        :param list instance: List of instance identifiers.
        """
        raise NotImplementedError

    def wait_for_acq(self, tag=False, timeout=5.0, instance=[]):
        """Wait for next waveform acquisition to complete.
        If tag=True, then wait for the next acquisition which includes the
        side-effects of all preceding register writes

        ;param bool tag: Whether to use the tag mechanism to wait for a update
        :param float timeout: How long to wait for an acquisition.  Seperate from the communications timeout.
        :param list instance: List of instance identifiers.
        """
        raise NotImplementedError

    def get_channels(self, chans=[], instance=[]):
        """:returns: a list of :py:class:`numpy.ndarray` with the numbered channels.
        chans may be a bit mask or a list of channel numbers.

        The returned arrays have been scaled.
        """
        raise NotImplementedError

    def get_timebase(self, instance=[]):
        """Return an array of times for each sample returned by :py:meth:`get_channels`.

        Note that this timebase may be one sample longer than some data arrays
        when the number of channels selected is not a power of 2.
        """
        raise NotImplementedError

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
