import datetime
import sys
import os
import json
import numpy
import codecs
import logging
import time
from sel_waves import interpret_slow_data
from fitter import fitter, create_basist, fpga_regs
from cav_energy import rev_area, cav_eval

_log = logging.getLogger(__name__)

# Grab the start time early, so things like
# python beg2.py | tee `date "+%Y%m%d_%H%M%S"`.log
# will usually get a timestamp that matches
start_time = datetime.datetime.now()


def angle_shift(angle, shift):
    a = angle+shift
    if (a > 262144):
        a -= 262144
    if (a < 0):
        a += 262144
    return a


# Documents scaling of our second-order CIC filters programmed with
# a sampling interval that's a multiple of wave_samp_per, followed by
# a barrel shift right of 2*wave_shift bits.  Based on a 20-bit
# waveform readout (see iq_buf_collect() in sel_waves.v).
# Valid range of wave_samp_per is [0, 256], but we don't use 256
# in real life, because it gets confusingly encoded as a value of 0
# in the 8-bit wave_samp_per register.  The wave_shift register should
# match the computation here; if not, you can override it with the
# python keyword arg: valid values are [0, 7].
def adc_config(wave_samp_per, wave_shift=None):
    # prefactor is conceptually 1.0, but see comments in llrf_dsp.v
    prefactor = 74762*(33/32.0)**2*1.646760258*0.5**17
    if wave_shift is None:
        wave_shift = max(0, int(numpy.ceil(numpy.log2(wave_samp_per)))-1)
    adc_fs = 2**19 * prefactor * wave_samp_per**2 / 4.0**(wave_shift+1)
    return adc_fs, wave_shift


class c_rf_controls:

    def __init__(self, conf, save_all_buffers=False):

        import leep

        logging.basicConfig(level=conf['debug'])

        for key in conf:
            setattr(self, key, conf[key])

        # Open log file
        try:
            self.logfile = open(self.data_dir + "/beg.log", "a")
            UTF8Writer = codecs.getwriter('utf8')
            self.logfile = UTF8Writer(self.logfile)
        except ValueError:
            print("Failed opening log file: %s", self.data_dir + "/beg.log")

        # Open communication with the RFS
        _log.info('RFS address %s [%s]', self.rfs_addr, self.zone)
        self.rfs = leep.open(self.rfs_addr, instance=[self.zone])

        # Open communication with PRC if
        # needed for 8pi/9 mode calibration
        if self.prc_ip is not None:
            self.prc = leep.open('leep://' + self.prc_ip, instance=[0])
        else:
            self.prc = None

        # Constants
        if self.loop_is_cav:
            self.CAV = 0
            self.FWD = 1
            self.REV = 2
            self.DRV = 3
        else:
            self.FWD = 0
            self.REV = 1
            self.CAV = 2
            self.DRV = 3
        self.save_all_buffers = save_all_buffers
        self.slow_abi_ver = 0
        try:
            self.slow_abi_ver = self.rfs.regmap["__metadata__"]["slow_abi_ver"]
        except KeyError:
            pass

        self.dt = 1  # fake for now
        self.tick_multiply = 1  # fake for now
        self.pulse_bank = 0  # not perfect
        # query number stored in tag memory
        #   0  reserved for never used
        #   1  reserved for parameter update in progress
        #   2  reserved for process stopped
        #   3  unused
        #  4-11 rotated through
        # I use 4 bits here, even though the hardware is capable of 8
        self.qnum = 0

        # wave_samp_per = 1 works well for 8 kHz CMOC simulation or 7 kHz hardware (crystal) simulator,
        # expect wave_samp_per = 256 to work well for actual SRF cavity
        # Actual value set in json file or overridden on command line.
        self.adc_fs, self.wave_shift = adc_config(self.wsp)

        self.cic_base_period = 33  # default parameter in llrf_dsp.v
        self.Tstep = 14./1320e6
        self.tick_multiply = self.cic_base_period * 2 * self.wsp
        # Factor of 2 comes from default use_hb=1 in llrf_dsp instantiation of ccfilt
        self.dt = self.tick_multiply * self.Tstep   # seconds
        self.start = -1
        self.piezo_dc = 1000
        self.in_level = 8000
        self.digaree_Tstep = 32*33*self.Tstep  # baked into piezo_control.v

        self.imag_clip = int(round(79500 * self.dmax_magn * self.dmax_imag))
        self.real_clip = int(round(79500 * self.dmax_magn * numpy.sqrt(1.0 - self.dmax_imag**2)))
        self.out_level_goal = self.cav_goal / self.cav_fs  # as fraction of cavity ADC full-scale
        self.fwd_scale = numpy.sqrt(self.fwd_fs*1000)  # convert dimensionless fraction of full-scale to sqrt(W)
        self.rev_scale = numpy.sqrt(self.rev_fs*1000)  # convert dimensionless fraction of full-scale to sqrt(W)
        self.cav_scale = self.cav_fs  # convert dimensionless fraction of full-scale to MV/m

        # Basic controller setup
        self.sel_set = [('sel_en', 1)]
        self.base_set = [
            ('sel_en', 0),
            ('ph_offset', 0),
        ]
        self.base_set += self.simple_pulse(self.in_level, 240)

        self.log("Run configuration:", stdout=True)
        if self.desc is not None:
            self.log("  Description %s" % self.desc, stdout=True)
        self.log("  RFS Address %s" % self.rfs_addr, stdout=True)
        if self.prc_ip is not None:
            self.log("  PRC IP %s" % self.prc_ip, stdout=True)
        self.log("  Config loop_is_cav %s" % self.loop_is_cav, stdout=True)
        self.log("  Config wsp %d" % self.wsp, stdout=True)
        self.log("  Config out_level_goal %.5f" % self.out_level_goal, stdout=True)
        self.log("  Config real_clip %d" % self.real_clip, stdout=True)
        self.log("  Config imag_clip %d" % self.imag_clip, stdout=True)
        self.log("  Config channels %d %d" % (self.CAV, self.FWD), stdout=True)
        self.log("  Config adc_fs %.1f" % self.adc_fs, stdout=True)

    def configure_fpga(self):

        # Initial setup for waveform acquisition
        self.rfs.reg_write([
            ('bank_next', 0),
        ])
        self.rfs.set_decimate(self.wsp)
        self.rfs.set_channel_mask(range(2, 10))

        # Initial setup is for loopback, forward, reverse, cavity.
        # Constants above reflect expectation for that to change,
        # to include drive instead of loopback,
        # before any real data analysis takes place.
        # Loopback tests are not possible/meaningful in the loop_is_cav case.
        self.detune_output_enable(False)

        # Write basic controller setup
        self.write_and_acquire(self.base_set)

    def triple_pulse_core(self, t1, t2, t3, level, maxq, span=0.2, cw=False):
        t1x = t1 * self.tick_multiply
        t2x = t2 * self.tick_multiply
        t3x = t3 * self.tick_multiply
        # print("triple_pulse  time setup %d %d %d  level %d" % (t1x, t2x, t3x, level))
        l = []

        # High and low limits for X and Y coordinates in the controller
        # are part of a single array in the FPGA (mp_proc_lim).
        # Mapping:
        #   lim_X_hi = mp_proc_lim_0
        #   lim_Y_hi = mp_proc_lim_1
        #   lim_X_lo = mp_proc_lim_2
        #   lim_Y_lo = mp_proc_lim_3

        if t1x > 0:
            l.extend([
                ('set', 'mp_proc_lim_3', 0),           # lim_Y_lo
                ('set', 'mp_proc_lim_1', 0),           # lim_Y_hi
                ('set', 'mp_proc_lim_2', level),       # lim_X_lo
                ('set', 'mp_proc_lim_0', level),       # lim_X_hi
                ('sleep', t1x),
            ])
        if t2x > 0:
            lmin = int(level*(1.0-span))
            lmax = int(level*(1.0+span))
            self.log("real clip limits %d %d" % (lmin, lmax), stdout=True)
            l.extend([
                ('set', 'mp_proc_lim_2', lmin),        # lim_X_lo
                ('set', 'mp_proc_lim_0', lmin),        # lim_X_hi
                ('sleep', t2x),
            ])
        if t3x > 0:
            l.extend([
                ('set', 'mp_proc_lim_3', 2**18-maxq),  # lim_Y_lo
                ('set', 'mp_proc_lim_1', maxq),        # lim_Y_hi
                ('sleep', t3x),
            ])
        if not cw:
            l.extend([
                ('set', 'mp_proc_lim_3', 0),           # lim_Y_lo
                ('set', 'mp_proc_lim_1', 0),           # lim_Y_hi
                ('set', 'mp_proc_lim_2', 0),           # lim_X_lo
                ('set', 'mp_proc_lim_0', 0),           # lim_X_hi
            ])

        return l

    # Parallels the end state of triple_pulse, but bypasses tgen.
    # Used to sync the register mirror with reality.
    def static_box(self, level, maxq, span=0.2):
        lmin = int(level*(1.0-span))
        lmax = int(level*(1.0+span))
        minq = 2**18-maxq
        lim = [lmax, maxq, lmin, minq]  # lim_X_hi, lim_Y_hi, lim_X_lo, lim_Y_lo
        write_list = [('mp_proc_lim_%d' % ix, value) for ix, value in enumerate(lim)]
        return write_list

    def triple_pulse(self, t1, t2, t3, level, maxq, span=0.2, cw=False):
        prog = self.triple_pulse_core(t1, t2, t3, level, maxq, span=span, cw=cw)
        return self.rfs.tgen_reg_sequence(prog)

    def simple_pulse(self, level, pulse_len):
        return self.triple_pulse(pulse_len, 0, 0, level, 0)

    def save_data_file(self, res, datetimestr, aux):
        isd = interpret_slow_data(aux, abi_ver=self.slow_abi_ver)
        header = datetimestr + " %d %d" % (isd[0], isd[5])  # circle_count and time_stamp
        fname = "%s/auto_%3.3d.dat" % (self.data_dir, self.qnum)
        self.qnum += 1
        numpy.savetxt(fname, numpy.asarray(res).T, fmt="%.7f", header=header)
        self.log("Wrote file %s" % fname)
        self.dsp_status = isd[6]

    def dsp_status_check_clipped(self, clip):
        clip_ok = (self.dsp_status >> 8) == (0xf if clip else 0)
        self.log("dsp_status 0x%3.3x  %s" % (self.dsp_status, "OK" if clip_ok else "??"), stdout=True)

    def write_and_acquire(self, query):

        # Write the query and wait for a full buffer where
        # the register write has taken effect
        self.rfs.reg_write(query)

        # Manage mixed case where external tag checking might be needed
        toggle_tag = self.save_all_buffers
        tag = not self.save_all_buffers

        # Get 8 channels
        channel_mask = range(2, 10)
        self.rfs.set_channel_mask(channel_mask)

        while True:
            tag_match, slow_data, time_now = self.rfs.wait_for_acq(tag=tag, toggle_tag=toggle_tag)
            datetimestr = time_now.isoformat()+"Z"
            channels = self.rfs.get_channels(channel_mask)
            toggle_tag = False

            if tag_match:
                # quick, save the data!
                self.save_data_file(channels, datetimestr, slow_data)
                break
            elif self.save_all_buffers:
                # Save the data and keep going
                self.save_data_file(channels, datetimestr, slow_data)

        self.data_array = [(channels[ix]+1j*channels[ix+1]) for ix in range(0, 8, 2)]
        # for ch in self.data_array:
        # print ch.min(), ch.max(), ch.dtype, ch.shape

    def check_fwd(self, verbose=True):
        fwdm = abs(self.data_array[self.FWD])  # forward wave magnitude
        revm = abs(self.data_array[self.REV])  # reverse wave magnitude
        mx = max(fwdm)
        ex = max(numpy.nonzero(fwdm > 0.5*mx)[0])
        fwd_watts = (mx*self.fwd_scale)**2
        self.fwd_watts = fwd_watts
        self.rev_watts = (max(revm)*self.rev_scale)**2
        self.log("check_fwd: magnitude %.5f (%.3f kW), end of pulse at %d" % (mx, 1e-3*fwd_watts, ex), stdout=verbose)
        return ex

    def check_rev(self, ex):
        data = self.data_array
        fwdm = abs(data[self.FWD])
        revm = abs(data[self.REV])
        cavm = abs(data[self.CAV])
        pf = (fwdm*self.fwd_scale)**2
        pr = (revm*self.rev_scale)**2
        if ex > 1024-74 or max(cavm) < 0.03:
            self.log('check_rev: insufficient decay data, ex=%d, max_cav=%.5f' % (ex, max(cavm)))
            return
        # Analysis takes place in units of Watts and unitless time step
        try:
            r, raw_area, raw_bw = rev_area(pf, pr, ex, plot=False, verbose=False)
        except:
            self.log('check_rev: exception in rev_area()')
            raise
        if r == 0:
            self.log('check_rev: rev_area() returned fault')
            return
        dt = 2*self.wsp*33*14/1320e6
        bw = raw_bw/dt/(2*numpy.pi)
        self.log('Edge detected at %d, refined to %.2f, bandwidth %.2f Hz' % (ex, r, bw))
        energy = raw_area * dt  # Joules
        gconvert = 2.875  # MV/sqrt(J),  sqrt(RoverQ*(f0*2*pi))
        book_l = 1.038  # m, cavity length, arguably fiction
        gradient = numpy.sqrt(energy) * gconvert / book_l
        self.log('Exponential area %.2f Joules, %.2f MV/m' % (energy, gradient))
        cav0 = cav_eval(cavm, r)
        cav_scale = gradient/cav0
        plist = cav0, cav_scale, self.cav_scale
        self.log('Cavity at that point measured %.4f, scaling %.2f MV/m (given %.2f MV/m)' % plist)

    # arange for amplitude printout
    # prange for phase fitting (frequency offset)
    # drange for log amplitude fitting (decay time)
    def find_slope(self, arange, prange, drange, verbose=True):
        cav = self.data_array[self.CAV]
        phase = numpy.angle(cav)
        y = numpy.unwrap(phase[prange])
        ix = range(len(y))
        pp = numpy.polyfit(ix, y, 1)
        delta_f = pp[0]/self.dt/2.0/numpy.pi
        # print("find_slope dt", dt)
        # print("debug", self.CAV, min(drange), max(drange), len(drange))
        amp = numpy.abs(cav)
        y = numpy.log(amp[drange])
        ix = range(len(y))
        pp = numpy.polyfit(ix, y, 1)
        bw = -pp[0]/self.dt/2.0/numpy.pi  # Hz
        max_amp = max(amp[arange])
        self.log("Measured bandwidth %.6f kHz, detune %.6f kHz, max amp %.5f" % (bw/1000, delta_f/1000, max_amp), stdout=verbose)
        self.delta_f = delta_f
        self.bandwidth = bw
        return (delta_f, bw, max_amp)

    def usual_find_slope(self, verbose=True):
        s = self.start
        slp = self.find_slope(range(0, s), range(s, s+50), range(s, s+50), verbose=verbose)
        self.check_rev(s)
        return slp

    def find_ph_offset(self):
        cav = self.data_array[self.CAV]
        # XXX range(5, 25) better on SRF, range(10, 30) better on emulator
        ix = range(10, 30)  # throw away first few points
        phase = numpy.unwrap(numpy.angle(cav))[ix]
        ixx = numpy.array(ix)-1  # extrapolate back to second time step
        pp = numpy.polyfit(ixx, phase, 1)
        pp_fit = numpy.polyval(pp, ixx)
        pp_err = numpy.sqrt(numpy.mean((phase-pp_fit)**2))
        self.log("Linear phase fit coefficients (%.3f %.3f), rms error %.1f degrees" %
                 (pp[0], pp[1], pp_err*180/numpy.pi), stdout=True)
        phase_orig = pp[1]
        use_ph_offset = numpy.pi + phase_orig
        self.log("uncorrected use_ph_offset %.4f" % use_ph_offset, stdout=True)
        if self.CAV != 3:
            use_ph_offset -= 1.3738  # see fdbk_phase.sh
        use_ph_offset += numpy.pi*0.5   # XXX total hack
        while use_ph_offset < -numpy.pi:
            use_ph_offset += 2*numpy.pi
        while use_ph_offset > numpy.pi:
            use_ph_offset -= 2*numpy.pi
        reg_ph_offset = int(round(use_ph_offset*262144/(2*numpy.pi)))
        self.log("found cavity phase %.1f degrees, guess register ph_offset should be %d" %
                 (phase_orig*180/numpy.pi, reg_ph_offset), stdout=True)
        return reg_ph_offset

    def find_model(self):
        cav = self.data_array[self.CAV]
        fwd = self.data_array[self.FWD]
        basist = create_basist(block=10, n=24)
        # print basist.shape
        ny = basist.shape[0]+1
        mr, rank, fitc = fitter(cav[4:ny+4], fwd[4:ny+4], basist=basist, dt=self.dt)
        sf = fpga_regs(mr, fq=self.detune_fq, dt=self.digaree_Tstep)
        self.log("model %.3f%+.3fj /s  fq=%f  %s" % (mr.real, mr.imag, self.detune_fq, repr(sf)))
        return mr

    def handle_mr_set(self, mr_set):
        mr_amp = numpy.median(numpy.abs(mr_set))
        mr_pha = numpy.median(numpy.angle(mr_set))
        mr_print = len(mr_set), mr_amp, mr_pha*180/numpy.pi
        self.log(u"handle_mr_set: %d points, median mag %.2f, phase %5.1f\u00b0" % mr_print, stdout=True)
        mr = mr_amp * numpy.exp(1j*mr_pha)
        sf = fpga_regs(mr, fq=self.detune_fq, dt=self.digaree_Tstep)
        self.log("model %.3f%+.3fj /s  fq=%f  %s" % (mr.real, mr.imag, self.detune_fq, repr(sf)))
        return sf
        # This is the result that should be sent to the hardware

    def push_digaree_coeff(self, sf):
        self.log("Writing detune coefficients to hardware: %s" % repr(sf), stdout=True)
        # Build the register write commands

        # Push the writes to the FPGA
        self.rfs.reg_write([('piezo_sf_consts', sf + [0]*4)])
        self.detune_output_enable(True)

    # Boolean True/False to enable/disable "valid" bit on RFS fiber output
    # application_top.v defines this as cct1_cavity{0,1}_status_aux[12]
    def detune_output_enable(self, b):
        # I don't think hierarchy can handle this one
        auxa = 'cct1_cavity%d_detune_en' % self.zone
        auxv = 1 if b else 0
        self.rfs.reg_write([(auxa, auxv)])

    def conf_lp_notch(self):
        from notch_setup import notch_setup

        ns_cav = notch_setup(bw=self.lp_bw, notch=self.notch_f)
        notch_base = 'shell_{0}_dsp_lp_notch_'.format(self.zone)
        notch_dict = ns_cav.dict(notch_base)

        ss = []
        for key, value in notch_dict.iteritems():
            for ix in range(len(value)):
                ss.append((key+'_{0}'.format(ix), value[ix]))

        self.log("\nWriting low-pass and notch filter coefficients onto hardware:", stdout=True)
        self.log("  Low-pass bw %.1f kHz" % (self.lp_bw/1e3), stdout=True)
        if self.notch_f is not None:
            self.log("  Notch frequency %.1f kHz" % (self.notch_f/1e3), stdout=True)
        else:
            self.log("  No notch filter present", stdout=True)
        self.log(str(ss))
        self.rfs.reg_write(ss)

    def find_level(self, verbose=True):
        cav = self.data_array[self.CAV]
        amp = numpy.abs(cav)
        s = self.start
        amp_avg = numpy.mean(amp[range(s-20, s-5)])
        out_mvpm = amp_avg * self.cav_scale
        self.log("Measured amplitude %.5f (%.3f MV/m)" % (amp_avg, out_mvpm), stdout=verbose)
        return amp_avg

    def analyze_loopback(self, check=False):
        y = abs(self.data_array[0])
        y_on_mean = numpy.mean(y[20:220])
        y_off_mean = numpy.mean(y[280:480])
        y_on_std = numpy.std(y[20:220])
        y_off_std = numpy.std(y[280:480])
        y_high = numpy.nonzero(y > (y_on_mean*0.5))[0]
        y_high_start = min(y_high)
        y_high_end = max(y_high)
        self.log("Loopback analysis:", stdout=True)
        self.log("  Loopback mean on/off = %.6f/%.6f" % (y_on_mean, y_off_mean), stdout=True)
        self.log("  Loopback std. on/off = %.6f/%.6f" % (y_on_std, y_off_std), stdout=True)
        self.log("  Loopback pulse start, end = %d, %d" % (y_high_start, y_high_end), stdout=True)
        if not check:
            return True  # no checks
        loop_ok = y_on_mean > 0.009314 and y_on_mean < 0.01378 and y_off_mean < 8e-6 and y_on_std < y_on_mean*2.5e-4 and y_off_std < 3e-6
        ls = "PASS" if loop_ok else "FAIL"
        self.log("Loopback test finished: %s" % ls, stdout=True)
        return loop_ok

    def offset_fit_quadratic(self, in_val, out_level):
        pp = numpy.polyfit(in_val, out_level, 2)
        self.log("pp = " + repr(pp))
        if pp[0] >= 0:
            self.log("Positive curvature, aborting!", stdout=True)
            return 0, 0, False
        center = -pp[1]/(2*pp[0])
        ant_peak = numpy.polyval(pp, center)
        return center, ant_peak, True

    def offset_fit_trig(self, in_val, out_level, include_pedestal=False):
        x = numpy.array(in_val) * 0.5**18 * 2 * numpy.pi
        y = numpy.array(out_level)
        basis = [numpy.cos(x), numpy.sin(x)]
        if include_pedestal:
            basis = basis + [1.0+0*x]
        basis = numpy.vstack(basis).T
        fitc, resid, rank, sing = numpy.linalg.lstsq(basis, y, rcond=-1)
        err = y - basis.dot(fitc)
        rms = numpy.sqrt(numpy.mean(err**2))
        z = fitc[0] + 1j*fitc[1]
        center = numpy.angle(z) * 2**18 / (2*numpy.pi)
        pedestal = 0
        print_suffix = ""
        if include_pedestal:
            pedestal = fitc[2]
            print_suffix = " (using pedestal)"
        valid_fit = abs(z) > 0.010 and rms < 0.2 * abs(z)
        if not valid_fit:
            print_suffix += " BAD!"
        self.log("offset_fit_trig: %.5f%+.5fi  rms error %.5f%s" % (z.real, z.imag, rms, print_suffix), stdout=True)
        if False:  # aid to understanding curve fit, also un-comment pyplot import above
            print("Pedestal %.5f" % pedestal)
            zp = numpy.exp(1j*x)*y
            pyplot.plot(zp.real, zp.imag, '*')
            zp = numpy.exp(1j*x)*basis.dot(fitc)
            pyplot.plot(zp.real, zp.imag)
            zcirc = numpy.exp(numpy.pi*2j*numpy.arange(0, 1.01, 0.01))
            zp = z*numpy.append(zcirc, 0)
            zp = zp+pedestal*numpy.exp(1j*numpy.angle(z))
            pyplot.plot(zp.real, zp.imag)
            pyplot.plot(0, 0, '*')
            pyplot.show()
        ant_peak = numpy.abs(z) + pedestal
        return center, ant_peak, valid_fit

    # Returns best estimate of ph_offset
    def analyze_poffset_scan(self, in_val, out_level, start_offset, check_range=True):
        self.log("x = " + repr(in_val))
        self.log("y = " + repr(out_level))
        mx = numpy.max(out_level)
        # important that this coefficient is less than cos(3*pi/8)/cos(pi/8) = 0.4142
        athresh = 0.15
        ix = numpy.nonzero(out_level > (mx*athresh))[0]
        if len(ix) < 3:
            self.log("Only %d points above mx*%.2f, aborting" % (len(ix), athresh), stdout=True)
            return None
        self.log("analyze_poffset_scan curve fit using %d of %d points" % (len(ix), len(out_level)), stdout=True)
        in_val_subset = [in_val[jx] for jx in ix]
        out_level_subset = [out_level[jx] for jx in ix]
        # center, ant_peak = self.offset_fit_quadratic(in_val_subset, out_level_subset)
        inc_ped = self.wsp < 4  # heuristic based on sel_theory2.m
        center, ant_peak, valid_fit = self.offset_fit_trig(in_val_subset, out_level_subset, include_pedestal=inc_ped)
        if not valid_fit:
            self.log("Invalid fit (wrong cavity signal?), aborting!", stdout=True)
            return None
        if check_range:
            self.log("Checking %d < %.1f < %d" % (min(in_val), center, max(in_val)), stdout=True)
            if center > max(in_val) or center < min(in_val):
                self.log("Minimum outside scan range, aborting!", stdout=True)
                return None
        # print center
        self.log("Shifting ph_offset by %.2f degrees, anticipated value %.5f" % (center*360/2**18, ant_peak), stdout=True)
        trial_offset = angle_shift(start_offset, center)
        self.log("Finally setting ph_offset to: %d" % trial_offset, stdout=True)
        return trial_offset

    def set_start(self):
        if self.start < 0:
            self.start = self.check_fwd()+4
            l = len(self.data_array[0])
            if self.start+50 > l:
                self.start = l - 50
            self.log("starting trailing waveform analysis at %d" % self.start, stdout=True)
            if self.start < 50 or self.start > 500:
                self.log("Aborting due to lack of reasonable trailing edge", stdout=True)
                self.shutdown_drive(terminate=True, code=2)

    def status_line(self):
        duty = self.start / 10.24  # hard-coded waveform length
        drv = self.in_level / 795.0
        fwd = numpy.sqrt(self.fwd_watts)
        rev = numpy.sqrt(self.rev_watts)
        cav = self.out_level * self.cav_scale
        p = duty, drv, fwd, rev, cav, self.bandwidth, self.delta_f
        return u"DF: %.0f%%  drv: %4.1f%%  fwd: %5.2f\u221aW  rev: %5.2f\u221aW  cav: %5.2f MV/m  decay BW: %5.2f Hz  det: %6.2f Hz" % p

    ########
    def coarse_tune_cavity0(self):
        vstep = 200
        slope = -100000000  # force first time through
        oldslope = slope
        real_keep = 0xfcc if self.loop_is_cav else 0x3fc
        self.log("\nCoarse-tune with SEL off", stdout=True)
        while abs(slope) > 8000:
            self.log("Piezo set %d" % self.piezo_dc, stdout=True)
            psu = self.piezo_dc+65536 if self.piezo_dc < 0 else self.piezo_dc
            # piezo_dc is no-op except for software cavity simulators
            # At some point this could shift to using steppers?
            query = self.base_set + [
                ('piezo_dc', psu),
            ]
            self.rfs.set_channel_mask(range(2, 10))
            self.write_and_acquire(query)
            self.set_start()
            slope, bw, max_amp = self.usual_find_slope()
            if bw < 0:
                self.log("Aborting due to negative bandwidth", stdout=True)
                self.shutdown_drive(terminate=True, code=2)
            if oldslope*slope < 0:  # direction change
                vstep = vstep/4
            self.piezo_dc += vstep*(1 if slope > 0 else -1)
            oldslope = slope

    ########
    def switch_to_sel0(self):
        # print data_array
        self.start = self.check_fwd()+4
        if self.start < 500 and self.start > 50:
            slope, bw, max_amp = self.usual_find_slope()
        else:
            self.log("Aborting due to lack of start of pulse in appropriate range", stdout=True)
            self.shutdown_drive(terminate=True, code=2)
        self.reg_ph_offset = self.find_ph_offset()
        # but then throw that away for now, because it hasn't been reliable
        pscan = []
        oscan = []
        snpt = 10
        for rotx in range(snpt):
            r2 = int(self.reg_ph_offset + rotx*262144/snpt + 0.5) % 262144
            r2_deg = r2 * 360 / 262144.0
            self.log(u"Full ph_offset scan %d/%d: %.1f\u00B0" % (rotx+1, snpt, r2_deg))
            op_str = u"  from: ph_off scan %d/%d: %5.1f\u00B0" % (rotx+1, snpt, r2_deg)
            self.write_and_acquire(self.sel_set + [('ph_offset', r2)])
            self.out_level = self.find_level(verbose=False)
            # self.log("  output level %.1f" % self.out_level, stdout=True)
            pscan += [r2]
            oscan += [self.out_level]
            self.usual_find_slope(verbose=False)
            self.check_fwd(verbose=False)
            self.find_model()
            self.log(self.status_line()+op_str, stdout=True)
        best_ix = oscan.index(max(oscan))
        self.log("Best index %d for maximum output %.5f" % (best_ix, max(oscan)))
        self.reg_ph_offset = (self.reg_ph_offset + best_ix*8192) % 262144
        self.reg_ph_offset = self.analyze_poffset_scan(pscan, oscan, 0.0, check_range=False)
        if self.reg_ph_offset is None:
            self.log("Coarse phase offset scan analysis failed, aborting", stdout=True)
            self.shutdown_drive(terminate=True, code=2)
        ph_offset_deg = self.reg_ph_offset * 360 / 262144.0
        self.log(u"Switching to SEL with ph_offset %.1f\u00B0" % ph_offset_deg, stdout=True)
        self.write_and_acquire(self.sel_set + [('ph_offset', self.reg_ph_offset)])

    ########
    def ramp_field0(self):
        self.log("\nRamping up field", stdout=True)
        self.out_level = 0
        while self.out_level < self.out_level_goal*0.84:
            out_mvpm = self.out_level * self.cav_scale
            self.log("Measured response %.5f (%.3f MV/m), will set new level %d" %
                     (self.out_level, out_mvpm, self.in_level))
            op_str = "  from: drive ramp"
            query = self.sel_set + self.simple_pulse(self.in_level, 240)
            self.write_and_acquire(query)
            # [['amp_max',in_level]])
            self.out_level = self.find_level(verbose=False)
            self.usual_find_slope(verbose=False)
            self.check_fwd(verbose=False)
            self.find_model()
            self.log(self.status_line()+op_str, stdout=True)
            in_level_step = int(self.in_level * 1.09050773)  # "0.75 dB" step
            self.in_level = min(self.in_level * self.out_level_goal * 0.85 / self.out_level, in_level_step)
            if self.in_level > self.real_clip:
                self.log("Max drive level exceeded in ramp_field(), aborting", stdout=True)
                self.shutdown_drive(terminate=True, code=2)

    ########
    def fine_adjust_poffset0(self):
        self.log("\nFine-adjust SEL phase offset", stdout=True)
        start_offset = self.reg_ph_offset
        in_vals = []
        self.out_levels = []
        phase_step = 18  # degrees
        num_steps = 3  # on each side of zero
        for ox in range(-num_steps, num_steps+1):
            # one degree is 728.2 counts
            phase_shift = ox*728*phase_step
            trial_offset = angle_shift(start_offset, phase_shift)
            self.log("Phase delta set %d degrees; Trying %d .." % (ox*phase_step, trial_offset))
            r2_deg = trial_offset*360/262144.0
            op_str = u"  from: ph_off rock %d/%d: %5.1f\u00B0" % (ox+num_steps+1, 2*num_steps+1, r2_deg)
            query = self.sel_set + [('ph_offset', trial_offset)]
            self.write_and_acquire(query)
            self.out_level = self.find_level(verbose=False)
            self.out_levels.append(self.out_level)
            in_vals.append(phase_shift)
            self.usual_find_slope(verbose=False)
            self.check_fwd(verbose=False)
            self.find_model()
            self.log(self.status_line()+op_str, stdout=True)
        trial_offset = self.analyze_poffset_scan(in_vals, self.out_levels, start_offset)
        if trial_offset is None:
            self.log("Fine phase offset scan analysis failed, aborting", stdout=True)
            self.shutdown_drive(terminate=True, code=2)
        query = self.sel_set + [('ph_offset', trial_offset)]
        r2_deg = trial_offset*360/262144.0
        op_str = u"  from: ph_off final: %5.1f\u00B0" % (r2_deg)
        self.write_and_acquire(query)
        self.out_level = self.find_level(verbose=False)
        self.usual_find_slope(verbose=False)
        self.check_fwd(verbose=False)
        self.find_model()
        self.log(self.status_line()+op_str, stdout=True)

    ########
    def fine_adjust_field0(self):
        self.log("\nFine-adjusting field", stdout=True)
        self.out_level = 0
        while abs(self.out_level/self.out_level_goal-1) > 0.01:
            out_mvpm = self.out_level * self.cav_scale
            self.log("Measured response %.1f (%.3f MV/m), will set new level %d" %
                     (self.out_level, out_mvpm, self.in_level))
            op_str = "  from: drive adjust, prev %5.1f%%" % (100*self.out_level/self.out_level_goal)
            query = self.sel_set + self.simple_pulse(self.in_level, 240)
            self.write_and_acquire(query)
            # [['amp_max',in_level]])
            self.out_level = self.find_level(verbose=False)
            self.usual_find_slope(verbose=False)
            self.check_fwd(verbose=False)
            self.find_model()
            self.log(self.status_line()+op_str, stdout=True)
            self.in_level = min(self.in_level * self.out_level_goal / self.out_level, self.in_level+1000)
            if self.in_level > self.real_clip:
                self.log("Max drive level exceeded in fine_adjust_field(), aborting", stdout=True)
                self.shutdown_drive(terminate=True, code=2)

    ########
    # XXX untested use of write_and_acquire
    # We used to use this on a software simulator;
    # actual RFS doesn't have direct control over piezo DC drive.
    def fine_tune_cavity(self):
        self.log("\nFine-tune cavity with SEL on", stdout=True)
        vstep = 100
        oldslope = 0
        slope, bw, max_amp = self.usual_find_slope()
        try_once = True
        while (abs(slope) > 4) or try_once:
            self.log("Piezo set %d" % self.piezo_dc, stdout=True)
            query = self.sel_set + [['piezo_dc', self.piezo_dc+65536 if self.piezo_dc < 0 else self.piezo_dc]]
            self.write_and_acquire(query)
            slope, bw, max_amp = self.find_slope(range(0, self.start), range(
                self.start-50, self.start-5), range(self.start, self.start+50))
            if oldslope*slope < 0:  # direction change
                vstep = vstep/4
            self.piezo_dc += vstep*(1 if slope > 0 else -1)
            oldslope = slope
            try_once = False

    ########
    def stretch_pulse0(self, llen_list=range(270, 690, 30)):
        # duration*8, duration=2000
        self.log("\nStretching pulse", stdout=True)
        # assume "start" still set from when pulse length was 240
        start_base = self.start - 240
        # last llen value is 660
        mr_set = []
        for ix, llen in enumerate(llen_list):  # range(270, 690, 30):
            self.log("Pulse length set %d * %d" % (llen, self.tick_multiply))
            op_str = "  from: pulse stretch %d/%d" % (ix+1, len(llen_list))
            self.start = start_base + llen
            query = self.simple_pulse(self.in_level, llen)
            # print(len_set)
            self.write_and_acquire(query)
            mr = self.find_model()
            mr_set += [mr]
            self.out_level = self.find_level(verbose=False)
            self.usual_find_slope(verbose=False)
            self.check_fwd(verbose=False)
            self.log(self.status_line()+op_str, stdout=True)
        sf = self.handle_mr_set(mr_set)
        if sf is not None:
            self.push_digaree_coeff(sf)

    ########
    def open_control0(self):
        self.log("\nOpening up control span", stdout=True)
        # given theta = 2*pi*7/33, z = exp(i*theta), and a = 1/16,
        # CORDIC gain = \prod_n \sqrt{1+4^{-n}} = 1.646760
        # lo = 74762/2^17 * CORDIC gain = 0.938439
        # fwashout gain = abs(1+a/(1-a-z)) = 1.031391
        # fdownconvert gain = lo * sin(theta) = 0.911986
        fdbk_scale = 0.774483  # fwashout gain * fdownconvert gain * CORDIC gain / 2
        set_R = int(self.out_level_goal * 2**17 * fdbk_scale)
        self.log("setmp_0 = %d" % set_R, stdout=True)
        len_set = [('setmp_0', set_R)] + self.triple_pulse(120, 540, 0, self.in_level, 0)
        self.write_and_acquire(len_set)

    ########
    def close_phase_loop0(self, set_P=2**17):
        self.log("\nClosing phase loop: clip %d" % self.imag_clip, stdout=True)
        len_set = [('setmp_1', set_P)] + \
            self.triple_pulse(120, 60, 480, self.in_level, self.imag_clip)
        self.write_and_acquire(len_set)

    ########
    def goto_cw0(self):
        self.log("\nGo To closed-loop CW", stdout=True)
        len_set = self.triple_pulse(120, 60, 480, self.in_level, self.imag_clip, cw=True)
        self.write_and_acquire(len_set)
        self.dsp_status_check_clipped(True)
        len_set = self.triple_pulse(0, 0, 0, self.in_level, self.imag_clip, cw=True)
        self.write_and_acquire(len_set)
        self.dsp_status_check_clipped(False)
        self.log("Idling tgen", stdout=True)
        len_set = self.rfs.tgen_reg_sequence([])
        self.write_and_acquire(len_set)
        self.dsp_status_check_clipped(False)
        self.log("Pushing static control values to chip (should be no-op)", stdout=True)
        len_set = self.static_box(self.in_level, self.imag_clip)
        self.write_and_acquire(len_set)
        self.dsp_status_check_clipped(False)
        self.check_fwd()
        cav = self.data_array[self.CAV] * 1j
        cav_mean = numpy.mean(cav)
        cav_norm = cav / cav_mean
        cav_std_r = numpy.std(cav_norm.real)
        cav_std_i = numpy.std(cav_norm.imag)
        cav_mean_mvpm = abs(cav_mean) * self.cav_scale
        cav_print = cav_mean.real, cav_mean.imag, cav_mean_mvpm, cav_std_r, cav_std_i
        self.log("cavity mean %.6f%+.6fj (%.3f MV/m)  std %.5f %.5fj" % cav_print, stdout=True)
        if cav_std_r < 0.0012 and cav_std_i < 0.0012:
            self.log("\nPASS", stdout=True)
        else:
            self.log("\nNoisy closed loop?", stdout=True)

    ########
    def scan_8pi_mode(self, f_list):  # 8p/9 mode, but no / in names

        self.log("\nScanning 8pi/9 mode", stdout=True)

        for f in f_list + [0]:
            # print("Setting offset frequency %.0f kHz" % f)
            # same frequency, two different representations
            n1 = 222425 + round(f*1e3*2**20*self.Tstep)
            n2 = n1*4096 + 868
            # print("n1, n2 = %d %d" % (n1, n2))
            exact = (n1 + 868.0/4092.0)*0.5**20/self.Tstep - 20e6
            self.log("Setting offset frequency %.0f kHz (%.1f Hz)" % (f, exact), stdout=True)
            self.rfs.set_channel_mask(0x3fc)
            # phase_step_h goes to PRC, not RFS!
            query = [('dsp_phase_step', n2)]
            prc_op = [['phase_step_h', n1]]
            if f == 0:
                query += [('dsp_ctlr_ph_reset', 1)]
                prc_op += [('prc_dds_ph_reset', 1)]
            if self.prc_target:
                prc_op = [{x[0]: x[1]} for x in prc_op]  # inconsistent API!
                self.prc_target.reg_write(prc_op)
                # Causality argument: this write happens first, then the write
                # to the RFS.  Waiting for RFS setup to propagate to the acquired
                # waveform also ensures the change has taken place in PRC.
            else:
                query += prc_op
            self.write_and_acquire(query)
            # analyze
            self.set_start()
            delta_f, bw, max_amp = self.usual_find_slope(verbose=False)
            delta_f = delta_f - exact
            self.log("Measured bandwidth %6.3f kHz, net detune %.3f kHz, max amp %.5f" %
                     (bw/1000, delta_f/1000, max_amp), stdout=True)
        # DDS phases have been reset by the "if f == 0:" stanza above
        # Depends on bitfile built by commit 9477c2d8 or later

    ########
    def shutdown_drive(self, terminate=False, code=0):
        self.log("\nShutting down drive", stdout=True)
        self.in_level = 0
        query = self.sel_set + self.simple_pulse(self.in_level, 240)
        self.write_and_acquire(query)
        self.log('Idling tgen and pushing static control values to chip')
        for kx in range(2):
            query = self.rfs.tgen_reg_sequence([])
            query += self.static_box(self.in_level, 0.0)
            self.write_and_acquire(query)
            self.out_level = self.find_level()
            self.check_fwd()
        if terminate:
            if code != 0:
                self.log("\nFAIL", stdout=True)
            print("\nLog file is available in " + master.data_dir + "/beg.log")
            exit(code)

    def log(self, line, stdout=False):
        if self.logfile:
            self.logfile.write(line+'\n')
        if stdout:
            print(line)
            sys.stdout.flush()


def get_conf(args, log_root='beg_'):
    """
    Generate configuration dictionary based on defaults and command-line arguments
    """
    # default values
    conf = dict(
        desc=None,
        rfs_addr='leep://192.168.165.44',
        port=50006,
        wsp=1,
        loopback=False,
        mode_center=None,
        zone=1,
        prc_ip=None,
        loop_is_cav=False,  # allows running on single RFS without fiber links
        dmax_magn=0.8,
        dmax_imag=0.8,
        fwd_fs=10.0,    # kW
        rev_fs=10.0,    # kW
        cav_fs=40.0,    # MV/m
        cav_goal=5.8,   # MV/m
        detune_fq=0.008,  # Hz frequency quantum, user choice
        lp_bw=150e3,    # Hz Low-pass filter bandwidth
        notch_f=None  # Hz notch filter frequency
    )

    # Remove defaulted arguments, defaults are set on conf dictionary
    args = {key: value for key, value in args.items() if value is not None}

    # Overwrite defaults with configuration in JSON file
    if 'json_file' in args:
        json_dict = json.load(open(args['json_file'], 'r'))
        conf.update((k, json_dict[k]) for k in conf.viewkeys() & json_dict.viewkeys())

    # Update configuration dictionary with values passed from the command line
    # Command-line arguments take priority over JSON file settings
    conf.update(args)

    data_dir = start_time.strftime(log_root + '%Y%m%d_%H%M%S')
    os.mkdir(data_dir)
    conf['data_dir'] = data_dir

    if conf['mode_center'] is None:
        conf['prc_ip'] = None

    return conf

def usage():
    print("Usage: python rf_controls.py -a $IP -d 255 -z 1")
    print("Better: python rf_controls.py -j foo.json")


if __name__ == '__main__':
    # maybe wrong in theory, but successfully lets this program
    # print its its fancy unicode symbols to pipes and redirects
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)

    save_all_buffers = False  # not appropriate for JSON file
    # process command line
    from argparse import ArgumentParser

    parser = ArgumentParser(description="RF Controls interface commands")

    parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG, default=logging.INFO, dest='debug')
    parser.add_argument('-q', '--quiet', action='store_const', const=logging.WARN, dest='debug')
    parser.add_argument('-a', '--address', dest="rfs_addr", default=None, help='RFS URL (leep://<IP> or ca://<PREFIX>)')
    parser.add_argument('-l', '--loopback', dest='loopback', action='store_true',
                        default=None, help='Enable loopback mode')
    parser.add_argument('-m', '--mode', dest='mode_center', type=int,
                        default=None, help='Center for 8pi/9 mode search (kHz)')
    parser.add_argument('-p', '--port', dest='port', default=None, type=int, help='UDP Port')
    parser.add_argument('-P', '--PRC', dest='prc_ip', default=None, help='PRC IP address (for 8pi/9 mode scan)')
    parser.add_argument('-d', '--decimate', dest='wsp', default=None, type=int, help='Samples per waveform buffer')
    parser.add_argument('-z', '--zone', dest='zone', default=None, type=int, help='RFS controller zone (0 or 1)')
    parser.add_argument('-j', '--json', dest='json_file', help='JSON configuration file')
    parser.add_argument('-s', '--save_all_buffers', action='store_true',
                        dest='save_all_buffers', default=False, help='Save all data buffers')

    # Get dictionary with command-line arguments
    args = vars(parser.parse_args())

    conf = get_conf(args)

    master = c_rf_controls(conf, save_all_buffers=conf['save_all_buffers'])
    master.configure_fpga()

    for key in ['zone', 'loopback', 'mode_center']:
        master.log("  Config %s %s" % (key, repr(conf[key])), stdout=True)

    # Loopback looks at results of first write_and_acquire() call from master.configure_fpga()
    lr = master.analyze_loopback(check=(conf['wsp'] == 255))  # no waiting
    if conf['loopback']:
        master.shutdown_drive(terminate=True, code=(0 if lr else 1))

    if conf['mode_center'] is not None:
        mc = conf['mode_center']
        # typical SRF use is mode_center 770 for 8pi/9, but we're flexible
        master.scan_8pi_mode(range(mc-70, mc+74, 4))
        master.shutdown_drive(terminate=True)
    try:
        master.conf_lp_notch()
        master.coarse_tune_cavity0()
        master.switch_to_sel0()
        master.ramp_field0()
        master.fine_adjust_poffset0()
        master.fine_adjust_field0()
        master.stretch_pulse0(llen_list=range(270, 690, 30))
        master.open_control0()
        master.close_phase_loop0(set_P=2**17)
        master.goto_cw0()
    except SystemExit:
        raise
    except:
        _log.exception('Something went wrong with one of the cavity bring-up functions')
        master.log("\nException!", stdout=True)
        master.shutdown_drive(terminate=True, code=1)
        raise
    print("\nLog file is available in " + master.data_dir + "/beg.log")
