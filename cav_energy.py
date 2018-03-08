import numpy
from numpy import log, exp, sqrt, pi, log10
from matplotlib import pyplot
from scipy.optimize import brentq


# pf: forward power array, Watts
# pr: reverse power array, Watts
# ex: index of nominal end-of-pulse
def rev_area(pf, pr, ex, plot=False, verbose=False):
    ix = numpy.arange(ex+4, ex+64)
    pp = numpy.polyfit(ix, log(pr[ix]), 1)
    jx = numpy.arange(ex-20, ex+74)
    kx = numpy.arange(ex-20, ex-3)
    pk = numpy.polyfit(kx, pr[kx], 1)
    if plot:
        pyplot.plot(ix, exp(numpy.polyval(pp, ix)))
        pyplot.plot(kx, numpy.polyval(pk, kx), '-')
        pyplot.plot(jx, pr[jx], 'x')
        pyplot.ylim(0, max(pr[jx]))
        pyplot.xlabel('Time step')
        pyplot.ylabel('Reverse Power (W)')
        pyplot.show()
    #
    ox = numpy.arange(ex-3, ex+4)  # overlap
    os = sum(pr[ox])
    oyl = numpy.polyval(pk, ox)
    oyr = exp(numpy.polyval(pp, ox))
    tl = ex-3.5
    tr = ex+3.5

    def area_match(t):
        int_l = (t-tl)*pk[1] + 0.5*(t**2-tl**2)*pk[0]
        int_r = exp(pp[1]) * (exp(pp[0]*tr) - exp(pp[0]*t)) / pp[0]
        return int_l+int_r-os
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html
    r = brentq(area_match, tl, tr)
    area = - exp(pp[1]+pp[0]*r) / pp[0]
    p1 = numpy.polyval(pk, r)
    p2 = exp(numpy.polyval(pp, r))
    if verbose:
        print('Power jumps factor of %.3f from %.1f to %.1f at edge' % (p2/p1, p1, p2))
    # Wish to reconstruct the measured waveform,
    # based on FIR filter coefficients in half_filt.v
    # [ 2 0 -9 0 39 64 39 0 -9 0 2 ] / 128
    if plot:
        pyplot.plot(ox, oyl, label='Left')
        pyplot.plot(ox, oyr, label='Right')
        pyplot.plot(ox, pr[ox], label='Data')
        pyplot.plot([r, r], [p1, p2])
        pyplot.legend()
        pyplot.xlabel('Time step')
        pyplot.ylabel('Reverse Power (W)')
        pyplot.show()
        pyplot.plot(pr[ix], pf[ix]/pr[ix])
        pyplot.xlabel('Time step')
        pyplot.ylabel('Forward/Reverse power ratio')
        pyplot.show()
    isol = pf[ix]/pr[ix]
    isol_min, isol_max = min(isol), max(isol)
    if verbose:
        print('Circulator S_22 range %.1f to %.1f dB' % (10*log10(isol_min), 10*log10(isol_max)))
    raw_bw = -0.5*pp[0]  # 1/tau, in time-step units
    return r, area, raw_bw
    # r: refined end-of-pulse time in ticks
    # area: in units of Watt-ticks


def cav_eval(y, r):
    ex = int(numpy.floor(r))
    kx = numpy.arange(ex-20, ex-3)
    pk = numpy.polyfit(kx, y[kx], 1)
    return numpy.polyval(pk, r)


def cav_energy(data, plot=False):
    FWD = 1
    REV = 2
    CAV = 3
    fwdm = abs(data[FWD])  # forward wave magnitude
    revm = abs(data[REV])
    cavm = abs(data[CAV])
    ex = max(numpy.nonzero(fwdm > 0.5*max(fwdm))[0])
    if ex > 1024-74:
        print('cav_energy: insufficient decay data, ex=%d' % ex)
        return
    # Static calibration for testing only, but at least it's consistent with
    # 9d226fdbdb0824e330127692af59f1cb57f9eeb9475831054ad00b6c65911c92  fitter_dat/auto_001.dat
    adc_fs = 519636.5
    fwd_fs = 11.97  # kW
    rev_fs = 10.48  # kW
    pf = (fwdm/adc_fs)**2 * fwd_fs * 1000
    pr = (revm/adc_fs)**2 * rev_fs * 1000
    # Analysis takes place in units of Watts and unitless time step
    r, raw_area, raw_bw = rev_area(pf, pr, ex, plot=plot, verbose=True)
    wsp = 255  # more static cal
    dt = 2*wsp*33*14/1320e6
    bw = raw_bw/dt/(2*pi)
    print('Edge detected at %d, refined to %.2f, bandwidth %.2f Hz' % (ex, r, bw))
    energy = raw_area * dt  # Joules
    gconvert = 2.875  # MV/sqrt(J),  sqrt(RoverQ*(f0*2*pi))
    book_l = 1.038  # m, cavity length, arguably fiction
    gradient = sqrt(energy) * gconvert / book_l
    print('Exponential area %.2f Joules, %.2f MV/m' % (energy, gradient))
    cav0 = cav_eval(abs(cavm)/adc_fs, r)
    cav_scale = gradient/cav0
    print('Cavity at that point measured %.4f, scaling %.2f MV/m' % (cav0, cav_scale))


def usage():
    print("python cav_energy -d fitter_dat/auto_001.dat -")

# Test with:
#    python cav_energy.py -d fitter_dat/auto_001.dat
# or
#    python cav_energy.py -d fitter_dat/auto_001.dat -p
if __name__ == '__main__':
    import getopt
    import sys
    dfile = None
    plot = False
    opts, args = getopt.getopt(sys.argv[1:], 'hj:d:p', ['help', 'json=', 'data=', 'plot'])
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            exit(2)
        if opt in ('-j', '--json'):
            pass  # wish we got rev_fs, fwd_fs, wsp from here
            # maybe also gconvert and book_l?
        if opt in ('-d', '--data'):
            dfile = arg
        if opt in ('-p', '--plot'):
            plot = True
    if dfile:
        foo = numpy.loadtxt(dfile).T
        foo = [foo[ix]+1j*foo[ix+1] for ix in range(0, 8, 2)]
        cav_energy(foo, plot=plot)
