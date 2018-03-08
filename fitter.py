import numpy
from numpy import diff, pi
from scipy import signal
try:
    from matplotlib import pyplot
except:
    pass


# Cardinal B-spline  https://en.wikipedia.org/wiki/B-spline
# Same as the output of a second-order CIC interpolator
def create_basist(block=15, n=10):
    npt = (n-1)*block+1
    basist = numpy.zeros([npt, n])
    x = numpy.arange(npt)/float(block)
    for jx in range(n):
        basist[:, jx] = signal.bspline(x-jx, 2)
    # end-effects, it's important that the sum is flat
    basist[:, 0] += signal.bspline(x+1, 2)
    basist[:, n-1] += signal.bspline(x-n, 2)
    if False:
        pyplot.plot(basist)
        pyplot.plot(numpy.sum(basist, axis=1))
        pyplot.show()
    return basist


# basist is n*m, where n=len(cav)-1, m is number of time-dependent bases
def fitter(cav, fwd, basist, dt=1, plotme=False, axes=None, dest={}):
    acav = 0.5 * (cav[1:] + cav[:-1])
    afwd = 0.5 * (fwd[1:] + fwd[:-1])
    dcav = diff(cav)/dt
    cave = acav * basist.T
    # Fit dcav to a*acav + b*afwd
    # Treat real and imaginary parts separately, since we want the imaginary
    # part of b to be sum_i g_i*f_i(t) while the real part is constant.
    goal = numpy.hstack([dcav.real, dcav.imag])
    basis1 = numpy.hstack([afwd.real, afwd.imag])   # b.real
    basis2 = numpy.hstack([-afwd.imag, afwd.real])  # b.imag
    basis3 = numpy.hstack([acav.real, acav.imag])   # a.real
    basisn = numpy.hstack([-cave.imag, cave.real])  # a.imag repeated
    basis = numpy.vstack([basis1, basis2, basis3, basisn])
    # print goal.shape, basis.shape
    (fitc, resid, rank, sing) = numpy.linalg.lstsq(basis.T, goal, rcond=-1)
    mr = fitc[0]+1j*fitc[1]
    # print rank, fitc[:3],
    if axes is not None or "fitr" in dest:
        n2 = len(goal)
        t = numpy.arange(n2)
        if False:
            fit = basis.T.dot(fitc)
            pyplot.plot(t, fit)
            pyplot.plot(t, goal)
            pyplot.show()
        if True:
            n1 = basist.shape[0]
            tt = (numpy.arange(n1)+0.5)*dt
            detune = basist.dot(fitc[3:])
            bandwidth = fitc[2]/2/pi  # Hz
            bandlabel = "Fit %.2f Hz" % -bandwidth
            if "fitr" in dest:
                dest["fitr"].set_label(bandlabel)
                dest["fitr"].set_data(tt, tt*0+bandwidth)
            if "fiti" in dest:
                dest["fiti"].set_data(tt, detune/2/pi)
            if axes is not None:
                axes[0].plot(tt, tt*0+bandwidth, label=bandlabel)
                axes[1].plot(tt, detune/2/pi, label='Fit')
                # pyplot.show()
    return mr, rank, fitc[:3]


# Find Lorentz coefficient by curve-fitting
#   Works because there's a nice broad-band term,
#   and the excitation term is relatively narrow-band
# g: cavity field in MV/m
# ssmi: state-space model imaginary part in Hz
def fit_lorentz(g, ssmi, plot=False):
    fit_x = g[2:-1]**2
    fit_y = ssmi[2:]
    pp = numpy.polyfit(fit_x, fit_y, 1)
    g_squared = numpy.polyval(pp, g**2)
    lor_coeff = -pp[0]  # Hz/(MV/m)^2
    lor_label = "%.0f - %.2f*Gradient^2" % (pp[1], lor_coeff)
    if plot:  # messes up later plots when run from fitter_test.py
        pyplot.figure(2)
        pyplot.plot(fit_x, fit_y)
        pyplot.plot(fit_x, numpy.polyval(pp, fit_x))
        pyplot.title(lor_label)
        pyplot.show()
    return g_squared, lor_label, lor_coeff


# see data-20170518/README and digaree/initgen_srf.py
# Identical (?) to sf_consts() in detune_coeff.py
def fpga_regs(mr, fq, dt):
    # sf2 = 1.0/(2*pi*dt*80*fq*0.25)
    fir_gain = 80
    fs = 0.25
    ffs = fq*fs*2*pi  # s^{-1} full-scale
    invT = 1.0/(dt*fir_gain)  # units are /s
    sf = [
        mr.real / ffs,
        mr.imag / ffs,
        invT / ffs,
        32768]
    return [int(round(sf[kx])) for kx in range(4)]
