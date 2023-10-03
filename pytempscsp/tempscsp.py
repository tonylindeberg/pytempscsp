"""Temporal Scale Space Toolbox for Python

For performing temporal smoothing with the time-causal limit kernel and
for computing discrete temporal derivative approximations by applying
temporal difference operators to the smoothed data.

This code is the result of porting a subset of the routines in the Matlab
package tempscsp to Python, however, with different interfaces for the functions.

References:

Lindeberg (2023) "A time-causal and time-recursive temporal scale-space representation
of temporal signals and past time", Biological Cybernetics 117(1-2): 21-59.

Lindeberg (2016) "Time-causal and time-recursive spatio-temporal receptive fields",
Journal of Mathematical Imaging and Vision 55(1): 50-88.

The time-causal limit kernel was first defined in Lindeberg (2016), however,
then also in combination with a spatial domain, and experimentally tested on
video data. The later overview paper (Lindeberg 2023) gives a dedicated treatment
for a a purely temporal domain, and also with relations to Koenderink's scale-time
kernels and the ex-Gaussian kernel.

Compared to the original Matlab code underlying the published experiments,
the following implementation is reduced in the following ways:
- there is no implementation of Lp-normalization (which for efficiency
  reasons should be done in combination with a disk cashing mechanism)
- there are no alternative non-causal temporal smoothing methods in this package
- this reimplementation has not yet been thoroughly tested

Concerning the default value of the parameter numlevels regarding the number of
discrete scale levels to approximate the time-causal limit kernel by a finite
number of first-order recursive filters coupled in cascade, this value may be
unnessarily large for c = 2, whereas a larger number of scale levels may be
needed for smaller values of c. If computational efficiency of the implementation
is important, such as in the combination with a spatial image domain for video
analysis, then it may be recommended to optimize the value of this parameter
by choosing a different trade-off between computational efficiency and accuracy
of the approximation.

Note: This code is for offline filtering, for experimentation purposes
to work out properties of algorithms building on the time-causal limit kernel.
For real-time filtering, do instead use the explicit recursive formulation
in Equation (56) in (Lindeberg 2016). Such an implementation will also
be more memory-efficient for processing e.g. spatio-temporal or spectro-temporal data.
"""
from math import sqrt, log, ceil
from scipy.signal import lfilter, sosfilt
import numpy as np


def mufromstddev(
        stddev: float,
        c: float = 2.0,
        numlevels: int = 8
) -> (np.ndarray, np.ndarray):
    """Determines the time constants mu for a set of recursive filters coupled
    in cascade that approximate the time-causal limit kernel at temporal
    scale stddev in units of the standard deviation of the time-causal
    limit kernel (not in terms of the variance tau, as used in the scientific papers)
    """
    # Determine the tau value at each level in the temporal scale hierarchy
    # (according to Equation (18) in (Lindeberg 2016))
    tau = np.zeros(numlevels)
    for i in range(numlevels):
        tau[i] = (stddev**2) / (c**(2*(numlevels-i-1)))

    # Determine the mu values between each pair of levels
    mu = np.zeros(numlevels)
    for i in range(1, numlevels):
        deltatau = tau[i] - tau[i-1]
        # Compute mu value from tau difference
        # According to Equation (58) in (Lindeberg 2016)
        mu[i] = (-1 + sqrt(1 + 4 * deltatau)) / 2

    # Special handling for the first layer of the recursive filters
    # (assuming that the input signal is acquired without temporal smoothing)
    deltatau = tau[0]
    mu[0] = (-1 + sqrt(1 + 4*deltatau)) / 2

    # Use sigma instead of tau in all interfaces to the functions
    sigma = np.sqrt(tau)

    return mu, sigma


def mufromstddevs(
        stddevmin: float,
        stddevmax: float,
        c: float = 2.0,
        numlevels : int = 8
) -> (np.ndarray, np.ndarray):
    """Determine the time constants mu needed to compute a set of temporal scale-space
    representations over the scale range [stddevmin, stddevmax] in units of the
    standard deviation of the time-causal limit kernel
    """
    # Determine how many extra levels are needed above stddevmin, assuming
    # that this level is to be preserved and that stddevmax may need to be
    # increased to guarantee a ratio between temporal scale levels equal to c
    numextralevels = ceil(log(stddevmax / stddevmin) / log(c))

    # Use the functionality for the function based on a single scale output
    newstddevmax = stddevmin * c ** numextralevels
    mu, sigma = mufromstddev(newstddevmax, c, numextralevels + numlevels)

    return mu, sigma


def limitkern_sospars_2layers(
        mu1 : float,
        mu2 : float
) -> np.ndarray:
    """Computes the sos parameters for two first-order recursive filters in cascade.

    The resulting parameters [b0, b1, b2, a0, a1, a2] represent the composition of 
    two generating functions of the form
    H1(z) * H2(z) = 1/(1 - mu1*(z-1)) * 1/(1 - mu2*(z-1))
                   = (b0 + b1*z + b2*z^2)/(a0 + a1*z + a2*z^2)
    based on Equation (57) in (Lindeberg 2016)
    """
    pars = np.array([
        1.0, 0.0, 0.0,
        1 + mu1 + mu2 + mu1 * mu2, -mu1 - mu2 - 2 * mu1 * mu2, mu1 * mu2,
    ])
    return pars / pars[3]


def limitkern_sospars_1layer(mu : float) -> np.ndarray:
    """Returns the sos parameters for a single first-order recursive filter

    The resulting parameters [b0, b1, b2, a0, a1, a2] represent a single generating 
    function of the form
    H(z) = 1/(1 - mu*(z-1))
         = (b0 + b1*z + b2*z^2)/(a0 + a1*z + a2*z^2)
    according to Equation (57) in (Lindeberg 2016)
    """
    pars = [1.0, 0, 0, 1 + mu, -mu, 0]
    return np.array(pars)


def limitkern_composedsospars_alllayers_list(muvec: np.array) -> np.ndarray:
    """Returns the composed sos parameters for multiple first-order recursive filters 
    coupled in cascade.

    This is done by recursive list concatenation (according to the documentation
    of another sos filtering routine, however, not the same as then later being
    used in the code below)
    """
    if not isinstance(muvec, np.ndarray):
        raise TypeError("muvec should be an array!")

    if len(muvec) > 2:
        # Pick out the first two elements from the list. Then, apply
        # the same function recursively to the rest of the list
        mu1, mu2 = muvec[:2]
        return np.concatenate([
                limitkern_sospars_2layers(mu1, mu2),
                limitkern_composedsospars_alllayers_list(muvec[2:])
        ])
    if len(muvec) == 2:
        mu1, mu2 = muvec[:2]
        return limitkern_sospars_2layers(mu1, mu2)
    if len(muvec) == 1:
        mu1 = muvec[0]
        return limitkern_sospars_1layer(mu1)

    raise ValueError("This case should not occur!")


def limitkern_composedsospars_alllayers(muvec: np.ndarray) -> np.ndarray:
    """Computes the sos parameters for computing the convolution with a discrete
    approximation of the time-causal limit kernel, as defined from the time constants
    in the mu vector muvec.
    """
    listformat = limitkern_composedsospars_alllayers_list(muvec)
    numlayers = int(round(len(listformat) / 6))

    # Reformat the previously generated array into a matrix of the desired format
    # for the routine sosfilt used in the code below.
    return np.reshape(listformat, [numlayers, 6])


def limitkernfilt(
        signal,
        stddev : float,
        c: float = 2.0,
        numlevels: int = 8,
        method: str = 'explicitcascade',
        axis: int = -1
) -> np.ndarray:
    """Performs temporal filtering with a discrete approximation of the time-causal
    limit kernel based on numlevels recursive filters coupled in cascade.

    The scale parameter stddev is expressed in units of the standard deviation 
    of the temporal scale-space kernel, corresponding the square root of
    the parameter tau in the scientific papers, which is expressed in units
    of the variance of the temporal scale-space kernel.

    The distribution parameter c should be strictly > 1, where a larger value
    leads to a more sparse sampling of the temporal scale levels implying
    less computational work, and also shorter temporal delays, whereas a
    smaller value of c leads to a denser sampling of the temporal scale
    levels at the costs of more computational work and longer temporal delays.
    """
    muvec, _ = mufromstddev(stddev, c, numlevels)

    if method == 'explicitcascade':
        for mu in muvec:
            # Set up the parameters for the recursive filter, defined according to
            # Equation (57) in (Lindeberg 2016)
            a = [1 + mu, -mu]
            b = [1.0]
            signal = lfilter(b, a, signal, axis)
        return signal

    if method == 'sosfilt':
        # The following implementation is based on incomplete information
        # regarding the sos parameter protocol, and is therefore, although
        # experimentally tested by reverse engineering, not guaranteed to work.
        # This method could, however, have the potential of running faster
        # and being more memory efficient on larger datasets, if the sosfilt
        # routine is written in such a way that it does not explicitly store
        # the intermediate results.
        # Use at your own risk, and make sure that you double-check the
        # functionality with respect to the default mode 'explicitcascade'
        sospars = limitkern_composedsospars_alllayers(muvec)
        return sosfilt(sospars, signal, axis)

    raise ValueError(f'Unknown filtering method {method}')


def limitkernfilt_mult(
        signal,
        stddevmin: float,
        stddevmax: float,
        c: float = 2.0,
        numlevels: int = 8,
        axis: int = -1
) -> (np.ndarray, np.ndarray):
    """Computes a set of temporal scale-space representations for discrete
    approximations of the time-causal limit kernel within the scale range spanned
    by the standard deviations stddevmin and stddevmax (possibly extended because
    of the ratio between successive scale levels as given by the scale ratio c),
    and using numlevels recursive filters coupled in cascade for the first 
    temporal scale level.

    The scale parameter stddev is expressed in units of the standard deviation 
    of the temporal scale-space kernel, corresponding the square root of
    the parameter tau in the scientific papers, which is expressed in units
    of the variance of the temporal scale-space kernel.

    The distribution parameter c should be strictly > 1, where a larger value
    leads to a more sparse sampling of the temporal scale levels implying
    less computational work, and also shorter temporal delays, whereas a
    smaller value of c leads to a denser sampling of the temporal scale
    levels at the costs of more computational work and longer temporal delays.
    """
    muvec, sigmavec = mufromstddevs(stddevmin, stddevmax, c, numlevels)

    # Perform the smoothing until the first level that is to be preserved
    for i in range(numlevels):
        # Set up parameter for recursive filter
        mu = muvec[i]
        a = [1 + mu, -mu]
        b = [1.0]
        signal = lfilter(b, a, signal, axis)

    numoutputs = len(sigmavec) - numlevels + 1
    signal_mult = np.zeros([numoutputs, signal.shape[0]])

    signal_mult[0] = signal
    for counter in range(1, numoutputs):
        mu = muvec[numlevels + counter - 1]
        a = [1 + mu, -mu]
        b = [1.0]
        signal = lfilter(b, a, signal, axis)
        signal_mult[counter] = signal

    sigma_out = sigmavec[slice(numlevels - 1, numlevels - 1 + numoutputs)]

    return signal_mult, sigma_out


def tempdelayfrommu(muvec: np.ndarray) -> np.ndarray:
    """Returns the temporal delay in each layer for a set of first-order recursive
    filters coupled in cascade

    The temporal delay for a single recursive filter is mu, whereas the temporal
    delay for a set of recursive filters in cascade is the sum of the mu-values
    (see the text preceding Equation (58) in (Lindeberg 2016))
    """
    return np.cumsum(muvec)


def tempdelayfromstddev(
        stddev: float,
        c: float = 2.0,
        numlevels: int = 8
) -> float:
    """Returns the temporal delay associated with a discrete approximation of the
    time-causal limit kernel performed by the function limitkernfilt()
    """
    muvec, _ = mufromstddev(stddev, c, numlevels)
    delay = tempdelayfrommu(muvec)

    return delay[-1]


def tempdelaysfromstddevs(
        stddevmin: float,
        stddevmax: float,
        c: float = 2.0,
        numlevels: int = 8
) -> np.ndarray:
    """Returns the temporal delays associated with a set of discrete approximations of
    the time-causal limit kernel performed by the function limitkernfilt_mult()
    """
    muvec, sigmavec = mufromstddevs(stddevmin, stddevmax, c, numlevels)
    delay = tempdelayfrommu(muvec)
    numoutputs = len(sigmavec) - numlevels + 1

    return delay[slice(numlevels - 1, numlevels - 1 + numoutputs)]


def normderfactor(
        stddev: float,
        order: int,
        normdermethod: str = 'variance',
        gamma: float = 1.0
) -> float:
    """Normalization factor for scale-normalized derivatives.
    """
    if normdermethod == "nonormalization":
        return 1.0

    if normdermethod == 'variance':
        # Scale normalization according to Equation (74) in (Lindeberg 2016)
        return stddev**(gamma * order)

    raise ValueError(
        f"Unknown temporal derivative normalization method {normdermethod}"
    )


def tempder(
        insignal,
        derivative: str,
        stddev: float,
        normdermethod: str = 'variance',
        gamma: float = 1.0,
        axis: int = -1
) -> np.ndarray:
    """Computes scale-normalized temporal derivatives from an already computed temporal
    scale-space representation.
    """
    # a and b are the parameters for the linear filtering operations
    a = 1

    if derivative == 'Lt':
        b = [1, -1]
        return (
                normderfactor(stddev, 1, normdermethod, gamma) *
                lfilter(b, a, insignal, axis)
        )

    if derivative == 'Ltt':
        b = [1, -2, 1]
        return (
                normderfactor(stddev, 2, normdermethod, gamma) *
                lfilter(b, a, insignal, axis)
        )

    raise ValueError(f'Unknown temporal derivative method {type}')


def quasiquad(
        insignal,
        stddev: float,
        normdermethod: str = 'variance',
        Gamma: float = 0.0,
        C: float = 1 / sqrt(2),
        axis: int = -1
) -> np.ndarray:
    """Computes the temporal quasi quadrature measure defined in Equation (55) in
    Lindeberg (2018) "Dense scale selection over spatial, temporal and
    spatio-temporal domains", SIAM Journal on Imaging Sciences, 11(1): 407–441.

    Note, however, that the default value of C has been determined for
    the non-causal Gaussian scale space, whereas the application of the
    quasi quadrature measure in combination with the time-causal scale space
    obtained by convolution with the time-causal limit kernel may call
    for a better determination of the relative weighting factor C and then also as
    function of the distribution parameter c of the time-causal limit kernel.
    """
    # The scale normalization by dividing by stddev^(2*Gamma) is transfered
    # to other gamma parameters for the 1:st- and 2:nd-order derivatives
    Lt = tempder(insignal, 'Lt', stddev, normdermethod, 1 - Gamma / 2, axis)
    Ltt = tempder(insignal, 'Ltt', stddev, normdermethod, 1 - Gamma / 4, axis)

    return Lt * Lt + C * Ltt * Ltt


def deltafcn1D(length: int) -> np.ndarray:
    """Discrete delta function in 1-D.
    """
    signal = np.zeros(length)
    signal[0] = 1.0

    return signal


def mean1D(signal) -> float:
    """Computes the temporal mean of a non-negative temporal signal.
    """
    if signal.ndim != 1:
        raise ValueError('Only implemented for 1-D signals')

    size = signal.shape[0]
    t = np.linspace(0, size-1, size)

    return np.sum(np.sum(t * signal)) / np.sum(np.sum(signal))


def variance1D(signal) -> float:
    """Computes the temporal variance of a non-negative temporal signal.
    """
    if signal.ndim != 1:
        raise ValueError('Only implemented for 1-D signals')

    size = signal.shape[0]
    t = np.linspace(0, size-1, size)

    t2mom = np.sum(np.sum(t * t * signal)) / np.sum(np.sum(signal))
    tmean = mean1D(signal)

    return t2mom - tmean * tmean


def whitenoisesignal1D(length: int) -> np.ndarray:
    """Generates a white noise signal of a given length.
    """
    signal = np.random.normal(0.0, 1.0, length)

    return signal


def browniannoisesignal1D(length: int) -> np.ndarray:
    """Generates a brownian noise signal of a given length.
    """
    whitenoise = whitenoisesignal1D(length)

    return np.cumsum(whitenoise)
