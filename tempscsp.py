# Temporal Scale Space Toolbox for Python
#
# For performing temporal smoothing with the time-causal limit kernel and
# for computing discrete temporal derivative approximations by applying
# temporal difference operators to the smoothed data.
#
# This code is the result of porting a subset of the routines in the Matlab
# package tempscsp to Python, however, with different interfaces for the functions.
#
# References:
#
# Lindeberg (2023) "A time-causal and time-recursive temporal scale-space representation
# of temporal signals and past time", Biological Cybernetics 117(1-2): 21-59.
#
# Lindeberg (2016) "Time-causal and time-recursive spatio-temporal receptive fields",
# Journal of Mathematical Imaging and Vision 55(1): 50-88.
#
# The time-causal limit kernel was first defined in Lindeberg (2016), however,
# then also in combination with a spatial domain, and experimentally tested on
# video data. The later overview paper (Lindeberg 2023) gives a dedicated treatment
# for a a purely temporal domain, and also with relations to Koenderink's scale-time
# kernels and the ex-Gaussian kernel.
#
# Compared to the original Matlab code underlying the published experiments,
# the following implementation is reduced in the following ways:
# - there is no implementation of Lp-normalization (which for effiency
#   reasons should be done in combination with a disk cashing mechanism)
# - there are no alternative non-causal temporal smoothing methods in this package
# - this reimplementation has not yet been thoroughly tested
#
# Concerning the default value of the parameter numlevels regarding the number of
# discrete scale levels to approximate the time-causal limit kernel by a finite
# number of first-order recursive filters coupled in cascade, this value may be
# unnessarily large for c = 2, whereas a larger number of scale levels may be
# needed for smaller values of c. If computational efficiency of the implementation
# is important, such as in the combination with a spatial image domain for video
# analysis, then it may be recommended to optimize the value of this parameter
# by choosing a different trade-off between computational effiency and accuracy
# of the approximation.
#
# Note: This code is for offline filtering, for experimentation purposes
# to work out properties of algorithms building on the time-causal limit kernel.
# For real-time filtering, do instead use the explicit recursive formulation
# in Equation (56) in (Lindeberg 2016). Such an implementation will also
# be more memory-efficient for processing e.g. spatio-temporal or spectro-temporal data.


from math import sqrt
from math import log
from math import ceil

from scipy.signal import lfilter
from scipy.signal import sosfilt

import numpy as np


def mufromstddev(stddev, c=2, numlevels=8):
    # Determine the time constants mu for a set of recursive filters coupled
    # in cascade that approximate the time-causal limit kernel at temporal
    # scale stddev in units of the standard deviation of the time-causal
    # limit kernel (not in tau, as used in the scientific papers)
    
    # Determine the tau value at each level in the temporal scale hierarchy
    # (according to Equation (18) in (Lindeberg 2016))
    tau = [None] * (numlevels)
    for i in range(0, numlevels):
        tau[i] = (stddev**2) / (c**(2*(numlevels-i-1)))
    
    # Determine the mu values between each pair of levels
    mu = [None] * (numlevels)
    for i in range(1, numlevels):
        deltatau = tau[i] - tau[i-1]
        # Compute mu value from tau difference according to Equation (58) in (Lindeberg 2016)
        mu[i] = (-1 + sqrt(1 + 4*deltatau))/2
        
    # Special handling for the first layer of the recursive filters
    # (assuming that the input signal is acquired without temporal smoothing)
    deltatau = tau[0]
    mu[0] = (-1 + sqrt(1 + 4*deltatau))/2

    # Use sigma instead of tau in all interfaces to the functions
    sigma = np.sqrt(np.array(tau)).tolist()
        
    return mu, sigma


def mufromstddevs(stddevmin, stddevmax, c=2, numlevels=8):
    # Determine the time constants needed to compute a set of temporal scale-space
    # representations over the scale range [stddevmin, stddevmax] in units of the
    # standard deviation of the time-causal limit kernel
    
    # Determine how many extra levels are needed above stddevmin, assuming
    # that this level is to be preserved and that stddevmax may need to be
    # increased to guarantee a ratio between temporal scale levels equal to c
    numextralevels = ceil(log(stddevmax/stddevmin)/log(c))

    # Use the functionality for the function based on a single scale output
    newstddevmax = stddevmin * c**(numextralevels)
    mu, sigma = mufromstddev(newstddevmax, c, numextralevels+numlevels)
    
    return mu, sigma


def limitkern_sospars_2layers(mu1, mu2):
    # Returns the sos parameters for two recursive filters in cascade
    # The following is the composition of two generating functions of the form
    # H1(z) * H2(z) = 1/(1 - mu1*(z-1)) * 1/(1 - mu2*(z-1))
    #               = (b0 + b1*z + b2*z^2)/(a0 + a1*z + a2*z^2)
    # based on Equation (57) in (Lindeberg 2016)
    a = [1 + mu1 + mu2 + mu1*mu2, -mu1 - mu2 - 2*mu1*mu2, mu1*mu2]
    b = [1.0, 0, 0]

    # The sos parameters should be a concatenated list [b0, b1, b2, a0, a1, a2]
    pars = b + a
    
    # The sosfilt routine requires the third element to be equal to one
    return (np.array(pars)/pars[3]).tolist()


def limitkern_sospars_1layer(mu):
    # Returns the sos parameters for a single recursive filter
    # The following is a single generating function of the form
    # H(z) = 1/(1 - mu*(z-1)) 
    #      = (b0 + b1*z + b2*z^2)/(a0 + a1*z + a2*z^2)
    # according to Equation (57) in (Lindeberg 2016)
    a = [1 + mu, -mu, 0]
    b = [1.0, 0, 0]

    # The sos parameters should be a concatenated list [b0, b1, b2, a0, a1, a2]
    pars = b + a

    # The sosfilt routine requires the third element to be equal to one
    return (np.array(pars)/pars[3]).tolist()


def limitkern_composedsospars_alllayers_list(muvec):
    # Returns the composed sos parameters for multiple recursive filters in cascade
    # This is done by recursive list concatenation (according to the documentation
    # of another sos filtering routine, however, not the same as then later being
    # used in the code below)
    if (len(muvec) > 2):
        # Pick out the first two elements from the list. Then, apply
        # the same function recursively to the rest of the list
        mu1, mu2 = muvec[:2]
        return limitkern_sospars_2layers(mu1, mu2) +\
               limitkern_composedsospars_alllayers_list(muvec[2:])
    elif (len(muvec) == 2):
        mu1, mu2 = muvec[:2]
        return limitkern_sospars_2layers(mu1, mu2)
    elif (len(muvec) == 1):
        mu1 = muvec[0]
        return limitkern_sospars_1layer(mu1)
    else:
        raise ValueError("This case should not occur")


def limitkern_composedsospars_alllayers(muvec):
    listformat = limitkern_composedsospars_alllayers_list(muvec)
    numlayers = int(round(len(listformat)/6))

    # Reformat the previously generated list into a matrix of the desired format
    # for the routine sosfilt used in the code below
    return np.reshape(listformat, [numlayers, 6])

    
def limitkernfilt(signal, stddev, c=2, numlevels=8, method='explicitcascade', axis=-1):
    # Performs temporal filtering with a discrete approximation of the time-causal
    # limit kernel based on numlevels recursive filters coupled in cascade
    
    muvec, sigmavec = mufromstddev(stddev, c, numlevels)

    if (method == 'explicitcascade'):
        for mu in muvec:
            # Set up the parameters for the recursive filter, defined according to
            # Equation (57) in (Lindeberg 2016)
            a = [1 + mu, -mu]
            b = [1.0]
            signal = lfilter(b, a, signal, axis)

    elif (method == 'sosfilt'):
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
        signal = sosfilt(sospars, signal, axis)

    else:
        raise ValueError('Unknown filtering method %s', method)

    return signal

    
def limitkernfilt_mult(signal, stddevmin, stddevmax, c=2, numlevels=8, axis=-1):
    # Computes a set of temporal scale-space representations for discrete approximations
    # of the time-causal limit kernel within scale range spanned by the standard
    # deviations stddevmin and stddevmax (possibly extended because of the ratio
    # between successive scale levels as given by the scale ratio c), and using
    # numlevels recursive filters for the first temporal scale level.
 
    muvec, sigmavec = mufromstddevs(stddevmin, stddevmax, c, numlevels)

    # Perform the smoothing until the first level that is to be preserved
    for i in range(0, numlevels):
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

    sigma_out = sigmavec[slice(numlevels-1,numlevels-1+numoutputs)]

    return signal_mult, sigma_out


def tempdelayfrommu(muvec):
    # Returns the temporal delay in each layer for a set of first-order recursive
    # filters coupled in cascade

    # The temporal delay for a single recursive filter is mu, whereas the temporal
    # delay for a set of recursive filters in cascade is the sum of the mu-values
    # (see the text preceding Equation (58) in (Lindeberg 2016))
    layerdelay = np.array(muvec)
    accdelay = np.cumsum(layerdelay)

    return accdelay.tolist()


def tempdelayfromstddev(stddev, c=2, numlevels=8):
    # Returns the temporal delay associated with a discrete approximation of the
    # time-causal limit kernel performed by the function limitkernfilt()
    
    muvec, sigmavec = mufromstddev(stddev, c, numlevels)
    delay = tempdelayfrommu(muvec)

    return delay[-1]


def tempdelaysfromstddevs(stddevmin, stddevmax, c=2, numlevels=8):
    # Returns the temporal delays associated with a set of discrete approximations of
    # the time-causal limit kernel performed by the function limitkernfilt_mult()

    muvec, sigmavec = mufromstddevs(stddevmin, stddevmax, c, numlevels)
    delay = tempdelayfrommu(muvec)
    numoutputs = len(sigmavec) - numlevels + 1
    
    return delay[slice(numlevels-1,numlevels-1+numoutputs)]

       
def normderfactor(stddev, order, normdermethod='variance', gamma=1.0):
    # Normalization factor for scale-normalized derivatives

    if (normdermethod == 'none'):
        normfactor = 1.0
    elif (normdermethod == 'variance'):
        # Scale normalization according to Equation (74) in (Lindeberg 2016)
        normfactor = stddev**(gamma*order)
    else:
        raise ValueError('Unknown temporal derivative normalization method %s', normdermethod)

    return normfactor


def tempder(insignal, type, stddev, normdermethod='variance', gamma=1.0, axis=-1):
    # Computes scale-normalized temporal derivatives from a
    # temporal scale-space representation
    
    # a and b are the parameters for the linear filtering operations
    a = 1

    if (type == 'Lt'):
        b = [1, -1]
        outsignal = normderfactor(stddev, 1, normdermethod, gamma) * \
          lfilter(b, a, insignal, axis)
    elif (type == 'Ltt'):
        b = [1, -2, 1]
        outsignal = normderfactor(stddev, 2, normdermethod, gamma) * \
          lfilter(b, a, insignal, axis)
    else:
        raise ValueError('Unknown temporal derivative method %s', type)

    return outsignal


def quasiquad(insignal, stddev, normdermethod='variance', Gamma=0.0, C = 1/sqrt(2), axis=-1):
    # The temporal quasi quadrature measure defined in Equation (55) in
    #
    # Lindeberg (2018) "Dense scale selection over spatial, temporal and
    # spatio-temporal domains", SIAM Journal on Imaging Sciences, 11(1): 407–441.
    #
    # Note, however, that the default value of C has been determined for
    # the non-causal Gaussian scale space, whereas the application of the
    # quasi quadrature measure in combination with the time-causal scale space
    # obtained by convolution with the time-causal limit kernel may call
    # for a better determination of the relative weighting factor C and then also as
    # function of the distribution parameter c of the time-causal limit kernel.
    
    # The scale normalization by dividing by stddev^(2*Gamma) is transfered
    # to other gamma parameters for the 1:st- and 2:nd-order derivatives
    Lt = tempder(insignal, 'Lt', stddev, normdermethod, 1-Gamma/2, axis)
    Ltt = tempder(insignal, 'Ltt', stddev, normdermethod, 1-Gamma/4, axis)
    outsignal = Lt*Lt + C*Ltt*Ltt

    return outsignal

    
def deltafcn1D(length):
    # Discrete delta function in 1-D
    
    signal = np.zeros(length)
    signal[0] = 1.0
    
    return signal


def mean1D(signal):
    # Computes the temporal mean of a non-negative temporal signal
    
    if (signal.ndim != 1):
        raise ValueError('Only implemented for 1-D signals')

    size = signal.shape[0]
    t = np.linspace(0, size-1, size)

    tmean = np.sum(np.sum(t * signal))/np.sum(np.sum(signal))

    return tmean


def variance1D(signal):
    # Computes the temporal variance of a non-negative temporal signal
    
    if (signal.ndim != 1):
        raise ValueError('Only implemented for 1-D signals')

    size = signal.shape[0]
    t = np.linspace(0, size-1, size)

    t2mom = np.sum(np.sum(t * t * signal))/np.sum(np.sum(signal))
    tmean = mean1D(signal)

    return t2mom - tmean*tmean


def whitenoisesignal1D(length):
    
    signal = np.random.normal(0.0, 1.0, length)

    return signal


def browniannoisesignal1D(length):

    whitenoise = whitenoisesignal1D(length)

    return np.cumsum(whitenoise)


if __name__ == '__main__': 
    main() 




    

    
        
        



