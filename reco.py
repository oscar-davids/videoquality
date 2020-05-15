#RECO: Relative Polar Edge Coherence
#RECO calculate module

import numpy
from numpy import sqrt, pi
import scipy.ndimage.filters

def Laguerre_Gauss_Circular_Harmonic_3_0(size, sigma):
    x = numpy.linspace(-size/2.0, size/2.0, size)
    y = numpy.linspace(-size/2.0, size/2.0, size)
    xx, yy = numpy.meshgrid(x, y)
    
    r = numpy.sqrt(xx*xx + yy*yy)
    gamma = numpy.arctan2(yy, xx)
    l30 = - (1/6.0) * (1 / (sigma * sqrt(pi))) * numpy.exp( -r*r / (2*sigma*sigma)) * (sqrt(r*r/(sigma*sigma)) ** 3) * numpy.exp( -1j * 3 * gamma )
    return l30

def Laguerre_Gauss_Circular_Harmonic_1_0(size, sigma):
    x = numpy.linspace(-size/2.0, size/2.0, size)
    y = numpy.linspace(-size/2.0, size/2.0, size)
    xx, yy = numpy.meshgrid(x, y)
    
    r = numpy.sqrt(xx*xx + yy*yy)
    gamma = numpy.arctan2(yy, xx)
    l10 = - (1 / (sigma * sqrt(pi))) * numpy.exp( -r*r / (2*sigma*sigma)) * sqrt(r*r/(sigma*sigma)) * numpy.exp( -1j * gamma )
    return l10

"""
Polar edge coherence map
Same size as source image
"""
def pec(img):
    # TODO scale parameter should depend on resolution
    l10 = Laguerre_Gauss_Circular_Harmonic_1_0(17, 2)
    l30 = Laguerre_Gauss_Circular_Harmonic_3_0(17, 2)
    y10 = scipy.ndimage.filters.convolve(img, numpy.real(l10)) + 1j * scipy.ndimage.filters.convolve(img, numpy.imag(l10))
    y30 = scipy.ndimage.filters.convolve(img, numpy.real(l30)) + 1j * scipy.ndimage.filters.convolve(img, numpy.imag(l30))
    pec_map = - (numpy.absolute(y30) / numpy.absolute(y10)) * numpy.cos( numpy.angle(y30) - 3 * numpy.angle(y10) )
    return pec_map

"""
Edge coherence metric
Just one number summarizing typical edge coherence in this image.
"""
def eco(img):
    l10 = Laguerre_Gauss_Circular_Harmonic_1_0(17, 2)
    l30 = Laguerre_Gauss_Circular_Harmonic_3_0(17, 2)
    y10 = scipy.ndimage.filters.convolve(img, numpy.real(l10)) + 1j * scipy.ndimage.filters.convolve(img, numpy.imag(l10))
    y30 = scipy.ndimage.filters.convolve(img, numpy.real(l30)) + 1j * scipy.ndimage.filters.convolve(img, numpy.imag(l30))
    eco = numpy.sum( - (numpy.absolute(y30) * numpy.absolute(y10)) * numpy.cos( numpy.angle(y30) - 3 * numpy.angle(y10) ) )
    return eco

"""
Relative edge coherence
Ratio of ECO
"""
def reco(img1, img2):
    C = 1 # TODO what is a good value?
    return (eco(img2) + C) / (eco(img1) + C)
