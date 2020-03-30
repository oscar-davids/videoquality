
import numpy
import re
import sys
import scipy.misc

import psnr
import ssim
import vifp

def img_greyscale(img):
    return 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]

def img_read_yuv(src_file, width, height):
    y_img = numpy.fromfile(src_file, dtype=numpy.uint8, count=(width * height)).reshape( (height, width) )
    u_img = numpy.fromfile(src_file, dtype=numpy.uint8, count=((width/2) * (height/2))).reshape( (height/2, width/2) )
    v_img = numpy.fromfile(src_file, dtype=numpy.uint8, count=((width/2) * (height/2))).reshape( (height/2, width/2) )
    return (y_img, u_img, v_img)

ref_file = sys.argv[1]
dist_file = sys.argv[2]

# Inputs are image files
ref = scipy.misc.imread(ref_file, flatten=True).astype(numpy.float32)
dist = scipy.misc.imread(dist_file, flatten=True).astype(numpy.float32)

width, height = ref.shape[1], ref.shape[0]
print "Comparing %s to %s, resolution %d x %d" % (ref_file, dist_file, width, height)

psnr_value = psnr.psnr(ref, dist)
print "PSNR=%f" % (psnr_value)

ssim_value = ssim.ssim_exact(ref/255, dist/255)
print "SSIM=%f" % (ssim_value)

vifp_value = vifp.vifp_mscale(ref, dist)
print "VIFP=%f" % (vifp_value)

