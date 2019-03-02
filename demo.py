#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.color as color
from skimage.io import  imsave
import scipy.ndimage.interpolation as sni
import caffe

#matplotlib inline
plt.rcParams['figure.figsize'] = (12, 6)

gpu_id = 0
caffe.set_mode_cpu()
#caffe.set_device(cpu_id)

# Select desired model
net = caffe.Net('colorization_deploy_v2.prototxt', 'colorization_release_v2.caffemodel', caffe.TEST)
# net = caffe.Net('../models/colorization_deploy_v2.prototxt', '../models/colorization_release_v2_norebal.caffemodel', caffe.TEST)
# net = caffe.Net('../models/colorization_deploy_v1.prototxt', '../models/colorization_release_v1.caffemodel', caffe.TEST)
# If you are training your own network, you may replace the *.caffemodel path with your trained network.

(H_in,W_in) = net.blobs['data_l'].data.shape[2:] # get input shape
(H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape

print 'Input dimensions: (%i,%i)'%(H_in,W_in)
print 'Output dimensions: (%i,%i)'%(H_out,W_out)

pts_in_hull = np.load('colorization/resources/pts_in_hull.npy') # load cluster centers
net.params['class8_ab'][0].data[:,:,0,0] = pts_in_hull.transpose((1,0)) # populate cluster centers as 1x1 convolution kernel
print 'Annealed-Mean Parameters populated'

# load the original image
#img_rgb = caffe.io.load_image('colorization/colorization/demo/imgs/ansel_adams3.jpg')
img_rgb = caffe.io.load_image('1.jpg')
print "load gray image .jpg"

img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
img_l = img_lab[:,:,0] # pull out L channel
(H_orig,W_orig) = img_rgb.shape[:2] # original image size

# create grayscale version of image (just for displaying)
img_lab_bw = img_lab.copy()
img_lab_bw[:,:,1:] = 0
img_rgb_bw = color.lab2rgb(img_lab_bw)

# resize image to network input size
img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
img_lab_rs = color.rgb2lab(img_rs)
img_l_rs = img_lab_rs[:,:,0]

# show original image, along with grayscale input to the network
img_pad = np.ones((H_orig,W_orig/10,3))
imginput = np.hstack((img_rgb, img_pad, img_rgb_bw))
print 'imginput', imginput.shape
plt.subplot(1, 2, 1)
plt.imshow(imginput)
plt.title('(Left) Loaded image   /   (Right) Grayscale input to network')
plt.axis('off');

net.blobs['data_l'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
net.forward() # run network

ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0)) # this is our result
ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1)) # upsample to match size of original image L
img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
img_rgb_out = (255*np.clip(color.lab2rgb(img_lab_out),0,1)).astype('uint8') # convert back to rgb
print "img_rgb_out", img_rgb_out.shape
imsave('x.jpg', img_rgb_out)
print "done"

img_inandout = np.hstack((imginput, img_rgb_out))
plt.subplot(133)
plt.imshow(img_rgb_out)
plt.title('Output img')
plt.axis('off');
plt.show()
