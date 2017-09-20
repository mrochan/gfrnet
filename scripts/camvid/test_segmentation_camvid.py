import numpy as np
import scipy
import argparse
from scipy import misc
caffe_root = '/home/amirul/caffe-gfrnet/'  # Change this to the absolute directoy to Caffe
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()
caffe.set_device(1)

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

fname = '/mnt/vana/amirul/code_release/cvpr2017_seg/data/CamVid/test.txt'

with open(fname) as f:
    labelFiles = f.read().splitlines()

for i in range(0, args.iter):

    net.forward()

    image = net.blobs['data'].data
    label = net.blobs['label'].data
    predicted = net.blobs['prob'].data
    image = np.squeeze(image[0,:,:,:])
    output = np.squeeze(predicted[0,:,:,:])
    ind = np.argmax(output, axis=0)
    
    
    r = ind.copy()
    g = ind.copy()
    b = ind.copy()
    r_gt = label.copy()
    g_gt = label.copy()
    b_gt = label.copy()

    Sky = [128,128,128]
    Building = [128,0,0]
    Pole = [192,192,128]
    Road = [128,64,128]
    Pavement = [60,40,222]
    Tree = [128,128,0]
    SignSymbol = [192,128,128]
    Fence = [64,64,128]
    Car = [64,0,128]
    Pedestrian = [64,64,0]
    Bicyclist = [0,128,192]
    Unlabelled = [0,0,0]

    label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    for l in range(0,11):
        r[ind==l] = label_colours[l,0]
        g[ind==l] = label_colours[l,1]
        b[ind==l] = label_colours[l,2]
        r_gt[label==l] = label_colours[l,0]
        g_gt[label==l] = label_colours[l,1]
        b_gt[label==l] = label_colours[l,2]

    rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
    rgb[:,:,0] = r/255.0
    rgb[:,:,1] = g/255.0
    rgb[:,:,2] = b/255.0
    '''rgb_gt = np.zeros((ind.shape[0], ind.shape[1], 3))
    rgb_gt[:,:,0] = r_gt/255.0
    rgb_gt[:,:,1] = g_gt/255.0
    rgb_gt[:,:,2] = b_gt/255.0'''

    image = image/255.0

    image = np.transpose(image, (1,2,0))
    output = np.transpose(output, (1,2,0))
    image = image[:,:,(2,1,0)]

    labelFile = labelFiles[i].split(' ')[1]
    labelname = labelFile.split('testannot/')
    
    misc.toimage(ind, cmin=0.0, cmax=255).save('/mnt/vana/amirul/code_release/cvpr2017_seg/predictions/CamVid/prediction_camvid_gate_release_code/'+labelname[1])
    #scipy.misc.toimage(rgb, cmin=0.0, cmax=1).save('/net/crane-08/data/mrochan/Deconvolution_SceneUnderstanding_1/prediction_camvid_gate_rgb/'+labelname[1])

    

print 'Success!'

