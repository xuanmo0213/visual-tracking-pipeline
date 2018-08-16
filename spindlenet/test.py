from argparse import ArgumentParser
import cv2
import copy
import numpy as np
import caffe

params = {}
params['use_gpu'] = 0
params['caffe_path'] = '/home/nvidia/caffe/python'
params['caffe_model'] = '/home/nvidia/sqz_spindle/sqz_inception_iter_70000.caffemodel'
params['deploy'] = '/home/nvidia/sqz_spindle/deploy.prototxt'
params['height'] = 96
params['width'] = 96
params['mean_value'] = [103.939, 116.779, 123.68]

img1 = 
img2 =

def init_caffe_model(params):
	caffe.set_device(params['use_gpu'])
	caffe.set_mode_gpu()
	net = caffe.Net(params['deploy'],params['caffe_model'], caffe.TEST)
	return net

def spindlenet(net, img_path):
        img = cv2.imread(img_path)

        assert img is not None
        assert img.shape[0] > 1
        assert img.shape[1] > 1
        assert img.shape[2] == 3

        x_scale = 1.0
        y_scale = 1.0
        img = cv2.resize(img, (params['W'], params['H']))
        flip = 1
        img = img[:,::flip,:]
        img[:,:,0] -= params['mean'][0]
        img[:,:,1] -= params['mean'][1]
        img[:,:,2] -= params['mean'][2]
        img = img.transpose((2,0,1))

        net.blobls['data'].data[0,...] = img
        output = net.forward()
        print output
        return output['fc7'][0]

def calcMean(x, y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        return x_mean, y_mean

def calcPearson(x, y):
        x_mean, y_mean = calcMean(x,y)
        n = len(x)
        sumTop = 0.0
        sumBottom = 0.0
        x_pow = 0.0
        y_pow = 0.0
        for i in range(n):
                sumTop += (x[i] - x_mean) * (y[i] - y_mean)
        for i in range(n):
                x_pow += np.power(x[i] - x_mean, 2)
        for i in range(n):
                y_pow += np.power(y[i] - y_mean, 2)
        sumBottom = np.sqrt(x_pow * y_pow)
        p = sumTop/sumBottom
        return p

def test(net, img1, img2):
        raw_input("Press Enter to Continue...")
        feature_1 = copy.deepcopy(spindlenet(net, img1))
        feature_2 = copy.deepcopy(spindlenet(net, img2))
        prr = calcPearson(feature_1, feature_2)
        print "The Pearson Similarity is " + str(prr)

def main(args):
        net = init_caffe_model(params)
        test(net, img1, img2)

if __name__ == '__main__':
        parser = ArgumentParser(
        description="Gen region proposal datalist")
        args = parser.parse_args()
        main(args)
