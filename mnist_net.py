# Entry point for training. This script also generates augmented data with various transforms.

import argparse
import gzip
import numpy as np
import math
import pickle
import random

import skimage
from skimage import transform

import neural_net
from neural_net import NeuralNetwork

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
PIXEL_COUNT = IMAGE_WIDTH * IMAGE_HEIGHT

NUM_CHARACTERS = 10 # representing 10 digits


def random_transformation(data):
    data = data.reshape((28, 28))

    w = data.shape[0]/2
    
    rotation = random.uniform(-0.6, 0.6)
    shear_rotation = random.uniform(-4*math.pi, 4*math.pi)
    shear_amount = random.uniform(-0.4, 0.4)
    trans_x = random.uniform(-1, 1)
    trans_y = random.uniform(-1, 1)
    scale_x = random.uniform(0.8, 1.2)
    scale_y = random.uniform(0.8, 1.2)

    t0 = transform.AffineTransform(translation=(w,w)).params
    s = transform.AffineTransform(scale=(scale_x,scale_y)).params
    sr = transform.AffineTransform(rotation=shear_rotation).params
    sh =  transform.AffineTransform(shear=shear_amount).params
    r0 = transform.AffineTransform(rotation=-shear_rotation+rotation).params
    t2 = transform.AffineTransform(translation=(-w + trans_x, -w + trans_y)).params
    
    data = transform.warp(data, t0.dot(s.dot(sr.dot(sh.dot(r0.dot(t2))))))
    return data.reshape((784,))

def args():
    parser = argparse.ArgumentParser(description='Neural network for mnist database.')
    parser.add_argument('mnist_path', help='Path for gzipped mnist training/test set pickle.')
    parser.add_argument('output_path', default='output.pkl', help='Path to save output pickle.')
    parser.add_argument('--activation', default='relu', help='Activation function to use, can be either sigmoid or relu')
    parser.add_argument('--hl-sizes', default='128x128', help="Sizes of the hidden layers  delimited by 'x'. Example: 128x128x128")
    return parser.parse_args()

def data_to_image(data):
    return Image.fromarray((data*255).reshape(IMAGE_WIDTH, IMAGE_HEIGHT))

def image_to_data(img):
    return np.asarray(img).reshape(784)/255

def transform_examples(data):
    return [
        (
            (data[0][i]/255).astype('float16'), 
            np.array([1 if j == data[1][i] else 0 for j in range(10)], dtype='int8')
        ) 
        for i in range(len(data[0]))]

def training_generator(base_examples):
    while True:
        example = random.choice(base_examples)
        if random.random() < 0.05:
            yield example
        else:
            yield (random_transformation(example[0]), example[1])

def main(mnist_path, output_path, activation, hl_sizes):
    f = gzip.open(mnist_path, 'rb')
    training_set, test_set = pickle.load(f, encoding='latin1')
 
    training_examples = transform_examples(training_set)
    test_examples = transform_examples(test_set)

    if activation == "sigmoid":
        activation = neural_net.sigmoid
        d_activation = neural_net.d_sigmoid
    elif activation == "relu":
        activation = neural_net.relu
        d_activation = neural_net.d_relu
    elif activation == "elu":
        activation = neural_net.elu
        d_activation = neural_net.d_elu

    network = NeuralNetwork([PIXEL_COUNT] + hl_sizes + [NUM_CHARACTERS], output_path, act_func=activation, act_func_deriv=d_activation)
    network.train(training_generator(training_examples), test_examples)

    pickle.dump(network, open(output_path, 'wb'))

if __name__ == "__main__":
    args = args()
    hl_sizes = [int(s) for s in args.hl_sizes.split('x')]
    main(args.mnist_path, args.output_path, args.activation, hl_sizes)
