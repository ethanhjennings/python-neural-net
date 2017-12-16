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
HIDDEN_LAYER_SIZE=16

NUM_CHARACTERS = 10 # representing 10 digits

def convolution(matrix, kernel):
    w, h = kernel.shape[0], kernel.shape[1]
    copy = np.pad(matrix, ((w//2, w//2),(h//2, h//2)), mode='constant')

    #import pdb; pdb.set_trace()
    for (r, c), val in np.ndenumerate(matrix):
        matrix[r, c] = np.sum(copy[r:r+w, c:c+h] * kernel)

    return matrix
    

def flow_field(data):
    copy = np.zeros((28, 28))

    a = random.uniform(0, 2*math.pi)
    b = random.uniform(0, 2*math.pi)
    c = random.uniform(0, 2*math.pi)
    d = random.uniform(0, 2*math.pi)
    e = random.uniform(-0.3, 0.3)
    f = random.uniform(-0.3, 0.3)
    g = random.uniform(-0.3, 0.3)
    h = random.uniform(-0.3, 0.3)
    i = random.uniform(-8, 8)
    j = random.uniform(-8, 8)

    for x in range(28):
        for y in range(28):
            v_x = x + 2*math.cos(e*x + a)*math.sin(f*y + b)
            v_y = y + 2*math.sin(g*x + c)*math.cos(h*y + d)
            v_x = max(0, min(27, v_x))
            v_y = max(0, min(27, v_y))
            copy[y,x] = data[int(v_y), int(v_x)]
    return copy

def random_transformation(data):
    data = data.reshape((28, 28))

    #from matplotlib import pyplot as plt
    #plt.imshow(data)
    #plt.show()
    
    #data = flow_field(data)
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
    #plt.imshow(data)
    #plt.show()
    return data.reshape((784,))

def args():
    parser = argparse.ArgumentParser(description='Neural network for mnist database.')
    parser.add_argument('mnist_path', help='Path for gzipped mnist training/test set pickle.')
    parser.add_argument('output_path', default='output.pkl', help='Path to save output pickle.')
    parser.add_argument('--activation', default='relu', help='Activation function to use, can be either sigmoid or relu')
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
        if random.random() < 0.1:
            yield example
        else:
            yield (random_transformation(example[0]), example[1])

def main(mnist_path, output_path, activation):
    f = gzip.open(mnist_path, 'rb')
    training_set, test_set = pickle.load(f, encoding='latin1')
 
    training_examples = transform_examples(training_set)
    test_examples = transform_examples(test_set)

    print("Creating new training data")
    generated_training_examples = []
    for i in range(2500000):
        example = random.choice(training_examples)
        example = (random_transformation(example[0]), example[1])
        generated_training_examples.append(example)
    print(len(generated_training_examples))
    training_examples += generated_training_examples
    print(len(training_examples))

    if activation == "sigmoid":
        activation = neural_net.sigmoid
        d_activation = neural_net.d_sigmoid
    elif activation == "relu":
        activation = neural_net.relu
        d_activation = neural_net.d_relu
    elif activation == "elu":
        activation = neural_net.elu
        d_activation = neural_net.d_elu


    network = NeuralNetwork([PIXEL_COUNT, 256, 256, NUM_CHARACTERS], output_path, act_func=activation, act_func_deriv=d_activation)
    network.train(training_examples, test_examples)

    pickle.dump(network, open(output_path, 'wb'))

if __name__ == "__main__":
    args = args()
    main(args.mnist_path, args.output_path, args.activation)
