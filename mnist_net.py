import argparse
import gzip
import numpy as np
import math
import pickle
import random

import skimage
from skimage import transform

#import matplotlib.pyplot as plt

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
    

def random_transformation(data):
    data = data.reshape((28, 28))

    w = data.shape[0]/2
    
    rotation = random.uniform(-0.2, 0.2)
    shear_rotation = random.uniform(-4*math.pi, 4*math.pi)
    shear_amount = random.uniform(-0.2, 0.2)
    trans_x = random.uniform(-1, 1)
    trans_y = random.uniform(-1, 1)
    scale_x = random.uniform(0.5, 1.2)
    scale_y = random.uniform(0.5, 1.2)

    t0 = transform.AffineTransform(translation=(w,w)).params
    s = transform.AffineTransform(scale=(w,w)).params
    r0 = transform.AffineTransform(rotation=shear_rotation).params
    s =  transform.AffineTransform(shear=shear_amount).params
    r1 = transform.AffineTransform(rotation=-shear_rotation+rotation).params
    #t1 = transform.AffineTransform(rotation=random.uniform(-0.3, 0.3)).params
    t2 = transform.AffineTransform(translation=(-w + trans_x, -w + trans_y)).params

    
    return transform.warp(data, t0.dot(s.dot(r0.dot(s.dot(r1.dot(t2)))))).reshape((784,))

def args():
    parser = argparse.ArgumentParser(description='Neural network for mnist database.')
    parser.add_argument('mnist_path', help='Path for gzipped mnist training/test set pickle.')
    parser.add_argument('output_path', default='output.pkl', help='Path to save output pickle.')
    return parser.parse_args()

def data_to_image(data):
    return Image.fromarray((data*255).reshape(IMAGE_WIDTH, IMAGE_HEIGHT))

def image_to_data(img):
    return np.asarray(img).reshape(784)/255

#def random_transformation(data):
#    image = data_to_image(data)
#    image = image.rotate(random.randint(-20, 20))
    return image_to_data(image)

def transform_examples(data):
    return [
        (
            data[0][i]/255, 
            np.array([1 if j == data[1][i] else 0 for j in range(10)])
        ) 
        for i in range(len(data[0]))]

def training_generator(base_examples):
    while True:
        example = random.choice(base_examples)
        #if random.random() < 0.1:
        #    yield example
        #else:
        yield (random_transformation(example[0]), example[1])

def main(mnist_path, output_path):
    f = gzip.open(mnist_path, 'rb')
    training_set, test_set = pickle.load(f, encoding='latin1')
 
    training_examples = transform_examples(training_set)
    test_examples = transform_examples(test_set)

    print("Creating new training data")
    generated_training_examples = []
    for i in range(800000):
        example = random.choice(training_examples)
        example = (random_transformation(example[0]), example[1])
        #from matplotlib import pyplot as plt
        #plt.imshow(example[0].reshape(28, 28))
        #plt.show()
        generated_training_examples.append(example)
    print(len(generated_training_examples))
    training_examples += generated_training_examples
    print(len(training_examples))


    network = NeuralNetwork([PIXEL_COUNT, 128, 64, 64, NUM_CHARACTERS], output_path)
    network.train(training_examples, test_examples)

    pickle.dump(network, open(output_path, 'wb'))

if __name__ == "__main__":
    args = args()
    main(args.mnist_path, args.output_path)
