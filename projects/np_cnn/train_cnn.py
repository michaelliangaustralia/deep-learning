import utils
import datasets

import IPython

# Datasets
mnist = datasets.load_dataset("mnist", split=["train"])
test_image = mnist[0]['image'][0]

# Convolution Layer
conv = utils.ConvolutionLayer(3, 5)
conv_patches = conv.patches_generator(test_image)
conv_output = conv.forward_prop(test_image)

# Max Pooling Layer
pool = utils.MaxPoolingLayer(2)
pool_patches = pool.patches_generator(test_image)
pool_output = pool.forward_prop(conv_output)