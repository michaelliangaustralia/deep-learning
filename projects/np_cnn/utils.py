import numpy as np
from PIL import Image
import IPython

class ConvolutionLayer:
    """Convolution Layer
    """
    def __init__(self, kernel_num, kernel_size):
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.kernels = np.random.randn(kernel_num, kernel_size, kernel_size) / (kernel_size ** 2)

    def patches_generator(self, image):
        image = np.array(image)
        image_h, image_w = image.shape
        self.image = image

        for h in range(image_h - self.kernel_size + 1):
            for w in range(image_w - self.kernel_size + 1):
                patch = image[h:(h + self.kernel_size), w:(w+self.kernel_size)]
                yield patch, h, w

    def forward_prop(self, image):
        image = np.array(image)
        image_h, image_w = image.shape
        convolution_output = np.zeros((image_h - self.kernel_size + 1, image_w - self.kernel_size + 1, self.kernel_num))
        for patch, h, w in self.patches_generator(image):
            convolution_output[h,w] = np.sum(patch * self.kernels, axis=(1,2))
        return convolution_output

    def back_prop(self, dE_dY, alpha):
        dE_dk = np.zeros(self.kernels.shape)
        for patch, h, w in self.patches_generator(self.image):
            for f in range(self.kernel_num):
                dE_dk[f] += patch * dE_dY[h, w, f]
        self.kernels -= alpha * dE_dk
        return dE_dk

class MaxPoolingLayer:
    """Max Pooling Layer
    """
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def patches_generator(self, image):
        image = np.array(image)
        output_h = image.shape[0] // self.kernel_size
        output_w = image.shape[1] // self.kernel_size
        self.image = image

        for h in range(output_h):
            for w in range(output_w):
                patch = image[(h*self.kernel_size):(h*self.kernel_size + self.kernel_size), (w*self.kernel_size):(w*self.kernel_size + self.kernel_size)]
                yield patch, h, w

    def forward_prop(self, image):
        image = np.array(image)
        image_h, image_w, num_kernels = image.shape
        max_pooling_output = np.zeros((image_h//self.kernel_size, image_w//self.kernel_size, num_kernels))
        for patch, h, w in self.patches_generator(image):
            max_pooling_output[h, w] = np.amax(patch, axis=(0,1))
        return max_pooling_output

    def back_prop(self, dE_dY):
        dE_dk = np.zeros(self.image.shape)

        for patch, h, w in self.patches_generator(self.image):
            image_h, image_w, num_kernels = patch.shape
            max_val = np.amax(patch, axis=(0,1))

            for idx_h in range(image_h):
                for idx_w in range(image_w):
                    for idx_k in range(num_kernels):
                        if patch[idx_h, idx_w, idx_k] == max_val[idx_k]:
                            dE_dk[h*self.kernel_size + idx_h, w*self.kernel_size + idx_w, idx_k] = dE_dY[h, w, idx_k]
        
        return dE_dk