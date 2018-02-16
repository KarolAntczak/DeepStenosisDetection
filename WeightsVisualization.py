from __future__ import print_function

import tensorflow
from scipy.misc import imsave
import numpy as np
import time
from keras import backend as K

# dimensions of the generated pictures for each filter.
from Keras.keras.models import load_model

img_width = 32
img_height = 32

# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
layer_name = 'conv2d_1'


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# load model
model = load_model("networks/network.net")

print('Model loaded.')

for layer in model.layers:
    for v in layer.__dict__:
        v_arg = getattr(layer, v)
        if hasattr(v_arg, 'initializer'):
            initializer_method = getattr(v_arg, 'initializer')
            initializer_method.run(session=K.get_session())
            print('reinitializing layer {}.{}'.format(layer.name, v))

keras_learning_phase = tensorflow.get_default_graph().get_tensor_by_name('dropout_1/keras_learning_phase:0')

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


kept_filters = []

print(layer_dict)
for filter_index in range(0, 8):
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss

    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img, keras_learning_phase], [loss, grads])

    # step size for gradient ascent
    step = 1.
    # we start from a gray image with some random noise
    input_img_data = np.random.random((1, img_width, img_height, 1))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    for i in range(20000):
        loss_value, grads_value = iterate([input_img_data, False])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)

    # decode the resulting input image
    img = deprocess_image(input_img_data[0])
    kept_filters.append((img, []))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)

# fill the picture with our saved filters
for i, (img, loss) in enumerate(kept_filters):
    img = img[:, :, 0]
    imsave('visualization/%s/filters_%d.png' % (layer_name, i), img)
