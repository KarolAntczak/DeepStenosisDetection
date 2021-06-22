import pickle

import numpy as np
from keras.models import load_model
from tensorflow.python.keras.backend import get_session


def load_dataset(filename):
    dataset = pickle.load(open(filename, 'rb'))
    return dataset


def generate_output_set(dataset, assigned_class):
    return np.full(dataset.shape[0], assigned_class)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

negative_dataset = load_dataset("datasets natural/1394x32x32 negative.pickle")[0:125]
positive_dataset = load_dataset("datasets natural/125x32x32 positive.pickle")

x_set = np.concatenate((negative_dataset,
                        positive_dataset)).reshape((250, 32, 32, 1))
y_set = np.concatenate((generate_output_set(negative_dataset, 0),
                        generate_output_set(positive_dataset, 1)))

x_set, y_set = unison_shuffled_copies(x_set, y_set)

print("X set: %s, Y set: %s" % (x_set.shape, y_set.shape))

# nauczona
model = load_model("networks/network.net")
print(model.evaluate(x_set[125:250], y_set[125:250]))

# strojenie
model = load_model("networks/network.net")
session = get_session()
for layer in model.layers:
     for v in layer.__dict__:
         v_arg = getattr(layer,v)
         if hasattr(v_arg,'initializer') and getattr(v_arg, 'initializer') is not None:
             initializer_method = getattr(v_arg, 'initializer')
             initializer_method.run(session=session)
             print('reinitializing layer {}.{}'.format(layer.name, v))
model.fit(x_set[0:125], y_set[0:125], batch_size=125, epochs=1000, verbose=0)
print(model.evaluate(x_set[125:250], y_set[125:250]))

# nauczona + strojenie
model = load_model("networks/network.net")
model.fit(x_set[0:125], y_set[0:125], batch_size=125, epochs=1000, verbose=0)
print(model.evaluate(x_set[125:250], y_set[125:250]))
