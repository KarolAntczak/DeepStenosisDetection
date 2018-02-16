import pickle

from Keras.keras.backend import *
from Keras.keras.layers import *
from Keras.keras.models import *
from Keras.keras.optimizers import SGD, Adam


def load_dataset(filename):
    dataset = pickle.load(open(filename, 'rb'))
    return dataset


def generate_output_set(dataset, assigned_class):
    return np.full(dataset.shape[0], assigned_class)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

negative_dataset = load_dataset("datasets/5000x32x32 negative.pickle")
positive_dataset = load_dataset("datasets/5000x32x32 positive.pickle")

x_set = np.concatenate((negative_dataset,
                        positive_dataset))
y_set = np.concatenate((generate_output_set(negative_dataset, 0),
                        generate_output_set(positive_dataset, 1)))

x_set, y_set = unison_shuffled_copies(x_set, y_set)

print("X set: %s, Y set: %s" % (x_set.shape, y_set.shape))

model = Sequential()

model.add(Flatten(input_shape=(x_set.shape[1], x_set.shape[2], x_set.shape[3])))
model.add(Dense(128, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='random_uniform'))

model.compile(loss='binary_crossentropy', optimizer=SGD(momentum=0.8), metrics=['binary_accuracy'])

print(model.summary())

model.fit(x_set, y_set, batch_size=100, epochs=1000, verbose=2, validation_split=0.2)
model.save("networks/128d1d.net", overwrite=True)
