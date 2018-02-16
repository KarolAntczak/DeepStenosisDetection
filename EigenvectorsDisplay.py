import pickle
import matplotlib.pyplot as plt

dataset = pickle.load(open("datasets/5000x32x32 negative.pickle", 'rb'))
dataset = dataset.squeeze()
fig = plt.figure()

for i in range(0, 4):
    fig.add_subplot(2, 2, i+1)
    plt.imshow(dataset[i], cmap="gray", )
    plt.axis('off')
fig.tight_layout()
plt.show()
