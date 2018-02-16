import pickle
import matplotlib.pyplot as plt

from Stenosis.StenosisGenerator import draw_cardiographic_image

dataset = pickle.load(open("datasets/10000x32x32 negative.pickle", 'rb'))
dataset = dataset.squeeze()
fig = plt.figure()

for i in range(0, 4):
    fig.add_subplot(2, 2, i+1)
    plt.imshow(draw_cardiographic_image(32,32,3,0), cmap="gray", )
    plt.axis('off')
fig.tight_layout()
plt.show()
