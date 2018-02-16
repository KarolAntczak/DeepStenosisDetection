from Stenosis.StenosisGenerator import draw_cardiographic_image
import numpy as np
import pickle


def generate_case(image_width, image_height, number_of_veins, number_of_stenoses):
    image = draw_cardiographic_image(image_width, image_height, number_of_veins, number_of_stenoses)
    data = np.array(list(image.getdata())) / 255
    return data.reshape(image_width, image_height, 1)


def generate_dataset(image_width, image_height, size, with_stenoses=False):
    dataset = []
    for i in range(0, size):
        print("generating %i/%i" % (i, size))
        veins = 0 if with_stenoses else 3
        stenoses = 3 if with_stenoses else 0
        case = generate_case(image_width, image_height, veins, stenoses)
        dataset.append(case)
    return np.array(dataset)


def generate_and_save_dataset(image_width, image_height, size, filename, with_stenoses=False):
    dataset = generate_dataset(image_width, image_height, size, with_stenoses)
    pickle.dump(dataset, open(filename, 'wb'), pickle.HIGHEST_PROTOCOL)

print("Starting generating data")
generate_and_save_dataset(32, 32, 5000, "datasets/5000x32x32 negative.pickle", False)
generate_and_save_dataset(32, 32, 5000, "datasets/5000x32x32 positive.pickle", True)
print("Finished")
