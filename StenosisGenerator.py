import random

import math
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.mlab as mlab

from Stenosis.Curve import make_bezier


def draw_gradient(image):
    inner_color = random.randint(0, 25) + 25
    outer_color = random.randint(60, 80) + 25

    center = [random.randint(-image.width, image.width*2), random.randint(-image.height, image.height*2)]
    for y in range(image.height):
        for x in range(image.width):
            # Find the distance to the center

            distance_to_center = math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

            # Make it on a scale from 0 to 1
            distance_to_center = float(distance_to_center) / (math.sqrt(2) * image.width / 2)

            # Calculate r, g, and b values
            color = outer_color * distance_to_center + inner_color * (1 - distance_to_center)

            # Place the pixel
            image.putpixel((x, y), int(color))


def draw_vein(image, with_stenosis=False):
    start_point = (random.randint(-image.width, image.width*2),0)
    inter_point = (random.randint(0, image.width), random.randint(0, image.height))
    inter_point2 = (random.randint(0, image.width), random.randint(0, image.height))
    end_point =  (random.randint(-image.width, image.width*2),image.height)

    points = make_bezier([start_point, inter_point, inter_point2, end_point])
    color = (0, 0, 0, random.randint(100, 100))
    basic_width = random.randint(1, 12) * 0.1

    poly = Image.new('RGBA', (image.width, image.height))
    pdraw = ImageDraw.Draw(poly)

    for index, point in enumerate(points):
        stenosis_factor = max(0, 1 - mlab.normpdf(index,len(points) / 2,4)*11) if with_stenosis else 1
        width = basic_width * stenosis_factor
        pdraw.ellipse((point[0] - width, point[1] - width, point[0] + width, point[1] + width), fill=color)

    image.paste(poly, mask=poly)


def add_white_noise(image, delta):
    for x in range(0, image.width):
        for y in range(0, image.height):
            col = random.randint(-delta, delta)
            color = image.getpixel((x, y))
            image.putpixel((x, y),color + col)


def draw_cardiographic_image(width, height, number_of_veins, number_of_stenoses):
    image = Image.new('L', (width, height), 80)

    draw_gradient(image)  # Struktura wewnętrzna ciała
    draw_gradient(image)

    for i in range(0, number_of_stenoses): # Żyły ze stenozami
        draw_vein(image, with_stenosis=True)
    for i in range(0, number_of_veins):  # Żyły
        draw_vein(image)

    add_white_noise(image, 16)  # Szum pomiaru
    image = image.filter(ImageFilter.GaussianBlur(random.randint(7,8) * 0.1))  # Rozmycie

    add_white_noise(image, 1)  # Szum obrazu
    return image


