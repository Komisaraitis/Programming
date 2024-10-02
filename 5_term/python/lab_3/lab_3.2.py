from PIL import Image
import argparse
from matplotlib import pyplot as plt
from skimage import io
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dirpath", required=True)

with Image.open(parser.parse_args().dirpath) as img:
    r, g, b = img.split()
    histogram_r = r.histogram()
    histogram_g = g.histogram()
    histogram_b = b.histogram()

img = io.imread(parser.parse_args().dirpath)

red_hist, red_bins = np.histogram(img[:, :, 0], bins=256, range=(0, 256))
green_hist, green_bins = np.histogram(img[:, :, 1], bins=256, range=(0, 256))
blue_hist, blue_bins = np.histogram(img[:, :, 2], bins=256, range=(0, 256))

# Построение RGB-гистограмм
plt.figure(figsize=(10, 10))

plt.subplot(4, 2, (1, 7))
plt.axis("off")
plt.imshow(img)

plt.subplot(4, 2, 2)
plt.bar(range(256), histogram_r, color="pink", width=1)
plt.bar(range(256), histogram_g, color="pink", width=1)
plt.bar(range(256), histogram_b, color="pink", width=1)
plt.title("Гистограмма")
plt.xlabel("Интенсивность")
plt.ylabel("Частота")

plt.subplot(4, 2, 4)
plt.bar(red_bins[:-1], red_hist, color="red", width=1)
plt.title("R гистограмма")
plt.xlabel("Интенсивность")
plt.ylabel("Частота")

plt.subplot(4, 2, 6)
plt.bar(green_bins[:-1], green_hist, color="green", width=1)
plt.title("G гистограмма")
plt.xlabel("Интенсивность")
plt.ylabel("Частота")

plt.subplot(4, 2, 8)
plt.bar(blue_bins[:-1], blue_hist, color="blue", width=1)
plt.title("B гистограмма")
plt.xlabel("Интенсивность")
plt.ylabel("Частота")

plt.tight_layout()
plt.show()
