#!/usr/bin/env python3
# pylint: disable=missing-docstring
import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_gradient_magnitude

from wordcloud import WordCloud, ImageColorGenerator

# get data directory
d = (os.path.dirname(__file__) if '__file__' in locals() else os.getcwd())

# load sample text
text = open(os.path.join(d, 'sample_text.txt')).read()

# load image
parrot_color = np.array(Image.open(os.path.join(d, 'sample_parrot.jpg')))

# subsample by factor of 3
parrot_color = parrot_color[::3, ::3]

# create mask white is "masked out"
parrot_mask = parrot_color.copy()
parrot_mask[parrot_mask.sum(axis = 2) == 0] = 255

# enforce boundaries between colors so they get less washed out
# edge detection in the image
edges = np.mean([gaussian_gradient_magnitude(parrot_color[:, :, i] / 255., 2) for i in range(3)], axis = 0)
parrot_mask[edges > .08] = 255

# create wordcloud
# relative_scaling=0 means the frequencies in the data are reflected less
wc = WordCloud(max_words = 2000, mask = parrot_mask, max_font_size = 40, random_state = 42, relative_scaling = 0)

# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(parrot_color)
wc.recolor(color_func=image_colors)

# create image
wc.to_file('word_cloud.png')

# plot image in console
plt.imshow(wc, interpolation='bilinear')
plt.show()