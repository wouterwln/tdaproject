from utils import read_data
from MorseSmale import MorseSmaleComplex
import os
import matplotlib.pyplot as plt

river_height_data = read_data("braided-river")
print("Data Read")

for i, frame in enumerate(river_height_data):
    if i % 10 == 0:
        plt.figure(figsize=(100, 10), dpi=100)
        ms = MorseSmaleComplex(frame, 1)
        ms.plot("output/desc_frame_{}.png".format(i))
        plt.clf()
        print("Frame {} written".format(i))

