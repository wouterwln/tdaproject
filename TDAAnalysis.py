import numpy as np
from utils import read_data
from MorseSmale import MorseSmaleComplex

river_height_data = read_data("braided-river")
print("Data Read")
ms = MorseSmaleComplex(river_height_data[400], 1)
ms.plot()

