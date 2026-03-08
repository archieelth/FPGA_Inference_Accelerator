import csv
import numpy as np
from PIL import Image

input_csv = "data/mnist_train.csv"
output_hex = "input.hex"

num = 2

with open(input_csv, newline="") as csvfile:
    reader = csv.reader(csvfile)
    row = list(reader)[num]

# Remove label
pixels = np.array(row[1:], dtype=np.uint8)

# Reshape to 28x28
image_array = pixels.reshape((28, 28))

# Create image
image = Image.fromarray(image_array, mode="L")
image.show()

with open(input_csv, newline="") as csvfile:
    reader = csv.reader(csvfile)
    row = list(reader)[num]

# remove the first item
data = row[1:]

with open(output_hex, "w") as hexfile:
    for item in data:
        hexfile.write(f"{item}\n")
