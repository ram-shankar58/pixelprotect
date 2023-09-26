from PIL import Image
import numpy as np

def arnold_cat_map(image_array, iterations):
    n = len(image_array)
    x, y = np.meshgrid(range(n), range(n))
    x_map = (2*x + y) % n
    y_map = (x + y) % n
    for _ in range(iterations):
        image_array = image_array[x_map, y_map]
    return image_array

def pad_to_square(image_array):
    max_dim = max(image_array.shape)
    padded_array = np.pad(image_array, ((0, max_dim - image_array.shape[0]), (0, max_dim - image_array.shape[1]), (0, 0)), mode='constant')
    return padded_array

image = Image.open("./images/img1.jpeg")
imagearray = np.array(image)

# Pad the image to make it square
imagearray = pad_to_square(imagearray)

# Apply Arnold's Cat Map
encryptedimagearray = arnold_cat_map(imagearray, iterations=10)

# Create an image from the encrypted array
encryptedimage = Image.fromarray(encryptedimagearray.astype(np.uint8))

# Save the encrypted image
encryptedimage.save("./images/encryptedimg.jpeg")
