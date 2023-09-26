from PIL import Image
import numpy as np

# Creating a chaotic key map:
def chaotickey(mapiteration):
    x = 0.1
    r = 3.9
    chaotic_values = []
    for i in range(mapiteration):
        x = r * x * (1 - x)  # Logistic map equation
        chaotic_values.append(x)
    
    # Convert the list of chaotic values into a binary key
    binkey = "".join([str(int(x > 0.5)) for x in chaotic_values])
    
    return binkey

keyiterations = 1000

image = Image.open("./images/img1.jpeg")
imagearray = np.array(image)

# Generate the chaotic encryption key
encryptionkey = chaotickey(keyiterations)

# Ensure that the encryption key is the same length as the image data
encryptionkey = (encryptionkey * (len(imagearray.flatten()) // len(encryptionkey) + 1))[:len(imagearray.flatten())]

# Convert the binary encryption key to a list of integers
encryptionkey = np.array([int(bit) for bit in encryptionkey], dtype=np.uint8)

# Perform the XOR operation between imagearray and encryptionkey
encryptedimagearray = np.bitwise_xor(imagearray.flatten(), encryptionkey).reshape(imagearray.shape)

# Create an image from the encrypted array
encryptedimage = Image.fromarray(encryptedimagearray.astype(np.uint8))

# Save the encrypted image
encryptedimage.save("./images/encryptedimg.jpeg")
