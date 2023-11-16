import numpy as np
from PIL import Image
import warnings
import os
import secrets
from getpass import getpass

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hindmarsh-Rose model parameters
a = 1.0
b = 3.0
c = 1.0
d = 5.0
r = 0.001
x_r = -1.6
I = 3.6

def split_channel(channel, r, num_shares, pin):
    # Get channel dimensions
    width, height = channel.size

    # Split the channel into num_shares shares
    shares = [np.zeros((width, height), dtype=np.uint8) for _ in range(num_shares)]

    # Initialize the Hindmarsh-Rose model
    x, y, z = np.random.random(3)

    for i in range(width):
        for j in range(height):
            pixel = channel.getpixel((i, j))

            # Update the Hindmarsh-Rose model
            x, y, z = hindmarsh_rose_model(x, y, z, a, b, c, d, r, x_r, I)

            # Calculate the chaotic maps and split the pixel value among the shares
            chaotic_maps = [int(r * (1 - r) * pixel * (x + y + z)) for _ in range(num_shares - 1)]
            for k, share in enumerate(shares[:-1]):
                share[i, j] = chaotic_maps[k]
            shares[-1][i, j] = pixel - sum(chaotic_maps)

    # Save each share as an image with a random name containing the PIN digits
    for i, share in enumerate(shares):
        share_image = Image.fromarray(share)
        share_name = f"share{secrets.choice(pin[:2])}{pin}{secrets.choice(pin[1:])}.png"
        share_image.save(share_name)

    return shares

def combine_channel(shares):
    # Combine the shares to reconstruct the original channel
    combined_channel = sum(shares)
    combined_channel = np.clip(combined_channel, 0, 255)
    combined_channel = combined_channel.astype(np.uint8)
    combined_channel = Image.fromarray(combined_channel)

    return combined_channel

def hindmarsh_rose_model(x, y, z, a, b, c, d, r, x_r, I):
    dt = 0.01  # Time step
    dx = y - a * x**3 + b * x**2 - z + I
    dy = c - d * x**2 - y
    dz = r * (x - x_r - z)

    x += dx * dt
    y += dy * dt
    z += dz * dt

    return x, y, z

def split_image(image, r, num_shares, pin):
    # Get image dimensions
    width, height = image.size

    # Split the image into RGB channels
    r_channel, g_channel, b_channel = image.split()

    # Split each channel into num_shares shares with PIN digits in the name
    r_shares = split_channel(r_channel, r, num_shares, pin)
    g_shares = split_channel(g_channel, r, num_shares, pin)
    b_shares = split_channel(b_channel, r, num_shares, pin)

    # Combine the shares for each channel
    combined_r = combine_channel(r_shares)
    combined_g = combine_channel(g_shares)
    combined_b = combine_channel(b_shares)

    # Merge the combined channels into a single RGB image
    combined_image = Image.merge("RGB", (combined_r, combined_g, combined_b))
    combined_image = combined_image.transpose(Image.TRANSPOSE)

    return combined_image

def register(image, num_shares, pin):
    combined_image = split_image(image, r, num_shares, pin)
    combined_image.save("combined_image.png")

def authenticate(original_image, num_shares, pin):
    combined_image = Image.open("combined_image.png")

    if np.array_equal(original_image, combined_image):
        return True
    else:
        return False

# Example usage
original_image = Image.open('tank.png')
r = 99  # Chaotic map parameter

# Ask the user for the number of shares and the PIN
num_shares = int(input("Enter the number of shares: "))
pin = getpass("Enter the PIN: ")

# Register the user
register(original_image, num_shares, pin)

# Authenticate the user
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if authenticate(original_image, num_shares, pin):
        print('Authentication successful!')
    else:
        print('Authentication failed!')
