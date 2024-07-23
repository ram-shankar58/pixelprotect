import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

a = 1.0
b = 3.0
c = 1.0
d = 5.0
r = 0.001
x_r = -1.6
I = 3.6

def split_channel(channel, r, num_shares):
    width, height = channel.size
    shares = [np.zeros((width, height), dtype=np.uint8) for _ in range(num_shares)]
    x, y, z = np.random.random(3)
    x_logistic = np.random.random()
    for i in range(width):
        for j in range(height):
            pixel = channel.getpixel((i, j))
            x, y, z = hindmarsh_rose_model(x, y, z, a, b, c, d, r, x_r, I)
            chaotic_maps = [int(r * (1 - r) * pixel * (x + y + z)) for _ in range(num_shares - 1)]
            chaotic_maps = np.clip(chaotic_maps, 0, 255)
            for k, share in enumerate(shares[:-1]):
                share[i, j] = chaotic_maps[k]
            shares[-1][i, j] = np.clip(pixel - sum(chaotic_maps), 0, 255)
            if num_shares % 2 == 0:
                x_logistic = r * x_logistic * (1 - x_logistic)
                if not np.isnan(x_logistic) and not np.isinf(x_logistic):
                    for share in shares:
                        share[i, j] = np.clip(int(r * (1 - r) * share[i, j] * x_logistic), 0, 255)
    for i, share in enumerate(shares):
        share_image = Image.fromarray(share)
        share_image.save(f"share{i+1}.png")
    return shares

def combine_channel(shares):
    if len(shares) % 2 == 0:
        x_logistic = np.random.random()
        for i in range(shares[0].shape[0]):
            for j in range(shares[0].shape[1]):
                x_logistic = r * x_logistic * (1 - x_logistic)
                if not np.isnan(x_logistic) and not np.isinf(x_logistic):
                    for share in shares:
                        share[i, j] = int(share[i, j] / (x_logistic + 1e-10))
    combined_channel = sum(shares)
    combined_channel = np.clip(combined_channel, 0, 255)
    combined_channel = combined_channel.astype(np.uint8)
    combined_channel = Image.fromarray(combined_channel)
    return combined_channel

def hindmarsh_rose_model(x, y, z, a, b, c, d, r, x_r, I):
    dt = 0.01
    dx = y - a * x**3 + b * x**2 - z + I
    dy = c - d * x**2 - y
    dz = r * (x - x_r - z)
    x += dx * dt
    y += dy * dt
    z += dz * dt
    return x, y, z

def split_image(image, r, num_shares):
    width, height = image.size
    r_channel, g_channel, b_channel = image.split()
    r_shares = split_channel(r_channel, r, num_shares)
    g_shares = split_channel(g_channel, r, num_shares)
    b_shares = split_channel(b_channel, r, num_shares)
    return r_shares, g_shares, b_shares

def combine_image(r_shares, g_shares, b_shares):
    combined_r = combine_channel(r_shares)
    combined_g = combine_channel(g_shares)
    combined_b = combine_channel(b_shares)
    combined_image = Image.merge("RGB", (combined_r, combined_g, combined_b))
    return combined_image

def register(image, num_shares):
    print("Please select a point in the original image:")
    image_array = np.array(image)
    plt.imshow(image_array)
    plt.axis("off")
    plt.show(block=False)
    point = plt.ginput(1)  # Allow the user to select a single point
    plt.close()

    # Convert the selected point to integer coordinates
    x, y = map(int, point[0])

    # Save the selected point as a separate image
    point_image = image.crop((x, y, x + 1, y + 1))
    point_image.save("selected_point.png")

    # Split the image into shares
    split_image(image, r, num_shares)

    return x, y  # Return the coordinates of the selected point

def dhash(image, hash_size=8):
    image = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS).convert("L")
    pixels = np.array(image)
    diff = pixels[:, 1:] > pixels[:, :-1]
    decimal_value = 0
    for (i, j), value in np.ndenumerate(diff):
        decimal_value += int(value) * (2 ** (i * hash_size + j))
    return hex(decimal_value)[2:]

def authenticate(original_image, num_shares, x, y, hash_size=8, max_distance=20):
    # Prompt the user to input the file paths of the share images
    share_paths = []
    for i in range(num_shares):
        share_path = input(f"Enter the file path for share{i+1}.png: ")
        share_paths.append(share_path)

    # Reconstruct the image from uploaded shares
    r_shares = []
    g_shares = []
    b_shares = []
    for share_path in share_paths:
        share_image = Image.open(share_path).convert("RGB")
        r_channel, g_channel, b_channel = share_image.split()
        r_shares.append(np.array(r_channel))
        g_shares.append(np.array(g_channel))
        b_shares.append(np.array(b_channel))

    combined_image = combine_image(r_shares, g_shares, b_shares)

    combined_image.show()

    if np.array_equal(np.array(original_image), np.array(combined_image)):
        print("Reconstructed image matches the original image.")

        # Ask the user to select the same point in the original image
        print("Please select the same point in the original image:")
        original_image_array = np.array(original_image)
        plt.imshow(original_image_array)
        plt.axis("off")
        plt.show(block=False)
        point = plt.ginput(1)  # Allow the user to select a single point
        plt.close()

        # Convert the selected point to integer coordinates
        x_selected, y_selected = map(int, point[0])

        # Calculate the distance between the chosen coordinates and the set coordinates
        distance = abs(x_selected - x) + abs(y_selected - y)

        if distance > max_distance:
            print(f"Authentication failed: Point selection distance too large ({distance} > {max_distance}).")
            return False

        point_image = original_image.crop((x_selected, y_selected, x_selected + 1, y_selected + 1))
        original_hash = dhash(point_image, hash_size)

        # Compare the hash values of the selected point in each share
        for i in range(1, num_shares + 1):
            share_image = Image.open(share_paths[i-1]).convert("RGB")
            share_point_image = share_image.crop((x_selected, y_selected, x_selected + 1, y_selected + 1))
            share_hash = dhash(share_point_image, hash_size)

            # Compare the hash values using a threshold (adjust as needed)
            if hamming_distance(original_hash, share_hash) < hash_size // 2:
                print(f"Authentication successful!")
                return True

        print("Authentication failed: Selected point does not match in one or more shares.")
    else:
        print("Reconstructed image does not match the original image. Authentication process aborted.")
    return False

def hamming_distance(hash1, hash2):
    return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')

# Example usage
original_image = Image.open('tank.png')
r = 99
tolerance = 15

# Ask the user for the number of shares
num_shares = int(input("Enter the number of shares: "))

# Register the user
x, y = register(original_image, num_shares)

# Authenticate the user
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if authenticate(original_image, num_shares, x, y, hash_size=8, max_distance=20):
        print('Authentication successful!')
    else:
        print('Authentication failed!')
