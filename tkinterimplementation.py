import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import warnings
import numpy as np
import tkinter.simpledialog as simpledialog
from tkinter import filedialog
warnings.filterwarnings("ignore", category=DeprecationWarning)

a = 1.0
b = 3.0
c = 1.0
d = 5.0
r = 0.001
x_r = -1.6
I = 3.6

l=[]
def addxynum(x,y,num_shares):
    l.append(x)
    l.append(y)
    l.append(num_shares)

def getxynum():
    return l[0], l[1], l[2]

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
            for k, share in enumerate(shares[:-1]):
                share[i, j] = abs(chaotic_maps[k])%256
            shares[-1][i, j] = (pixel - sum(chaotic_maps)) % 256
            if num_shares % 2 == 0:
                x_logistic = r * x_logistic * (1 - x_logistic)
                for share in shares:
                    share[i, j] = int(r * (1 - r) * share[i, j] * x_logistic)
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
    combined_r = combine_channel(r_shares)
    combined_g = combine_channel(g_shares)
    combined_b = combine_channel(b_shares)
    combined_image = Image.merge("RGB", (combined_r, combined_g, combined_b))
    combined_image = combined_image.transpose(Image.TRANSPOSE)
    return combined_image

def registernaor(image, num_shares):
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
    combined_image = split_image(image, r, num_shares)
    combined_image.save("combined_image.png")

    return x, y  # Return the coordinates of the selected point

def dhash(image, hash_size=8):
    image = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS).convert("L")
    pixels = np.array(image)
    diff = pixels[:, 1:] > pixels[:, :-1]
    decimal_value = 0
    for (i, j), value in np.ndenumerate(diff):
        decimal_value += int(value) * (2 ** (i * hash_size + j))
    return hex(decimal_value)[2:]

def authenticatenaor(original_image, num_shares, x, y, hash_size=8, max_distance=20, share_images=[]):
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
        print(f"Authentication failed: Point is too far from the original selection.")
        return False

    # Hash of the selected point in the original image
    point_image = original_image.crop((x_selected, y_selected, x_selected + 1, y_selected + 1))
    original_hash = dhash(point_image, hash_size)

    # Hash check across shares
    for i, share_image in enumerate(share_images):
        share_point_image = share_image.crop((x_selected, y_selected, x_selected + 1, y_selected + 1))
        share_hash = dhash(share_point_image, hash_size)

        # Compare each share’s hash to the original image hash
        if hamming_distance(original_hash, share_hash) > hash_size // 4:
            print(f"Authentication failed: Hash mismatch in share {i + 1}.")
            return False

    # Cross-verify hashes between the shares to ensure consistency
    for i in range(len(share_images) - 1):
        for j in range(i + 1, len(share_images)):
            hash_i = dhash(share_images[i].crop((x_selected, y_selected, x_selected + 1, y_selected + 1)), hash_size)
            hash_j = dhash(share_images[j].crop((x_selected, y_selected, x_selected + 1, y_selected + 1)), hash_size)
            if hamming_distance(hash_i, hash_j) > hash_size // 4:
                print(f"Authentication failed: Share {i + 1} and Share {j + 1} do not match.")
                return False

    print("Authentication successful! All shares match and are consistent with the original image.")
    return True



def hamming_distance(hash1, hash2):
    return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')
def open_image():
    filepath = filedialog.askopenfilename()
    image = Image.open(filepath)
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(image=photo)
    label.image = photo  # keep a reference!
    label.pack()
    return image

def register():
    original_image = open_image()
    r = 99
    tolerance = 15

    # Ask the user for the number of shares
    num_shares = simpledialog.askinteger("Input", "Enter the number of shares:",parent=root)

    # Register the user
    x, y = registernaor(original_image, num_shares)
    addxynum(x,y,num_shares)

def authenticate():
    x, y, num_shares = getxynum()
    original_image = Image.open('tank.png')
    
    # Ask the user to open the share images
    share_images = []
    for i in range(num_shares):
        share_image = open_image()
        share_images.append(share_image)

    # Authenticate the user
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # Check the authentication
        if authenticatenaor(original_image, num_shares, x, y, hash_size=8, max_distance=20, share_images=share_images):
            print('Authentication successful!')
        else:
            print('Authentication failed!')

root = tk.Tk()
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()
register_button = tk.Button(root, text="Register", command=register)
register_button.pack()
authenticate_button = tk.Button(root, text="Authenticate", command=authenticate)
authenticate_button.pack()
root.mainloop()
