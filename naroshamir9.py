import numpy as np
from PIL import Image

def hindmarsh_rose(x, y, z, a=1, b=3, c=1, d=5, r=0.001, s=4, I=3.25, dt=0.01):
    dx = y + a * x**2 - b * x**3 - z + I
    dy = c - d * x**2 - y
    dz = r * (s * (x - 1.6) - z)
    x += dx * dt
    y += dy * dt
    z += dz * dt
    return x, y, z

def split_channel(channel, num_shares, a=1, b=3, c=1, d=5, r=0.001, s=4, I=3.25, dt=0.01):
    width, height = channel.size
    shares = [np.zeros((width, height), dtype=np.uint8) for _ in range(num_shares)]
    random_shares = np.random.randint(0, 256, (num_shares - 1, width, height), dtype=np.uint8)

    x, y, z = np.random.random(3)
    
    for i in range(width):
        for j in range(height):
            pixel = channel.getpixel((i, j))
            sum_shares = sum(random_shares[:, i, j])
            shares[-1][i, j] = (pixel - sum_shares) % 256
            
            for k in range(num_shares - 1):
                shares[k][i, j] = random_shares[k, i, j]
                x, y, z = hindmarsh_rose(x, y, z, a, b, c, d, r, s, I, dt)
                chaotic_value = int((x + y + z) * 1e6) % 256
                shares[k][i, j] = (shares[k][i, j] + chaotic_value) % 256
                shares[-1][i, j] = (shares[-1][i, j] - chaotic_value) % 256

    return shares

def combine_channel(shares, a=1, b=3, c=1, d=5, r=0.001, s=4, I=3.25, dt=0.01):
    x, y, z = np.random.random(3)

    width, height = shares[0].shape
    combined_channel = np.zeros((width, height), dtype=np.uint8)

    for i in range(width):
        for j in range(height):
            for k in range(len(shares) - 1):
                x, y, z = hindmarsh_rose(x, y, z, a, b, c, d, r, s, I, dt)
                chaotic_value = int((x + y + z) * 1e6) % 256
                shares[k][i, j] = (shares[k][i, j] - chaotic_value) % 256
                shares[-1][i, j] = (shares[-1][i, j] + chaotic_value) % 256

            combined_channel[i, j] = sum(share[i, j] for share in shares) % 256

    combined_channel = combined_channel.astype(np.uint8)
    return combined_channel

def split_image(image, num_shares):
    width, height = image.size
    shares = [np.zeros((width, height, 3), dtype=np.uint8) for _ in range(num_shares)]
    
    r_channel, g_channel, b_channel = image.split()
    
    r_shares = split_channel(r_channel, num_shares)
    g_shares = split_channel(g_channel, num_shares)
    b_shares = split_channel(b_channel, num_shares)
    
    for k in range(num_shares):
        for i in range(width):
            for j in range(height):
                shares[k][i, j, 0] = r_shares[k][i, j]
                shares[k][i, j, 1] = g_shares[k][i, j]
                shares[k][i, j, 2] = b_shares[k][i, j]
                
    return shares

def combine_image(shares):
    num_shares = len(shares)
    width, height, _ = shares[0].shape
    
    r_shares = [shares[k][:, :, 0] for k in range(num_shares)]
    g_shares = [shares[k][:, :, 1] for k in range(num_shares)]
    b_shares = [shares[k][:, :, 2] for k in range(num_shares)]
    
    combined_r = combine_channel(r_shares)
    combined_g = combine_channel(g_shares)
    combined_b = combine_channel(b_shares)
    
    combined_image = Image.merge("RGB", (Image.fromarray(combined_r), Image.fromarray(combined_g), Image.fromarray(combined_b)))
    return combined_image

def save_shares(shares, base_name):
    for i, share in enumerate(shares):
        share_image = Image.fromarray(share)
        share_image.save(f"{base_name}_share{i+1}.png")

def load_shares(base_name, num_shares):
    shares = []
    for i in range(num_shares):
        share_image = Image.open(f"{base_name}_share{i+1}.png")
        shares.append(np.array(share_image))
    return shares

def rotate_image(image, angle):
    return image.rotate(angle, expand=True)

def mirror_invert_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

# Example usage
original_image = Image.open('tank.png')

# Ask the user for the number of shares
num_shares = int(input("Enter the number of shares: "))

# Split the image into shares
shares = split_image(original_image, num_shares)

# Save the shares
save_shares(shares, "channel")

# Load the shares (user inputs the base name of the share images)
base_name = input("Enter the base name of the share images: ")

loaded_shares = load_shares(base_name, num_shares)

# Combine the shares to reconstruct the image
reconstructed_image = combine_image(loaded_shares)

# Rotate the reconstructed image 90 degrees to the right
rotated_image = rotate_image(reconstructed_image, -90)

# Mirror invert the rotated image horizontally
mirrored_image = mirror_invert_image(rotated_image)

# Save the mirrored image
mirrored_image.save('mirrored_image.png')

# Compare original and mirrored images (rotated back for comparison)
if np.array_equal(np.array(original_image), np.array(mirrored_image)):  # Rotate back for comparison
    print("Mirrored image matches the original image.")
else:
    print("Mirrored image does not match the original image.")
