import numpy as np
from PIL import Image

def split_channel(channel, num_shares):
    width, height = channel.size
    shares = [np.zeros((width, height), dtype=np.uint8) for _ in range(num_shares)]
    random_shares = np.random.randint(0, 256, (num_shares - 1, width, height), dtype=np.uint8)

    for i in range(width):
        for j in range(height):
            pixel = channel.getpixel((i, j))
            sum_shares = sum(random_shares[:, i, j])
            shares[-1][i, j] = (pixel - sum_shares) % 256
            for k in range(num_shares - 1):
                shares[k][i, j] = random_shares[k, i, j]

    return shares

def combine_channel(shares):
    combined_channel = sum(shares) % 256
    combined_channel = combined_channel.astype(np.uint8)
    return Image.fromarray(combined_channel)

def split_image(image, num_shares):
    r_channel, g_channel, b_channel = image.split()
    r_shares = split_channel(r_channel, num_shares)
    g_shares = split_channel(g_channel, num_shares)
    b_shares = split_channel(b_channel, num_shares)
    return r_shares, g_shares, b_shares

def combine_image(r_shares, g_shares, b_shares):
    combined_r = combine_channel(r_shares)
    combined_g = combine_channel(g_shares)
    combined_b = combine_channel(b_shares)
    combined_image = Image.merge("RGB", (combined_r, combined_g, combined_b))
    return combined_image

def save_shares(shares, base_name):
    for i, share in enumerate(shares):
        share_image = Image.fromarray(share)
        share_image.save(f"{base_name}_share{i+1}.png")

def load_shares(base_name, num_shares):
    shares = []
    for i in range(num_shares):
        share_image = Image.open(f"{base_name}_share{i+1}.png").convert("L")
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
r_shares, g_shares, b_shares = split_image(original_image, num_shares)

# Save the shares
save_shares(r_shares, "r_channel")
save_shares(g_shares, "g_channel")
save_shares(b_shares, "b_channel")

# Load the shares
loaded_r_shares = load_shares("r_channel", num_shares)
loaded_g_shares = load_shares("g_channel", num_shares)
loaded_b_shares = load_shares("b_channel", num_shares)

# Combine the shares to reconstruct the image
reconstructed_image = combine_image(loaded_r_shares, loaded_g_shares, loaded_b_shares)

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
