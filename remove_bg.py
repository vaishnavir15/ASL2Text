from rembg import remove
from PIL import Image

input_path = 'hand_without_landmarks.png'
output_path = 'hand_without_landmarks_clear.png'

# Load the input image
input_image = Image.open(input_path)

# Use rembg to remove the background
output_image = remove(input_image)

# Create a new image with a black background
new_image = Image.new("RGBA", output_image.size, (0, 0, 0, 255))

# Paste the transparent image onto the black background
new_image.paste(output_image, (0, 0), output_image)

# Save the final image with a black background
new_image.save(output_path)
