from rembg import remove
from PIL import Image

def remove_background(input_path, output_path):
    # Remove background
    input_image = Image.open(input_path)
    output_image = remove(input_image)

    # Create a new image with a black background
    new_image = Image.new("RGBA", output_image.size, (0, 0, 0, 255))
    new_image.paste(output_image, (0, 0), output_image)

    # Save the final image with a black background
    new_image.save(output_path)
