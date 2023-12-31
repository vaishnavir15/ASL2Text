from rembg import remove
from PIL import Image

def remove_background(input_path, output_path):
    # Removes the background, then adds a black canvas
    input_image = Image.open(input_path)
    output_image = remove(input_image)

    new_image = Image.new("RGBA", output_image.size, (0, 0, 0, 255))
    new_image.paste(output_image, (0, 0), output_image)

    new_image.save(output_path)
