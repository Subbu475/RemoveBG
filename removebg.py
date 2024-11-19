from rembg import remove
from PIL import Image
import io


input_path = "download.jpg"
output_path = "output.png"

with open(input_path, "rb") as inp_file:
    input_image = inp_file.read()


output_image = remove(input_image)

# Save the output
with open(output_path, "wb") as out_file:
    out_file.write(output_image)

print(f"Background removed successfully! Saved to {output_path}.")
