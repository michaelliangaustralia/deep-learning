from PIL import Image
import glob

# Resize all jpg images to 256 x 256
for file in glob.glob("IMG_1164.png"):
    img = Image.open(file)
    img = img.resize((256, 256))
    img.save(file)

# convert png to jpg
for file in glob.glob("IMG_1164.png"):
    img = Image.open(file)
    img = img.convert("RGB")
    img.save(file.replace(".png", ".jpg"))