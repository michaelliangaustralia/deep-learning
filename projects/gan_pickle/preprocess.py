from PIL import Image
import glob

# Resize all jpg images to 256 x 256
for file in glob.glob("data/pickle_jpg/*.jpg"):
    img = Image.open(file)
    img = img.resize((256, 256))
    img.save(file)
