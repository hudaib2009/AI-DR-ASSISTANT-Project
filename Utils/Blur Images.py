from PIL import Image
from PIL import ImageFilter

# Open an image file
# Replace "path/to/your/image.jpg" with the actual path to your image file
# Note: remove " " from image path 
img = Image.open(r'path/to/your/image.jpg')

# Display the original image
img.show()

# Apply a simple filter (blur) to the same image
img_blurred = img.filter(ImageFilter.BLUR)

# Display the modified image
img_blurred.show()

# Close the images (optional)
img.close()
img_blurred.close()