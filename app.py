import cv2
import numpy as np
from skimage import restoration
from skimage.restoration import inpaint
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

def content_aware_fill(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Instead of mouse interaction, use a predefined mask or let the user manually input coordinates.
    # Example: Simulating a small removal area in the center
    h, w = image.shape[:2]
    cv2.rectangle(mask, (w//3, h//3), (2*w//3, 2*h//3), 255, -1)

    # Perform content-aware filling
    filled = inpaint.inpaint_biharmonic(image, mask, multichannel=(image.shape[-1] == 3))

    # Show original and filled images
    print("Original Image:")
    cv2_imshow(image)
    print("Processed Image:")
    cv2_imshow(filled)

    plt.imshow(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Test function
image_path = '/content/ContentAwareFill/t.jpg'
content_aware_fill(image_path)
