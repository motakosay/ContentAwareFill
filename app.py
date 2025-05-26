import cv2
import numpy as np
from skimage import restoration
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

# Initialize variables
image = None
mask = None
filled = None
brush_radius = 5

def draw_circle(event, x, y, flags, param):
    global image, mask
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), brush_radius, (0, 0, 0), -1)
        cv2.circle(mask, (x, y), brush_radius, 255, -1)

def content_aware_fill(image_path):
    global image, mask, filled

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    filled = np.copy(image)

    cv2.setMouseCallback('Content-Aware Fill', draw_circle)

    while True:
        display = np.copy(image)
        cv2_imshow(display)  # Use cv2_imshow() instead of cv2.imshow()

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Press 'Esc' to exit
            break
        elif k == ord('s'):
            # Perform content-aware filling
            filled = restoration.inpaint.inpaint_biharmonic(filled, mask, multichannel=(image.shape[-1] == 3))
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            image = np.copy(filled)

    plt.imshow(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    image_path = '/content/ContentAwareFill/t.jpg'
    content_aware_fill(image_path)
