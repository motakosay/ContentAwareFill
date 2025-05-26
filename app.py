# import cv2
# import numpy as np

# # Initialize variables
# drawing = False
# mode = True  # If True, draw a rectangle. Press 'm' to toggle to curve
# ix, iy = -1, -1
# brush_size = 5
# image = None
# mask = None
# filled = None

# def draw(event, x, y, flags, param):
#     global ix, iy, drawing, mode, image, mask, filled

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y

#         if mode:
#             cv2.rectangle(image, (ix, iy), (x, y), (0, 0, 0), -1)
#             cv2.rectangle(mask, (ix, iy), (x, y), (255, 255, 255), -1)
#         else:
#             cv2.circle(image, (x, y), brush_size, (0, 0, 0), -1)
#             cv2.circle(mask, (x, y), brush_size, (255, 255, 255), -1)

#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             if mode:
#                 cv2.rectangle(image, (ix, iy), (x, y), (0, 0, 0), -1)
#                 cv2.rectangle(mask, (ix, iy), (x, y), (255, 255, 255), -1)
#             else:
#                 cv2.circle(image, (x, y), brush_size, (0, 0, 0), -1)
#                 cv2.circle(mask, (x, y), brush_size, (255, 255, 255), -1)

#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         if mode:
#             cv2.rectangle(image, (ix, iy), (x, y), (0, 0, 0), -1)
#             cv2.rectangle(mask, (ix, iy), (x, y), (255, 255, 255), -1)
#         else:
#             cv2.circle(image, (x, y), brush_size, (0, 0, 0), -1)
#             cv2.circle(mask, (x, y), brush_size, (255, 255, 255), -1)

# def content_aware_fill(image_path):
#     global image, mask, filled

#     image = cv2.imread(image_path)
#     mask = np.zeros_like(image)
#     filled = np.copy(image)

#     cv2.namedWindow('Content-Aware Fill')
#     cv2.setMouseCallback('Content-Aware Fill', draw)

#     while True:
#         cv2.imshow('Content-Aware Fill', image)
#         k = cv2.waitKey(1) & 0xFF
#         if k == 27:  # Press 'Esc' to exit
#             break
#         elif k == ord('m'):
#             mode = not mode  # Toggle drawing mode
#         elif k == ord('s'):
#             filled = cv2.inpaint(filled, mask[:, :, 0], 3, cv2.INPAINT_TELEA)
#             mask = np.zeros_like(image)
#             image = np.copy(filled)

#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     image_path = 'test.jpg'
#     content_aware_fill(image_path)


import cv2
import numpy as np
from skimage import restoration

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
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    filled = np.copy(image)

    cv2.namedWindow('Content-Aware Fill')
    cv2.setMouseCallback('Content-Aware Fill', draw_circle)

    while True:
        cv2.imshow('Content-Aware Fill', image)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Press 'Esc' to exit
            break
        elif k == ord('s'):
            # Perform content-aware filling
            filled = restoration.inpaint.inpaint_biharmonic(filled, mask, multichannel=True)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            image = np.copy(filled)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_path = '/content/ContentAwareFill/t.jpg'
    content_aware_fill(image_path)
