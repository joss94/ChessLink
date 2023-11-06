import cv2

image_path = "/workspace/CL/data/test_images/1000011472.jpg"

image = cv2.imread(image_path)
image = cv2.convertScaleAbs(image, 5, 1.5)

cv2.imwrite("/workspace/CL/sandbox.jpg", image)