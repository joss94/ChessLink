# STEP 1: Import the necessary modules.
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=4)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("gt.png")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the classification result. In this case, visualize it.
print(detection_result)
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# cv2_imwrite("test.jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))