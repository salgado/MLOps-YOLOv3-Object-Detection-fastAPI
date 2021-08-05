# Import required libraries
import os
import cv2
import cvlib as cv
import glob


# Function that detects objects and draw a box around them
def detect_object(filepath, model='yolov3-tiny', conf_level=0.5):
    img = cv2.imread(filepath)  # Load image
    boundary_box, label, conf = cv.detect_common_objects(img, confidence=conf_level,
                                                         model=model)  # Perform object detection
    # Print the results of detected objects in the image
    for l, c in zip(label, conf):
        print(f'In image {filepath.split("/")[-1]} detected object is {l} with confidence level of {c}')

    output_img = cv.object_detection.draw_bbox(img, boundary_box, label, conf)  # Create output image containing the box
    save_results(output_img, filepath.split("/")[-1])  # save the output img


# Function for saving the results
def save_results(img, filename):
    # Make directory to save the results
    path = "Results"
    if not os.path.exists(path):
        os.mkdir(path)
    cv2.imwrite(f'Results/{filename}', img)  # Save the img in the directory


path = glob.glob(os.path.join("data", "*.jpg"))  # Path to all files in data folder
for file in path:
    detect_object(file)
