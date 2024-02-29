# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import os
import sys
import onnxruntime
import onnx
import numpy as np
from PIL import Image, ImageTk
from object_detection import ObjectDetection
import tempfile
import cv2
import argparse
import time
import threading

MODEL_FILENAME = 'model.onnx'
LABELS_FILENAME = 'labels.txt'

# Visualization parameters
row_size = 20  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 255)  # red
font_size = 1
font_thickness = 1
fps_calculation_interval = 30  # Calculate FPS every 30 frames

# Global variable to store the state of the button
is_on = False

# Global variables to store camera settings
brightness = 100
contrast = 50
saturation = 50


def on_brightness_change(value, cap):
    global brightness
    brightness = value
    print(f"brightness: {brightness}")
    adjust_camera_settings(cap)

def on_contrast_change(value, cap):
    global contrast
    contrast = value
    print(f"contrast: {contrast}")
    adjust_camera_settings(cap)

def on_saturation_change(value, cap):
    global saturation
    saturation = value
    print(f"saturation: {saturation}")
    adjust_camera_settings(cap)

class ONNXRuntimeObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""
    def __init__(self, model_filename, labels, num_threads, threshold, overlap, max_detections):
        super(ONNXRuntimeObjectDetection, self).__init__(labels, num_threads, threshold, overlap, max_detections)
        model = onnx.load(model_filename)
        with tempfile.TemporaryDirectory() as dirpath:
            temp = os.path.join(dirpath, os.path.basename(MODEL_FILENAME))
            model.graph.input[0].type.tensor_type.shape.dim[-1].dim_param = 'dim1'
            model.graph.input[0].type.tensor_type.shape.dim[-2].dim_param = 'dim2'
            onnx.save(model, temp)
            self.session = onnxruntime.InferenceSession(temp)
        self.input_name = self.session.get_inputs()[0].name
        self.is_fp16 = self.session.get_inputs()[0].type == 'tensor(float16)'
        
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))

        if self.is_fp16:
            inputs = inputs.astype(np.float16)

        outputs = self.session.run(None, {self.input_name: inputs})
        return np.squeeze(outputs).transpose((1,2,0)).astype(np.float32)

def capture_frame(cap):
    grabbed = cap.grab()
    if not grabbed:
        print('ERROR: Unable to grab frame from camera.')
        sys.exit('ERROR: Unable to grab frame from camera.')
    _, frame = cap.retrieve()
    if frame is None:
        print('ERROR: Unable to grab frame from camera.')
        sys.exit('ERROR: Unable to retrieve frame from camera.')
    return frame

def draw_detection_results(image, detection_result):
    for detection in detection_result:
        bounding_box = detection['boundingBox']
        left = int(bounding_box['left'] * image.shape[1])
        top = int(bounding_box['top'] * image.shape[0])
        width = int(bounding_box['width'] * image.shape[1])
        height = int(bounding_box['height'] * image.shape[0])

        # Draw bounding box rectangle
        cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 2)

        # Draw label
        label = f"{detection['tagName']} {detection['probability']:.2f}"
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def toggle(event, x, y, flags, param):
    global is_on
    if event == cv2.EVENT_LBUTTONDOWN:
        is_on = not is_on

def adjust_camera_settings(cap):
    global brightness, contrast, saturation
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness )
    cap.set(cv2.CAP_PROP_CONTRAST, contrast )
    cap.set(cv2.CAP_PROP_SATURATION, saturation )

def display_fps(image, fps):
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)
    return image

def run(model: str, camera_id: int, width: int, height: int, num_threads: int, threshold: float, overlap: float, max_detections: int) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
        model: Name of the TFLite object detection model.
        camera_id: The camera id to be passed to OpenCV.
        width: The width of the frame captured from the camera.
        height: The height of the frame captured from the camera.
        num_threads: The number of CPU threads to run the model.
        threshold: Prediction probability for displaying
        max_detections: Maximum number of objects to display
    """
    
        # Load the custom vision labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [label.strip() for label in f.readlines()]

    # Load the custom vision ML model 
    od_model = ONNXRuntimeObjectDetection(model, labels, num_threads, threshold, overlap, max_detections) 
    
    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    adjust_camera_settings(cap)

    # Set up mouse callback function for the OpenCV window
    cv2.namedWindow('PiHQ camera')
    cv2.setMouseCallback('PiHQ camera', toggle)

    # Create sliders to adjust camera settings
    cv2.createTrackbar('Brightness', 'PiHQ camera', brightness, 200, lambda value: on_brightness_change(value, cap))
    cv2.createTrackbar('Contrast', 'PiHQ camera', contrast, 100, lambda value: on_contrast_change(value, cap))
    cv2.createTrackbar("Saturation", 'PiHQ camera', saturation, 200, lambda value: on_saturation_change(value, cap))
    
    # Read the first frame to determine its dimensions
    success, image = cap.read()
    if not success:
        sys.exit(
            'ERROR: Unable to read from webcam. Please verify your webcam settings.'
        )
        
    global is_on

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        frame = capture_frame(cap)
        counter += 1

        # Adjust camera settings
        adjust_camera_settings(cap)

        # Display toggle status on the frame
        toggle_text = "ON" if is_on else "OFF"
        cv2.putText(frame, f"WBC Detect: {toggle_text}", (160, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_on else (0, 0, 255), 2)

        if is_on:
            # Convert to PIL Image
            pil_image = Image.fromarray(frame)

            # Run object detection using the ObjectDetection instance
            detection_result = od_model.predict_image(pil_image)

            if detection_result is not None:
                # Draw bounding boxes based on the detection result
                draw_detection_results(frame, detection_result)
        
        # Calculate the FPS periodically
        if counter % fps_calculation_interval == 0:
            end_time = time.time()
            fps = fps_calculation_interval / (end_time - start_time)
            start_time = time.time()

        frame_with_fps = display_fps(frame, fps)
        cv2.imshow('PiHQ camera', frame_with_fps)
        
        # Stop the program if the ESC key is pressed.
        if (cv2.waitKey(1) == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='model.onnx')
    parser.add_argument(
      '--cameraId', 
      help='Id of camera.',
      required=False, 
      type=int, 
      default=0)
    parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
    parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
    parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
    parser.add_argument(
      '--threshold',
      help="Probability threshold.",
      required=False,
      type=float, 
      default=0.5)
    parser.add_argument(
      '--overlap',
      help="Overlap threshold.",
      required=False,
      type=float, 
      default=0.4)
    parser.add_argument(
      '--max_detections',
      help="Maximum number of detections.",
      required=False,
      type=int, 
      default=8)
    
    args = parser.parse_args()
    
    try:
        run(args.model, args.cameraId, args.frameWidth, args.frameHeight, args.numThreads, args.threshold, args.overlap, args.max_detections)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()

