# Real-Time Emotion and Detection with YOLOv8 and FER

This project integrates real-time detection using YOLOv8 and emotion recognition with the FER (Facial Expression Recognition) library. The application detects and counts people in a webcam feed, analyzes their facial expressions, and provides counts for "Happy" and "Not Happy" individuals.


## Features

1. Real-Time Detection: Processes live webcam feed for object and emotion detection.
2. YOLOv8 Integration: Utilizes the powerful YOLOv8 model for detecting people.
3. Emotion Recognition: Leverages the FER library to classify emotions from detected faces.
4. Annotated Output: Displays bounding boxes with labels for detected objects and emotions.
5. Interactive Dashboard: Shows real-time counts for total individuals, happy, and not happy states.

## Dependencies
1. Python 3.8+
2. OpenCV: For video capture and frame processing.
3. Ultralytics YOLOv8: For object detection.
4. FER: For facial emotion recognition.
5. Supervision: For annotation and bounding box visualization.


## How It Works
1. The YOLOv8 model detects objects in the video feed.
2. Detected "person" bounding boxes are cropped and passed to the FER emotion detection model.
3. Emotions are classified (e.g., "Happy," "Neutral"), and the results are annotated on the frame.
4. Real-time counts of "Happy," "Not Happy," and "Total" individuals are displayed on the video feed.
