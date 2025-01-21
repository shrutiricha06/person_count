import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
from fer import FER


def parse_arguments() -> argparse.Namespace:
    parser= argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-resolution", 
                        default= [1280,720],
                          nargs=2, 
                          type=int)
    
    args = parser.parse_args()
    return args

def main():
    args= parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    emotion_detector = FER(mtcnn=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width )
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


    model= YOLO("yolov8l.pt")


   

    while True:
        ret, frame = cap.read()       # image_path

        results= model(frame)

        happy_count = 0
        not_happy_count = 0
        total_count = 0

        box_annotator= sv.BoxAnnotator(
        thickness= 2,
       #text_thickness=2,
       # text_scale=1
    )
       

        for result in results:
            boxes= result.boxes.xyxy.cpu().numpy()
            confidences= result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names= result.names

        # counts the number of people detected
        person_class_id = 0  # Update if the "person" class ID is different
        person_indices = (class_ids == person_class_id)
        person_boxes = boxes[person_indices]




        labels = []
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]  # Crop the face region

                # Perform emotion detection
            emotions = emotion_detector.top_emotion(face)
            if emotions is not None:
                emotion, score = emotions
                if emotion == "happy":
                    happy_count += 1
                else:
                    not_happy_count += 1
                labels.append(f"{emotion} {score:.2f}")
            else:
                labels.append("Unknown")

            # Update the total person count
        total_count = happy_count + not_happy_count

        
        detections = sv.Detections(xyxy= boxes,
        confidence= confidences[person_indices],
        class_id= class_ids[person_indices]
        )
        
        # Add labels to detections
        detections.labels = labels

        

        annotated_image= box_annotator.annotate(scene=frame, detections=detections)

        # display the count on the frame
        overlay_text = f"Total: {total_count} | Happy: {happy_count} | Not Happy: {not_happy_count}"
        cv2.putText(
            annotated_image,
            overlay_text,
            (10, 30),  # Position of the text
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # Font scale
            (0, 255, 0),  # Text color
            2,  # Thickness
        )
        

        cv2.imshow("YOLOv8 Detection", annotated_image)

        

        if (cv2.waitKey(30) == 27):
            break


if __name__ == '__main__':
    main()
