import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO('yolov8n-seg.pt')

# Open the video file
video_path = 'output_cropped.avi'
#cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
output_path = 'output_video_with_detections.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Extract the first result
    result = results[0]

    # Check if there are any masks detected
    if result.masks is not None and result.masks.xy:
        try:
            # Get the mask of the segmented object as a list of segments in pixel coordinates
            segments = result.masks.xy[0]

            # Create a blank mask
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            # Fill the mask with the segments
            cv2.fillPoly(mask, [np.array(segments, dtype=np.int32)], 255)

            # Calculate the bounding box of the mask
            x, y, w, h = cv2.boundingRect(mask)

            # Calculate the length of the object (the longest dimension of the bounding box)
            length = max(w, h)

            # Convert the length from pixels to a real-world unit of measurement (e.g., cm)
            scale_factor = 0.1  # 1 pixel = 0.1 cm
            real_length = length * scale_factor

            # Draw the bounding box on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Add the length as a label on the image
            label = f'Length: {real_length:.2f} cm'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            top_left_corner = (x, y + label_size[1] + base_line)
            cv2.putText(frame, label, top_left_corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        except Exception as e:
            print(f"Error processing frame: {e}")

    # Write the frame to the output video
    out.write(frame)

    # Display the frame with the detections
    cv2.imshow('YOLOv8 Detections', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
