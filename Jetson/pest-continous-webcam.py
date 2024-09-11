import cv2
import requests
import time

# Replace with your API URL
api_url = 'http://localhost:1337/api/image'

# Open the default camera (usually index 0)
cap = cv2.VideoCapture(0)

# Set the resolution to 640x640
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define a list of distinct colors for up to 4 categories (in BGR format for OpenCV)
category_colors = [
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
]

# Dictionary to map label names to colors
label_to_color = {}

def get_label_color(label):
    # Assign colors dynamically to each label
    if label not in label_to_color:
        if len(label_to_color) < len(category_colors):
            label_to_color[label] = category_colors[len(label_to_color)]
        else:
            # If more than 4 categories are detected, fallback to a default color
            label_to_color[label] = (255, 255, 255)  # White
    return label_to_color[label]

def draw_bounding_boxes(frame, detections):
    """
    Draw bounding boxes on the frame based on the detection results.
    detections: expected to be a list of dicts, where each dict contains:
                - 'x': top-left x-coordinate
                - 'y': top-left y-coordinate
                - 'width': width of the bounding box
                - 'height': height of the bounding box
                - 'label': class label
                - 'value': confidence score
    """
    for detection in detections:
        x = detection['x']
        y = detection['y']
        width = detection['width']
        height = detection['height']
        label = detection['label']
        confidence = detection['value']

        # Get the color assigned to this label
        color = get_label_color(label)

        # Draw rectangle (bounding box) with the color
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

        # Prepare the label with confidence
        label_text = f"{label} ({confidence:.2f})"

        # Put the label above the bounding box
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame is captured successfully
    if ret:
        # Save the frame temporarily as an image file
        frame_filename = 'frame.jpg'
        cv2.imwrite(frame_filename, frame)

        # Send the image file to the API via POST request
        with open(frame_filename, 'rb') as image_file:
            files = {'file': image_file}
            response = requests.post(api_url, files=files)

        # Parse the API response
        if response.status_code == 200:
            try:
                # Extract bounding boxes from the response
                result = response.json().get("result", {})
                bounding_boxes = result.get("bounding_boxes", [])
                
                # Draw bounding boxes on the frame if there are any detections
                draw_bounding_boxes(frame, bounding_boxes)

            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Error: API request failed with status code {response.status_code}")

        # Display the frame with bounding boxes
        cv2.imshow('Webcam Frame with Detections', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Optional: Add a small delay if you don’t want to overwhelm the API
        time.sleep(0.1)  # 100ms delay between frames

    else:
        print("Error: Could not read frame.")
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
