import cv2
from deepface import DeepFace
import random

# Initialize the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Access the local webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        # Calculate offsets to include more context around the face
        offset = 40
        x_offset = max(x - offset, 0)
        y_offset = max(y - offset, 0)
        w_offset = min(w + 2 * offset, frame.shape[1] - x_offset)
        h_offset = min(h + 2 * offset, frame.shape[0] - y_offset)

        # Extract the face region with offsets from the frame
        face = frame[y_offset:y_offset+h_offset, x_offset:x_offset+w_offset]

        # Predict gender and emotion using DeepFace
        result = DeepFace.analyze(face, actions=['gender', 'emotion'], enforce_detection=False)
        print(result)

        # Extract the dominant gender and emotion from the result
        dominant_gender = result[0]['dominant_gender']
        dominant_emotion = result[0]['dominant_emotion']

        # Generate a random color for the rectangle
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Draw a rectangle around the face with offset
        cv2.rectangle(frame, (x_offset, y_offset), (x_offset + w_offset, y_offset + h_offset), color, 2)

        # Display the dominant gender and emotion label above the rectangle
        label = f"{dominant_gender}, {dominant_emotion}"
        cv2.putText(frame, label, (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
