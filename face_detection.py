import cv2

# Load the pre-trained Haar Cascade classifier for face detection
# Ensure that 'haarcascade_frontalface_default.xml' is in the working directory
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the video capture object to read from the default camera (usually the first connected camera)
video_capture = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Loop to continuously get frames from the camera feed
while True:
    # Capture frame-by-frame from the video feed
    ret, frame = video_capture.read()
    
    # Check if the frame was captured correctly
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale (face detection is usually done on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame with rectangles around faces
    cv2.imshow('Video', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
