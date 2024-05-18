# Real-time Face Recognition with DeepFace

## Overview

This project utilizes computer vision techniques to perform real-time face recognition using the DeepFace library in Python. The system captures live video feed from a webcam, detects faces, and predicts the gender and emotion of each detected face.

## Features

- Detects faces in real-time using the Haar Cascade classifier.
- Predicts gender and emotion using the DeepFace library.
- Draws bounding boxes around detected faces with gender and emotion labels.
- Provides an intuitive user interface for real-time face analysis.

## Libraries Used

- **OpenCV**: Used for capturing live video feed and performing face detection.
- **DeepFace**: Utilized for gender and emotion prediction from detected faces.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/face-recognition.git
   ```
2. Install the required libraries:
   ```pip install opencv-python-headless
      pip install deepface
   ```
3. Run the program:

   ```
   python gender.py

   ```
