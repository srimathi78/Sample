# import dlib
import cv2
import os
import numpy as np
# from tensorflow.keras.models import load_model

# Load pre-trained facial landmark detector
detector = cv2.get_frontal_face_detector()
predictor = cv2.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load pre-trained expression classification model
model = os("expression_model.h5")

# Load and preprocess the image
img_path = 'Sp_happy.jpg'
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

# Iterate over detected faces
for face in faces:
    # Get facial landmarks
    landmarks = predictor(gray, face)
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

    # Extract features (e.g., distances between landmarks)
    # Example: Calculate the distance between the eyes
    eye_dist = np.linalg.norm(landmarks[36] - landmarks[45])
    
    # Perform expression classification using extracted features
    # Example: Create feature vector and normalize it
    features = np.array([eye_dist])
    features = (features - features.mean()) / features.std()
    features = np.expand_dims(features, axis=0)

    # Predict expression
    prediction = model.predict(features)
    expression = "Happy" if prediction > 0.5 else "Not Happy"

    # Draw bounding box and expression label on the image
    cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
    cv2.putText(img, expression, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the annotated image
cv2.imshow('Face emotion Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
