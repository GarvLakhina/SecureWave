from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
try:
    model = load_model('models/gender_detection_model_final.keras')  # Updated model path and name
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define the classes
classes = ['man', 'woman']

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Loop to process video frames
while webcam.isOpened():
    # Capture frame-by-frame
    status, frame = webcam.read()

    if not status:
        break

    # Detect faces in the frame
    faces, confidences = cv.detect_face(frame)

    # Loop through each detected face
    for i, f in enumerate(faces):
        # Get the coordinates of the face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw rectangle around face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the face from the frame
        face_crop = np.copy(frame[startY:endY, startX:endX])

        # Skip small faces
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocess the face for prediction
        face_crop = cv2.resize(face_crop, (224, 224))  # Resize to 224x224 (model's expected input size)
        face_crop = face_crop.astype("float") / 255.0  # Normalize the image
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)  # Add batch dimension

        # Predict gender
        conf = model.predict(face_crop)[0][0]  # Access the scalar value directly

        # Determine the label based on the prediction (threshold of 0.5)
        idx = 1 if conf >= 0.5 else 0  # If prediction is >= 0.5, it's a woman; else, man
        label = classes[idx]

        # Display label and confidence
        label_text = "{}: {:.2f}%".format(label, conf * 100)  # Confidence as a percentage
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, label_text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Gender Detection", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
