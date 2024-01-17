import cv2
import dlib
import time

# Load pre-trained models
face_detector = dlib.cnn_face_detection_model_v1('./Models/mmod_human_face_detector.dat')
shape_predictor = dlib.shape_predictor('./Models/shape_predictor_68_face_landmarks.dat')

# Initialize video capture
cap = cv2.VideoCapture(1)

# Frame processing parameters
detection_interval = 10  # Reduced to 1 for face detection on every frame
new_width, new_height = 200, 200  # Adjust as needed

# Initialize frame count and timer
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    frame_count += 1

    # Resize frame
    frame = cv2.resize(frame, (new_width, new_height))

    if frame_count % detection_interval == 0:
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_detector(gray)

        for face in faces:
            # Get facial landmarks
            shape = shape_predictor(gray, face.rect)
            for i in range(68):
                x, y = shape.part(i).x, shape.part(i).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Draw rectangle around the face
            x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate and print the frames per second (FPS)
end_time = time.time()
fps = frame_count / (end_time - start_time)
print(f"Average FPS: {fps}")

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
