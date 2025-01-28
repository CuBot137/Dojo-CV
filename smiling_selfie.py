import cv2
import mediapipe as mp
import pyautogui
import winsound 

x1 = 0 
y1 = 0
x2 = 0
y2 = 0
# Mediapipe is an object detection library with some LLM elements
# Face mesh detects landmarks on a persons face. Refine landmarks detects subtle differences like a smile
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) # Capture face landmarks, e.g. Smiles 
camera = cv2.VideoCapture(0)
while True:
    _, image = camera.read() # Get a single image 
    image = cv2.flip(image, 1) # FLip it around
    fh, fw, _ = image.shape # Get the image height and width
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert the image to RedGreenBlue format
    output = face_mesh.process(rgb_image) # Detect face landmarks
    landmark_points = output.multi_face_landmarks # Store them
    if landmark_points: # If there is a face on the screen
        landmarks = landmark_points[0].landmark
        # Loop over each facial landmark in a single frame
        for id, landmark in enumerate(landmarks):
            # Get a whole number 
            x = int(landmark.x * fw)
            y = int(landmark.y * fh)
            # Display all landmarks and their IDs
            cv2.putText(image, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            if id == 57:
                x1 = x
                y1 = y
            if id == 273:
                x2 = x
                y2 = y
        # Find the distance between two points of a straight line 
        dist = int(((x2-x1)**2 + (y2-y1)**2)**(0.5))
        if dist > 100:
            cv2.imwrite("selfie.png", image)
            winsound.PlaySound("mixkit-long-pop-2358.wav", winsound.SND_FILENAME)
            cv2.waitKey(100)
        cv2.imshow("Selfie", image)
        print(dist)
        
    cv2.imshow("Auto selfy for smiling faces using python", image)
    key = cv2.waitKey(100)
    if key == 27: # This represents the escape key
        break
    
camera.release()
cv2.destroyAllWindows()