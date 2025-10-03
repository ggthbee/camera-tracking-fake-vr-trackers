import cv2  # This is a tool for working with video and images

# Start the camera (0 is usually your webcam; try 1 or 2 if it doesn’t work)
camera = cv2.VideoCapture(0)

# Function to make the IR image clearer
def clean_up_image(frame):
    # If the image is in color, make it grayscale (IR cameras often give grayscale)
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Boost contrast to make body parts stand out
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frame = clahe.apply(frame)
    
    # Smooth out noise (like static on old TVs)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Make brightness consistent
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    return frame

# Keep showing the camera feed until you press 'q'
while camera.isOpened():
    ret, frame = camera.read()  # Grab one frame (picture) from the camera
    if not ret:  # If no frame, stop
        break
    
    # Clean up the frame
    clean_frame = clean_up_image(frame)
    
    # Show the cleaned-up image on your screen
    cv2.imshow("Camera Feed", clean_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()