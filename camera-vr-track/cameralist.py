import cv2

def test_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Camera {index}: Not available")
        return False
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera {index}: {width}x{height}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {index}: Failed to read frame")
            break
        cv2.imshow(f"Camera {index}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return True

for i in range(4):
    print(f"Testing Camera {i}")
    test_camera(i)