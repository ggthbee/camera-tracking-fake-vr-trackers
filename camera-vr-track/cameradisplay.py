import cv2
import numpy as np

def configure_camera(cap, width=640, height=480, fps=60, brightness=-10):
    """Configure camera to desired resolution, FPS, and brightness."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    # Verify settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    return actual_width, actual_height, actual_fps, actual_brightness

def main():
    # Define the camera indices to display
    camera_indices = [0, 1, 2, 3]
    captures = []
    brightness_value = -10  # Shared brightness value for all cameras

    # Open and configure each camera
    for index in camera_indices:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows
        if cap.isOpened():
            width, height, fps, brightness = configure_camera(cap, brightness=brightness_value)
            print(f"Camera {index}: {width}x{height} @ {fps} FPS, Brightness: {brightness}")
            captures.append((index, cap))
        else:
            print(f"Camera {index} not available")

    if not captures:
        print("No specified cameras are available.")
        return

    print(f"Displaying {len(captures)} camera(s): {[idx for idx, _ in captures]}")
    print("Press '+' to increase brightness, '-' to decrease brightness, 'q' to exit")

    try:
        while True:
            for index, cap in captures:
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to grab frame from camera {index}")
                    continue

                # Display the frame
                window_name = f"Camera {index}"
                cv2.imshow(window_name, frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):
                brightness_value += 10  # Increase brightness for all cameras
                for index, cap in captures:
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)
                print(f"All cameras brightness set to {brightness_value}")
            elif key == ord('-'):
                brightness_value -= 10  # Decrease brightness for all cameras
                for index, cap in captures:
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)
                print(f"All cameras brightness set to {brightness_value}")

    finally:
        # Release all cameras and close windows
        for _, cap in captures:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()