import cv2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_camera(index, backend=cv2.CAP_ANY, backend_name="CAP_ANY", fourcc=None):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        logging.error(f"Camera {index} ({backend_name}): Not available")
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_val = cap.get(cv2.CAP_PROP_FOURCC)
    fourcc_str = int(fourcc_val).to_bytes(4, 'little').decode('utf-8', errors='ignore')
    logging.info(f"Camera {index} ({backend_name}): {width}x{height} @ {fps} FPS, FOURCC={fourcc_str}")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            logging.error(f"Camera {index} ({backend_name}): Failed to read frame")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow(f"Camera {index} ({backend_name})", gray)
        frame_count += 1
        if frame_count % 30 == 0:
            logging.info(f"Camera {index} ({backend_name}): Successfully read {frame_count} frames")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return True

# Test camera 3 with multiple backends and formats
backends = [(cv2.CAP_MSMF, "CAP_MSMF"), (cv2.CAP_DSHOW, "CAP_DSHOW"), (cv2.CAP_ANY, "CAP_ANY")]
formats = ['MJPG', 'YUY2', None]  # Test MJPEG, YUY2, and default
index = 3  # Adjust if camera 3’s index differs
for backend, name in backends:
    for fmt in formats:
        logging.info(f"Testing Camera {index} with {name}, FOURCC={fmt if fmt else 'default'}")
        if test_camera(index, backend, name, fmt):
            break
    else:
        continue
    break
else:
    logging.error(f"Camera {index}: Failed with all backends and formats")