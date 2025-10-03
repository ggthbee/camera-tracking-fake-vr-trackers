import cv2
import numpy as np
import pickle

# Chessboard pattern settings
CHECKERBOARD = (6, 8)  # (rows, cols)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints1, imgpoints2 = [], []

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

while len(objpoints) < 30:  # Capture 30 frames
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not (ret1 and ret2):
        continue
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, None)
    if ret1 and ret2:
        objpoints.append(objp)
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)
    cv2.imshow("Calibration", np.hstack([frame1, frame2]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calibrate cameras
ret1, camera_matrix1, dist_coeffs1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, gray1.shape[::-1], None, None)
ret2, camera_matrix2, dist_coeffs2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1], None, None)
ret, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, gray1.shape[::-1])
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, gray1.shape[::-1], R, T)

# Save calibration
calib = {
    'camera_matrix1': camera_matrix1, 'dist_coeffs1': dist_coeffs1,
    'camera_matrix2': camera_matrix2, 'dist_coeffs2': dist_coeffs2,
    'R': R, 'T': T, 'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2
}
with open('stereo_calib.pkl', 'wb') as f:
    pickle.dump(calib, f)

cap1.release()
cap2.release()
cv2.destroyAllWindows()