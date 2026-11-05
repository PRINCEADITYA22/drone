
import cv2
import cv2.aruco as aruco
import numpy as np
import math

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam!")
    exit()

MARKER_SIZE = 0.06

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

ret, frame = cap.read()

if not ret:
    print("Camera error!")
    cap.release()
    exit()

h, w = frame.shape[:2]

focal = w / (2 * math.tan(math.radians(60) / 2))

camera_matrix = np.array([
    [focal, 0, w / 2],
    [0, focal, h / 2],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.zeros((5, 1), dtype=np.float64)

s = MARKER_SIZE / 2

obj_points = np.array([
    [-s, s, 0],
    [s, s, 0],
    [s, -s, 0],
    [-s, -s, 0]
], dtype=np.float64)

print("ARUCO 3D POSE ESTIMATION STARTED")
print("Press Q to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:

        for i, marker_id in enumerate(ids.flatten()):

            img_points = corners[i][0].astype(np.float64)

            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if success:

                x, y, z = tvec.flatten()

                distance = np.linalg.norm(tvec) * 100

                R, _ = cv2.Rodrigues(rvec)

                angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)

                roll, pitch, yaw = angles

                points = corners[i][0].astype(int)

                cv2.polylines(
                    frame,
                    [points],
                    True,
                    (0, 255, 0),
                    3
                )

                center = np.mean(points, axis=0).astype(int)

                cv2.circle(
                    frame,
                    tuple(center),
                    5,
                    (0, 0, 255),
                    -1
                )

                cv2.drawFrameAxes(
                    frame,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    0.04,
                    3
                )

                info = [
                    f"Marker ID: {marker_id}",
                    f"Distance: {distance:.2f} cm",
                    f"X: {x:.3f} m",
                    f"Y: {y:.3f} m",
                    f"Z: {z:.3f} m",
                    f"Roll: {roll:.2f} deg",
                    f"Pitch: {pitch:.2f} deg",
                    f"Yaw: {yaw:.2f} deg"
                ]

                cv2.rectangle(
                    frame,
                    (10, 10),
                    (350, 280),
                    (0, 0, 0),
                    -1
                )

                for j, text in enumerate(info):

                    cv2.putText(
                        frame,
                        text,
                        (20, 40 + j * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

    else:

        cv2.putText(
            frame,
            "NO ARUCO MARKER DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("ARUCO 3D POSE ESTIMATION", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()