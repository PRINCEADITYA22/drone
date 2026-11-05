import cv2
import cv2.aruco as aruco
import numpy as np
import math
import threading

MARKERS = {
    "prince":    {"id": 23, "size": 0.06},
    "aryan":     {"id": 24, "size": 0.06},
    "somyashree":    {"id": 25, "size": 0.06},
    "shubhasmita": {"id": 26, "size": 0.06},
}

CAMERA_INDEX = 0
ASSUMED_HFOV_DEG = 60

ID_TO_NAME = {cfg["id"]: name for name, cfg in MARKERS.items()}

state = {"target": None, "quit": False}
lock = threading.Lock()


def print_help():
    print("\nCommands (type here in the terminal, then press Enter):")
    print("  <name>  -> search ONLY for that person's marker, e.g. 'prince'")
    print("  stop    -> back to scan mode (names only, no distance/position)")
    print("  list    -> show available names")
    print("  quit    -> exit")
    print("Available names:", ", ".join(MARKERS.keys()), "\n")


def command_listener():
    print_help()
    while True:
        try:
            text = input("> ").strip().lower()
        except EOFError:
            return

        if not text:
            continue

        if text in ("quit", "exit", "q"):
            with lock:
                state["quit"] = True
            return
        elif text == "stop":
            with lock:
                state["target"] = None
            print("Search stopped. Scan mode: names only.")
        elif text == "list":
            print("Available names:", ", ".join(MARKERS.keys()))
        elif text in MARKERS:
            with lock:
                state["target"] = text
            print(f"Searching ONLY for '{text}'. Nothing else will be reported "
                  f"until that marker appears.")
        else:
            print(f"Unknown name '{text}'. Available:", ", ".join(MARKERS.keys()))


threading.Thread(target=command_listener, daemon=True).start()

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Cannot open webcam!")
    exit()

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
detector = aruco.ArucoDetector(dictionary, parameters)

ret, frame = cap.read()
if not ret:
    print("Camera error!")
    cap.release()
    exit()

h, w = frame.shape[:2]

focal = w / (2 * math.tan(math.radians(ASSUMED_HFOV_DEG) / 2))

camera_matrix = np.array([
    [focal, 0, w / 2],
    [0, focal, h / 2],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.zeros((5, 1), dtype=np.float64)

FLIP = np.diag([1, -1, -1]).astype(np.float64)

WINDOW_NAME = "ARUCO NAME SEARCH"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def make_obj_points(size):
    s = size / 2
    return np.array([
        [-s,  s, 0],
        [ s,  s, 0],
        [ s, -s, 0],
        [-s, -s, 0]
    ], dtype=np.float64)


announced = set()

print("ARUCO NAME SEARCH STARTED (press Q in the video window to exit)")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    with lock:
        target = state["target"]
        should_quit = state["quit"]

    if should_quit:
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    visible_now = set()
    found_target = False

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):

            marker_id = int(marker_id)

            if marker_id not in ID_TO_NAME:
                continue

            name = ID_TO_NAME[marker_id]
            points = corners[i][0].astype(int)

            if target is None:
                visible_now.add(name)

                cv2.polylines(frame, [points], True, (0, 255, 0), 3)
                cv2.putText(
                    frame, name.capitalize(),
                    (points[0][0], max(points[0][1] - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2
                )

                if name not in announced:
                    print(f"Detected: {name.capitalize()}")
                    announced.add(name)
                continue

            if name != target:
                continue

            size = MARKERS[name]["size"]
            success, rvec, tvec = cv2.solvePnP(
                make_obj_points(size),
                corners[i][0].astype(np.float64),
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if not success:
                continue

            found_target = True
            visible_now.add(name)

            x, y, z = tvec.flatten()
            distance = np.linalg.norm(tvec) * 100

            R, _ = cv2.Rodrigues(rvec)
            pitch, yaw, roll = cv2.RQDecomp3x3(R @ FLIP)[0]

            cv2.polylines(frame, [points], True, (0, 255, 0), 3)

            center = tuple(int(v) for v in np.mean(points, axis=0))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            cv2.drawFrameAxes(
                frame, camera_matrix, dist_coeffs, rvec, tvec, size * 0.7, 3
            )

            cv2.putText(
                frame, name.capitalize(),
                (points[0][0], max(points[0][1] - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2
            )

            info = [
                f"Name: {name.capitalize()}",
                f"Marker ID: {marker_id}",
                f"Distance: {distance:.2f} cm",
                f"X: {x:.3f} m",
                f"Y: {y:.3f} m",
                f"Z: {z:.3f} m",
                f"Roll: {roll:.2f} deg",
                f"Pitch: {pitch:.2f} deg",
                f"Yaw: {yaw:.2f} deg"
            ]

            cv2.rectangle(frame, (10, 10), (400, 300), (0, 0, 0), -1)
            for j, text in enumerate(info):
                cv2.putText(
                    frame, text, (20, 40 + j * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
                )

            if name not in announced:
                print(f"Found {name.capitalize()}!")
                announced.add(name)

    announced &= visible_now

    if target is None:
        if not visible_now:
            cv2.putText(
                frame, "SCAN MODE - type a name in the terminal to search", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
    elif not found_target:
        cv2.putText(
            frame, f"SEARCHING FOR '{target.upper()}'...", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()