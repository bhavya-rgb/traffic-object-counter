import cv2
from ultralytics import YOLO

# ==============================
# 1. LOAD YOLO MODEL
# ==============================
model = YOLO("yolo11s.pt")

# ==============================
# 2. VIDEO PATHS
# ==============================
input_video = "input/traffic.mp4"
output_video = "output/result.mp4"

# ==============================
# 3. OPEN VIDEO
# ==============================
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video Width :", width)
print("Video Height:", height)
print("FPS         :", fps)

# ==============================
# 4. OUTPUT VIDEO
# ==============================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

# ==============================
# 5. VEHICLE CLASSES
# ==============================
# COCO classes:
# 2 = Car
# 3 = Motorcycle
# 5 = Bus
#
# Truck = 7
# We are NOT including truck.

vehicle_classes = {
    2: "Car",
    3: "Bike",
    5: "Bus"
}

# ==============================
# 6. COUNTING LINE
# ==============================
# Change this value later if needed.
line_y = int(height * 0.55)

# ==============================
# 7. TRACKING DATA
# ==============================
previous_positions = {}

counted_ids = {
    "Car": set(),
    "Bike": set(),
    "Bus": set()
}

# ==============================
# 8. FRAME LOOP
# ==============================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ==============================
    # YOLO TRACKING
    # ==============================
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5],
        conf=0.35,
        verbose=False
    )

    result = results[0]

    # ==============================
    # PROCESS DETECTIONS
    # ==============================
    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes

        for i in range(len(boxes)):

            # Class ID
            class_id = int(boxes.cls[i])

            if class_id not in vehicle_classes:
                continue

            vehicle_type = vehicle_classes[class_id]

            # Tracking ID
            track_id = int(boxes.id[i])

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                boxes.xyxy[i]
            )

            # Confidence
            confidence = float(boxes.conf[i])

            # ==============================
            # CENTER POINT
            # ==============================
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # ==============================
            # LINE CROSSING
            # ==============================

            if track_id in previous_positions:

                previous_y = previous_positions[track_id]

                # Vehicle moving downward
                crossed_down = (
                    previous_y < line_y
                    and center_y >= line_y
                )

                # Vehicle moving upward
                crossed_up = (
                    previous_y > line_y
                    and center_y <= line_y
                )

                if crossed_down or crossed_up:

                    # Count only once
                    if track_id not in counted_ids[vehicle_type]:

                        counted_ids[vehicle_type].add(track_id)

            # Save current position
            previous_positions[track_id] = center_y

            # ==============================
            # DRAW BOX
            # ==============================
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ==============================
            # DRAW CENTER
            # ==============================
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # ==============================
            # LABEL
            # ==============================
            label = (
                f"{vehicle_type} "
                f"ID:{track_id} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    # ==============================
    # DRAW COUNTING LINE
    # ==============================

    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        "COUNTING LINE",
        (20, line_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # ==============================
    # COUNTS
    # ==============================

    car_count = len(counted_ids["Car"])
    bike_count = len(counted_ids["Bike"])
    bus_count = len(counted_ids["Bus"])

    total_count = (
        car_count +
        bike_count +
        bus_count
    )

    # ==============================
    # COUNT PANEL
    # ==============================

    cv2.rectangle(
        frame,
        (10, 10),
        (280, 145),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Cars  : {car_count}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Bikes : {bike_count}",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses : {bus_count}",
        (25, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ==============================
    # WRITE FRAME
    # ==============================
    out.write(frame)

# ==============================
# 9. RELEASE
# ==============================
cap.release()
out.release()

# ==============================
# 10. FINAL RESULT
# ==============================
print()
print("================================")
print("PROCESSING COMPLETED")
print("================================")

print("Cars  :", len(counted_ids["Car"]))
print("Bikes :", len(counted_ids["Bike"]))
print("Buses :", len(counted_ids["Bus"]))

print()
print("Output saved to:", output_video)