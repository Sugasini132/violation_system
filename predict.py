from ultralytics import YOLO
import cv2
import os
import math

# Load trained model
model = YOLO("runs/detect/train-4/weights/best.pt")

# Input image
image_path = "test.jpeg"

# Run detection
results = model(image_path)

# Read image
img = cv2.imread(image_path)

# Store detections
no_helmet_boxes = []
plate_boxes = []

for r in results:

    for box in r.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cls = int(box.cls[0])

        conf = float(box.conf[0])

        label = model.names[cls]

        # Colors
        if label == "rider":
            color = (0, 255, 255)  # Yellow

        elif label == "with helmet":
            color = (0, 255, 0)    # Green

        elif label == "without helmet":
            color = (0, 0, 255)    # Red

            no_helmet_boxes.append(
                (x1, y1, x2, y2)
            )

        elif label == "number plate":
            color = (255, 0, 0)    # Blue

            plate_boxes.append(
                (x1, y1, x2, y2)
            )

        else:
            color = (255, 255, 255)

        # Draw box
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Draw label
        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

# Save detection image
cv2.imwrite("output.jpeg", img)

# Create violations folder
os.makedirs("violations", exist_ok=True)

case_number = 1

# Process every no helmet rider
for rider_box in no_helmet_boxes:

    rx1, ry1, rx2, ry2 = rider_box

    rider_crop = img[ry1:ry2, rx1:rx2]

    rider_center_x = (rx1 + rx2) / 2
    rider_center_y = (ry1 + ry2) / 2

    nearest_plate = None
    min_distance = float("inf")

    # Find nearest plate
    for plate_box in plate_boxes:

        px1, py1, px2, py2 = plate_box

        plate_center_x = (px1 + px2) / 2
        plate_center_y = (py1 + py2) / 2

        distance = math.sqrt(
            (rider_center_x - plate_center_x) ** 2 +
            (rider_center_y - plate_center_y) ** 2
        )

        if distance < min_distance:

            min_distance = distance
            nearest_plate = plate_box

    # Create case folder
    case_folder = f"violations/case_{case_number}"

    os.makedirs(case_folder, exist_ok=True)

    # Save rider
    cv2.imwrite(
        f"{case_folder}/rider.jpg",
        rider_crop
    )

    # Save plate
    if nearest_plate:

        px1, py1, px2, py2 = nearest_plate

        plate_crop = img[py1:py2, px1:px2]

        cv2.imwrite(
            f"{case_folder}/plate.jpg",
            plate_crop
        )

    print(f"Case {case_number} saved")

    case_number += 1

print("\nDetection Completed")
print("Output Image Saved: output.jpeg")
print("Violations Saved Successfully")

os.startfile("output.jpeg")