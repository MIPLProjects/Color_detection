import cv2
from ultralytics import YOLO
import os

# === Load YOLOv11 segmentation model ===
model = YOLO(r"train\weights\best.pt")

# === Ask user for input ===
input_path = input("🔍 Enter path to input image or video: ").strip()

# === Check if input file exists ===
if not os.path.isfile(input_path):
    print(f"❌ File not found: {input_path}")
    exit()

# === Determine file type ===
ext = os.path.splitext(input_path)[-1].lower()
is_image = ext in [".jpg", ".jpeg", ".png", ".bmp"]

# === Set output path ===
output_path = "output_detected.jpg" if is_image else "output_detected.avi"

# === For Images ===
if is_image:
    image = cv2.imread(input_path)
    results = model.predict(source=image, conf=0.25, save=False, stream=True, retina_masks=True)

    for r in results:
        annotated = r.plot(conf=False, boxes=False, labels=True)

    cv2.imshow("Detected Blue Dots", annotated)
    cv2.imwrite(output_path, annotated)
    print(f"✅ Saved annotated image to {output_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === For Videos ===
else:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Couldn't open video: {input_path}")
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.25, save=False, stream=True, retina_masks=True)

        for r in results:
            if r.masks is not None:
                annotated_frame = r.plot(conf=False, boxes=False, labels=True)
            else:
                annotated_frame = frame.copy()

            out.write(annotated_frame)
            cv2.imshow("Detected Blue Dots", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Saved annotated video to {output_path}")
