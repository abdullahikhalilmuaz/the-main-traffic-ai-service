from pathlib import Path

from PIL import Image, ImageOps
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "images" / "traffic.jpg"
OUTPUT_PATH = BASE_DIR / "detected_traffic.jpg"


MODEL_NAME = "yolov8s.pt"

IMAGE_SIZE = 1280
CONFIDENCE = 0.18
IOU = 0.50


VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


print("Loading model...")

model = YOLO(MODEL_NAME)

print("Model loaded.")
print(f"Model: {MODEL_NAME}")


# ------------------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------------------

image = Image.open(IMAGE_PATH)

image = ImageOps.exif_transpose(image)

if image.mode != "RGB":
    image = image.convert("RGB")


# ------------------------------------------------------------
# DETECTION
# ------------------------------------------------------------

print("Running detection...")

results = model.predict(
    source=image,
    imgsz=IMAGE_SIZE,
    conf=CONFIDENCE,
    iou=IOU,
    classes=list(VEHICLE_CLASSES.keys()),
    augment=True,
    verbose=False,
)


result = results[0]


# ------------------------------------------------------------
# COUNT VEHICLES
# ------------------------------------------------------------

counts = {
    "car": 0,
    "motorcycle": 0,
    "bus": 0,
    "truck": 0,
}


for box in result.boxes:

    class_id = int(box.cls[0].item())

    if class_id in VEHICLE_CLASSES:

        vehicle = VEHICLE_CLASSES[class_id]

        counts[vehicle] += 1


total = sum(counts.values())


# ------------------------------------------------------------
# SAVE RESULT
# ------------------------------------------------------------

annotated = result.plot(
    conf=True,
    labels=True,
    boxes=True,
    line_width=2,
)


Image.fromarray(annotated).save(
    OUTPUT_PATH,
    quality=95,
)


# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

print("")
print("=" * 50)
print("TRAFFIC DETECTION RESULT")
print("=" * 50)

print(f"Cars:         {counts['car']}")
print(f"Motorcycles:  {counts['motorcycle']}")
print(f"Buses:        {counts['bus']}")
print(f"Trucks:       {counts['truck']}")
print(f"Total:        {total}")

print("=" * 50)

print(f"Result saved to:")
print(OUTPUT_PATH)