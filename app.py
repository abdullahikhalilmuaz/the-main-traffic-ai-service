import os
import uuid
import threading
import torch
import cv2
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps

from ultralytics import YOLO


# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "https://traffic-ai-ui.vercel.app",
            ]
        }
    },
)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300MB


# ============================================================
# YOLO MODEL
# ============================================================

print("Loading YOLO model...")

MODEL_NAME = "yolov8s.pt"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

model = YOLO(MODEL_NAME)

model.fuse()

print(f"YOLO model loaded successfully: {MODEL_NAME}")


# ============================================================
# VEHICLE CLASSES
# COCO dataset class IDs
# ============================================================

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ============================================================
# DETECTION SETTINGS (IMAGES)
# ============================================================

IMAGE_SIZE = 1536

# Lower confidence helps recover smaller/distant vehicles.
CONFIDENCE = 0.25

# IoU used for Non-Maximum Suppression.
IOU = 0.50


# ============================================================
# DETECTION SETTINGS (VIDEO)
# Video runs at a lower resolution than single images — YOLO
# cost scales roughly with pixel count squared, so dropping
# from 1536 to 640 is a big speedup and still plenty accurate
# for vehicle-sized objects in typical traffic footage.
# ============================================================

VIDEO_IMAGE_SIZE = 640

# Running YOLO on every single frame is expensive — process every
# Nth frame and reuse the last annotated frame for the ones in between.
# Raise this for more speed at the cost of temporal resolution.
VIDEO_FRAME_SKIP = 3

# Hard cap so a long/huge upload can't hang the worker forever.
VIDEO_MAX_FRAMES = 900

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}


def allowed_video(filename):
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_VIDEO_EXTENSIONS


# ============================================================
# VIDEO JOB STORE
# In-memory job tracking for background video processing.
# job_id -> {status, progress, result, error}
# status: "queued" | "processing" | "done" | "error"
# Fine for a single-process dev server; swap for Redis/DB if
# you ever run multiple Flask workers.
# ============================================================

video_jobs = {}


# ============================================================
# TRAFFIC CLASSIFICATION
# ============================================================

def classify_traffic(total):
    if total < 20:
        return "LOW"

    if total < 50:
        return "MODERATE"

    return "HIGH"


# ============================================================
# ALLOWED IMAGE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
}


def allowed_image(filename):
    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# NORMALIZE IMAGE
# ============================================================

def normalize_image(input_path, output_path):
    """
    Opens uploaded image,
    fixes EXIF rotation,
    converts to RGB,
    saves clean JPEG.
    """

    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        image.save(
            output_path,
            format="JPEG",
            quality=95,
        )


# ============================================================
# FORMAT DETECTION RESULTS
# ============================================================

def process_detections(result):
    counts = {
        "cars": 0,
        "motorcycles": 0,
        "buses": 0,
        "trucks": 0,
    }

    detections = []

    if result.boxes is None:
        return counts, detections

    for box in result.boxes:

        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        if class_id not in VEHICLE_CLASSES:
            continue

        vehicle_type = VEHICLE_CLASSES[class_id]

        if vehicle_type == "car":
            counts["cars"] += 1

        elif vehicle_type == "motorcycle":
            counts["motorcycles"] += 1

        elif vehicle_type == "bus":
            counts["buses"] += 1

        elif vehicle_type == "truck":
            counts["trucks"] += 1

        coordinates = box.xyxy[0].tolist()

        detections.append(
            {
                "class": vehicle_type,
                "confidence": round(confidence, 4),
                "box": {
                    "x1": round(coordinates[0], 2),
                    "y1": round(coordinates[1], 2),
                    "x2": round(coordinates[2], 2),
                    "y2": round(coordinates[3], 2),
                },
            }
        )

    return counts, detections


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "success": True,
            "status": "online",
            "service": "Smart Traffic AI Service",
            "model": MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "confidence": CONFIDENCE,
        }
    )


# ============================================================
# IMAGE DETECTION
# ============================================================

@app.route("/detect", methods=["POST"])
def detect_image():

    original_path = None
    clean_path = None

    try:

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file was uploaded."
            }), 400

        image_file = request.files["image"]

        if image_file.filename == "":
            return jsonify({
                "success": False,
                "error": "No image was selected."
            }), 400

        if not allowed_image(image_file.filename):
            return jsonify({
                "success": False,
                "error": "Unsupported image format. Use JPG, JPEG, PNG or WEBP."
            }), 400

        original_name = secure_filename(image_file.filename)

        unique_id = uuid.uuid4().hex

        original_path = UPLOAD_FOLDER / f"{unique_id}_{original_name}"
        clean_path = UPLOAD_FOLDER / f"{unique_id}_clean.jpg"

        output_name = f"{unique_id}_detected.jpg"
        output_path = OUTPUT_FOLDER / output_name

        image_file.save(original_path)

        normalize_image(
            original_path,
            clean_path
        )

        print(f"Analyzing image: {original_name}")

        results = model.predict(
            source=str(clean_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=IOU,
            classes=list(VEHICLE_CLASSES.keys()),
            augment=False,
            device=DEVICE,
            verbose=False
        )

        if not results:
            return jsonify({
                "success": False,
                "error": "The AI model returned no result."
            }), 500

        result = results[0]

        counts, detections = process_detections(result)

        total = (
            counts["cars"]
            + counts["motorcycles"]
            + counts["buses"]
            + counts["trucks"]
        )

        traffic_level = classify_traffic(total)

        annotated = result.plot(
            conf=True,
            labels=True,
            boxes=True,
            line_width=2
        )

        # result.plot() returns a BGR array (OpenCV's native channel
        # order), but PIL's Image.fromarray() assumes RGB. Without this
        # swap, red/blue channels get crossed — skies turn orange,
        # reds turn blue, giving a "negative"-looking image.
        annotated_rgb = annotated[..., ::-1]

        Image.fromarray(annotated_rgb).save(
            output_path,
            format="JPEG",
            quality=95,
            optimize=True
        )

        response = {
            "success": True,
            "media_type": "image",
            "model": MODEL_NAME,
            "settings": {
                "image_size": IMAGE_SIZE,
                "confidence": CONFIDENCE,
                "iou": IOU
            },
            "vehicles": {
                "cars": counts["cars"],
                "motorcycles": counts["motorcycles"],
                "buses": counts["buses"],
                "trucks": counts["trucks"],
                "total": total
            },
            "traffic": {
                "level": traffic_level
            },
            "detections": detections,
            "result_image": f"outputs/{output_name}"
        }

        print(
            f"Detection complete | "
            f"Cars: {counts['cars']} | "
            f"Motorcycles: {counts['motorcycles']} | "
            f"Buses: {counts['buses']} | "
            f"Trucks: {counts['trucks']} | "
            f"Total: {total} | "
            f"Traffic: {traffic_level}"
        )

        return jsonify(response), 200

    except Exception as error:

        print("IMAGE DETECTION ERROR:")
        print(error)

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500

    finally:

        try:
            if original_path and original_path.exists():
                original_path.unlink()

            if clean_path and clean_path.exists():
                clean_path.unlink()

        except Exception:
            pass


# ============================================================
# VIDEO DETECTION (BACKGROUND JOB)
# ============================================================

def run_video_detection(job_id, input_path, output_path, original_name):
    cap = None
    writer = None

    try:
        video_jobs[job_id]["status"] = "processing"

        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            raise RuntimeError("Could not open the uploaded video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Frame count from metadata is sometimes wrong/unavailable
        # depending on container format — fall back to the max-frames
        # cap so percentage still moves sensibly either way.
        raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_frames = min(raw_total, VIDEO_MAX_FRAMES) if raw_total > 0 else VIDEO_MAX_FRAMES

        # H.264 plays inline in every browser's <video> tag; mp4v
        # (the previous codec) often won't render in Chrome/Firefox,
        # only downloads. Fall back to mp4v only if avc1 isn't
        # available in this OpenCV build.
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            print(f"[{job_id}] avc1 codec unavailable, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_index = 0
        frames_processed = 0
        totals_per_frame = []
        aggregate_counts = {"cars": 0, "motorcycles": 0, "buses": 0, "trucks": 0}
        last_annotated = None
        last_logged_pct = -1

        print(f"[{job_id}] Analyzing video: {original_name} (~{expected_frames} frames expected)")

        while True:
            success, frame = cap.read()
            if not success:
                break
            if frame_index >= VIDEO_MAX_FRAMES:
                print(f"[{job_id}] Reached VIDEO_MAX_FRAMES ({VIDEO_MAX_FRAMES}); stopping early.")
                break

            if frame_index % VIDEO_FRAME_SKIP == 0:
                results = model.predict(
                    source=frame,
                    imgsz=VIDEO_IMAGE_SIZE,
                    conf=CONFIDENCE,
                    iou=IOU,
                    classes=list(VEHICLE_CLASSES.keys()),
                    augment=False,
                    device=DEVICE,
                    verbose=False,
                    half=(DEVICE != "cpu"),
                )
                result = results[0]
                counts, _ = process_detections(result)

                frame_total = sum(counts.values())
                totals_per_frame.append(frame_total)
                for key in aggregate_counts:
                    aggregate_counts[key] += counts[key]

                last_annotated = result.plot(conf=True, labels=True, boxes=True, line_width=2)
                frames_processed += 1

            writer.write(last_annotated if last_annotated is not None else frame)
            frame_index += 1

            # Cap reported progress at 99% until we've actually finished
            # writing/aggregating — avoids a premature "100%" flash.
            pct = min(99, int((frame_index / expected_frames) * 100)) if expected_frames else 0
            if pct != last_logged_pct:
                last_logged_pct = pct
                video_jobs[job_id]["progress"] = pct
                print(f"[{job_id}] Progress: {pct}% ({frame_index}/{expected_frames} frames)")

        cap.release()
        writer.release()
        cap = None
        writer = None

        if frames_processed == 0:
            raise RuntimeError("No frames could be read from the video.")

        avg_counts = {k: round(v / frames_processed) for k, v in aggregate_counts.items()}
        avg_total = round(sum(totals_per_frame) / frames_processed)
        traffic_level = classify_traffic(avg_total)

        result_payload = {
            "success": True,
            "media_type": "video",
            "model": MODEL_NAME,
            "settings": {
                "image_size": VIDEO_IMAGE_SIZE,
                "confidence": CONFIDENCE,
                "iou": IOU,
                "frame_skip": VIDEO_FRAME_SKIP,
            },
            "vehicles": {**avg_counts, "total": avg_total},
            "traffic": {"level": traffic_level},
            "frames_processed": frames_processed,
            "result_video": f"outputs/{output_path.name}",
        }

        video_jobs[job_id]["status"] = "done"
        video_jobs[job_id]["progress"] = 100
        video_jobs[job_id]["result"] = result_payload

        print(
            f"[{job_id}] Video detection complete | "
            f"frames: {frames_processed} | avg total: {avg_total} | traffic: {traffic_level}"
        )

    except Exception as error:
        print(f"[{job_id}] VIDEO DETECTION ERROR:", error)
        video_jobs[job_id]["status"] = "error"
        video_jobs[job_id]["error"] = str(error)

    finally:
        try:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            if input_path and input_path.exists():
                input_path.unlink()
        except Exception:
            pass


@app.route("/detect-video/start", methods=["POST"])
def start_video_detection():

    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file was uploaded."}), 400

    video_file = request.files["video"]

    if video_file.filename == "":
        return jsonify({"success": False, "error": "No video was selected."}), 400

    if not allowed_video(video_file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported video format. Use MP4, AVI, MOV, MKV or WEBM."
        }), 400

    original_name = secure_filename(video_file.filename)
    unique_id = uuid.uuid4().hex

    input_path = UPLOAD_FOLDER / f"{unique_id}_{original_name}"
    output_name = f"{unique_id}_detected.mp4"
    output_path = OUTPUT_FOLDER / output_name

    video_file.save(input_path)

    job_id = unique_id
    video_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "result": None,
        "error": None,
    }

    thread = threading.Thread(
        target=run_video_detection,
        args=(job_id, input_path, output_path, original_name),
        daemon=True,
    )
    thread.start()

    return jsonify({"success": True, "job_id": job_id}), 202


@app.route("/detect-video/status/<job_id>", methods=["GET"])
def video_detection_status(job_id):

    job = video_jobs.get(job_id)

    if not job:
        return jsonify({"success": False, "error": "Job not found."}), 404

    return jsonify({
        "success": True,
        "status": job["status"],
        "progress": job["progress"],
        "result": job["result"],
        "error": job["error"],
    }), 200


# ============================================================
# SERVE OUTPUT FILES (IMAGES + VIDEOS)
# ============================================================

@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_output(filename):

    return send_from_directory(
        OUTPUT_FOLDER,
        filename,
    )


# ============================================================
# SERVE UPLOADED FILES
# ============================================================

@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_upload(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename,
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("")
    print("=" * 60)
    print("SMART TRAFFIC AI SERVICE")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Image size (photos): {IMAGE_SIZE}")
    print(f"Image size (video frames): {VIDEO_IMAGE_SIZE}")
    print(f"Confidence: {CONFIDENCE}")
    print(f"Device: {DEVICE}")
    print("Vehicle classes: Car, Motorcycle, Bus, Truck")
    print("Server: http://127.0.0.1:5001")
    print("=" * 60)
    print("")

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,
    )