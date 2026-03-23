"""
Flask web application for the parking spot detection dashboard.
Streams processed video via MJPEG and exposes a stats JSON endpoint.
"""

from flask import Flask, Response, jsonify, render_template

from config import DEFAULT_MASK_PATH, DEFAULT_VIDEO_PATH
from src.detector import ParkingDetector

app = Flask(__name__, template_folder="templates")

detector = None


def init_detector(video_path=None, mask_path=None, step=None):
    """Initialize the shared detector instance."""
    global detector
    detector = ParkingDetector(
        video_path=video_path or DEFAULT_VIDEO_PATH,
        mask_path=mask_path or DEFAULT_MASK_PATH,
        step=step,
        loop=True,
    )


def generate_mjpeg():
    """Wrap the detector frame generator into an MJPEG stream."""
    for jpeg_bytes in detector.process_frames():
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg_bytes
            + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stats")
def stats():
    return jsonify(detector.stats)
