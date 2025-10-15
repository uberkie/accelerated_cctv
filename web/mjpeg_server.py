"""Lightweight MJPEG server using Flask.

Run this module to start an HTTP server that serves an index page and an
MJPEG stream at /stream.mjpg. Callers can push JPEG frames to
`FrameBuffer.set_frame(jpeg_bytes)` to update the stream.
"""

from flask import Flask, Response, render_template_string
import threading

app = Flask(__name__)

_buf_lock = threading.Lock()
_latest_frame = None


def set_frame(jpeg_bytes: bytes) -> None:
    global _latest_frame
    with _buf_lock:
        _latest_frame = jpeg_bytes


def generate_mjpeg():
    boundary = "--frame"
    while True:
        with _buf_lock:
            frame = _latest_frame
        if frame is None:
            # no frame yet
            import time
            time.sleep(0.05)
            continue
        yield (b"--%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % (boundary.encode(), len(frame))) + frame + b"\r\n"


@app.route("/")
def index():
    return render_template_string("""
    <html><body>
    <h1>MJPEG Stream</h1>
    <img src="/stream.mjpg" />
    </body></html>
    """)


@app.route("/stream.mjpg")
def stream():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


def run(host="0.0.0.0", port=5000):
    # Run Flask in a separate thread-safe mode
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    run()
