#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# simview.py - live simulator view for the dashboard (MJPEG over HTTP).
#
# Grabs PIXELS from the X display the AirSim/Blocks window renders on (e.g.
# the Xvfb display :99) and pushes them to the browser as an MJPEG stream on
# the dashboard's own port (/sim.mjpg). It never calls the AirSim API, so it
# cannot affect the simulator, the controller or the model.
#
# Push streaming instead of VNC: VNC is request/response, so over a ~190 ms
# round trip it tops out at ~5 fps; a pushed stream is only bandwidth-bound
# (~16 KB/frame at 854x480, i.e. ~470 KB/s at 30 fps).
#
# Each SimView has one capture thread shared by all its viewers; it starts
# with the first viewer and stops a few seconds after the last one leaves.
# ---------------------------------------------------------------------------
import io
import os
import threading
import time

from flask import Response, request

try:
    from PIL import Image, ImageGrab
except ImportError:  # pillow missing -> view unavailable
    ImageGrab = None


def display_available(display):
    if ImageGrab is None or not display.startswith(":"):
        return False
    num = display[1:].split(".")[0]
    return os.path.exists(f"/tmp/.X11-unix/X{num}")


class SimView:
    def __init__(self, display=":99", fps=30, width=854, quality=70, idle_stop_s=5.0):
        self.display = display
        self.period = 1.0 / max(1, fps)
        self.width = width
        self.quality = quality
        self.idle_stop_s = idle_stop_s
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._viewers = 0
        self._last_viewer_time = 0.0
        self._thread = None

    def available(self):
        return display_available(self.display)

    def _capture_loop(self):
        while True:
            t0 = time.monotonic()
            with self._cond:
                if self._viewers == 0 and t0 - self._last_viewer_time > self.idle_stop_s:
                    self._thread = None
                    return
            try:
                im = ImageGrab.grab(xdisplay=self.display)
                if im.width > self.width:
                    im = im.resize((self.width, round(im.height * self.width / im.width)), Image.BILINEAR)
                buf = io.BytesIO()
                im.convert("RGB").save(buf, "JPEG", quality=self.quality)
                frame = buf.getvalue()
            except Exception:  # display gone, etc. - retry in a second
                frame = None
            with self._cond:
                if frame is not None:
                    self._frame = frame
                    self._seq += 1
                self._cond.notify_all()
            time.sleep(max(0.0, self.period - (time.monotonic() - t0)) if frame else 1.0)

    def _ensure_thread(self):
        with self._cond:
            if self._thread is None:
                self._thread = threading.Thread(target=self._capture_loop, name="simview", daemon=True)
                self._thread.start()

    def stream(self):
        """Flask response: multipart/x-mixed-replace MJPEG stream."""
        def gen():
            with self._cond:
                self._viewers += 1
            self._ensure_thread()
            seq = -1
            try:
                while True:
                    with self._cond:
                        fresh = self._cond.wait_for(lambda: self._seq != seq, timeout=2.0)
                        frame, seq = self._frame, self._seq
                    if not fresh:
                        self._ensure_thread()  # capture thread may have just stopped
                        continue
                    if frame is None:
                        continue
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
            finally:
                with self._cond:
                    self._viewers -= 1
                    self._last_viewer_time = time.monotonic()

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})


def register(server, simview, simview_hd=None):
    """Add /sim.mjpg (and /sim.mjpg?q=hd for the enlarged/fullscreen view) to
    the dashboard's Flask server. Each quality has its own capture thread,
    which only runs while someone is watching that quality."""
    @server.route("/sim.mjpg")
    def sim_mjpg():
        view = simview_hd if (request.args.get("q") == "hd" and simview_hd is not None) else simview
        if not view.available():
            return Response("simulator display not available", status=503)
        return view.stream()
