#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# record_demo.py - record a backup video of one demo run.
#
# Starts one demo through the same DemoController the dashboard uses (same
# safety rules), and while it runs records:
#   <stamp>_<model>_sim.mp4        the simulator display, 1280x720 @ --fps
#                                  (pixels from the X display, no AirSim API)
#   <stamp>_<model>_dashboard.mp4  the dashboard page as seen in a browser
# Recording stops a few seconds after the demo process exits.
#
#   /mnt/disks/data/aerotail-dashboard-venv/bin/python dashboard/record_demo.py \
#       --model recurrent_ppo:0 --vehicle Drone1 --episodes 1
# Requires the dashboard (app.py) to be running and the simulator to render
# on --display (see README, "Simulator view").
# ---------------------------------------------------------------------------
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

import imageio_ffmpeg
from PIL import ImageGrab
from playwright.sync_api import sync_playwright

import control
import simview

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")


class DisplayRecorder:
    """Grabs the X display at a fixed frame rate and encodes H.264 with ffmpeg.
    If a grab is late the previous frame is repeated, so the video keeps real time."""

    def __init__(self, display, fps, path):
        self.display, self.fps, self.path = display, fps, path
        first = ImageGrab.grab(xdisplay=display)
        self.size = first.size
        self._stop = threading.Event()
        self.frames = 0
        self._proc = subprocess.Popen(
            [FFMPEG, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{self.size[0]}x{self.size[1]}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        period = 1.0 / self.fps
        start = time.monotonic()
        last = None
        while not self._stop.is_set():
            try:
                last = ImageGrab.grab(xdisplay=self.display).convert("RGB").tobytes()
            except Exception:
                pass  # keep repeating the previous frame
            # write as many frames as wall-clock time requires (repeats if we fell behind)
            due = int((time.monotonic() - start) / period) + 1
            while last is not None and self.frames < due:
                self._proc.stdin.write(last)
                self.frames += 1
            time.sleep(max(0.0, start + self.frames * period - time.monotonic()))

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        self._proc.stdin.close()
        self._proc.wait()


def main():
    p = argparse.ArgumentParser(description="Record a backup video of one dashboard demo run.")
    p.add_argument("--model", default="recurrent_ppo:0", help="algo:stage, e.g. ppo:0")
    p.add_argument("--vehicle", default="Drone1", choices=sorted(control.VEHICLES))
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--dashboard-url", default="http://127.0.0.1:8050/")
    p.add_argument("--display", default=":99")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--no-dashboard-video", action="store_true")
    args = p.parse_args()

    try:
        urllib.request.urlopen(args.dashboard_url, timeout=5)
    except Exception:
        sys.exit(f"dashboard not reachable at {args.dashboard_url} - start dashboard/app.py first")
    if not simview.display_available(args.display):
        sys.exit(f"X display {args.display} not available - run the simulator windowed on it (see README)")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"{time.strftime('%Y%m%d_%H%M%S')}_{args.model.replace(':', '_stage')}")
    sim_path, dash_path = stem + "_sim.mp4", stem + "_dashboard.mp4"

    ctl = control.DemoController()
    rec = DisplayRecorder(args.display, args.fps, sim_path)
    tmp_video_dir = tempfile.mkdtemp(prefix="dashrec_")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = None
        if not args.no_dashboard_video:
            # Larger viewport recorded down to 1080p shows more panels without
            # CSS zoom (which misplaces Plotly axis titles).
            context = browser.new_context(viewport={"width": 2240, "height": 1260},
                                          record_video_dir=tmp_video_dir,
                                          record_video_size={"width": 1920, "height": 1080})
            page = context.new_page()
            page.goto(args.dashboard_url, wait_until="commit")
            page.wait_for_selector(".pill", timeout=30000)
            # hide the control bar: its form shows defaults, not necessarily the running model
            page.add_style_tag(content=".page { max-width: none; } .control-bar { display: none; }")
        rec.start()
        ok, msg = ctl.start(args.model, args.vehicle, args.episodes)
        print(msg, flush=True)
        if not ok:
            rec.stop()
            browser.close()
            os.remove(sim_path)
            sys.exit(1)
        t0 = time.time()
        deadline = t0 + args.episodes * 90 + 60
        time.sleep(3)
        while ctl.status()["running"] and time.time() < deadline:
            time.sleep(1)
        if ctl.status()["running"]:
            print("timeout - stopping demo", flush=True)
            ctl.stop()
        time.sleep(4)  # show the final RUN FINISHED state
        rec.stop()
        if page is not None:
            webm = page.video.path()
            context.close()
            subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-i", webm, "-c:v", "libx264", "-preset",
                            "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            dash_path], check=True)
        browser.close()
    shutil.rmtree(tmp_video_dir, ignore_errors=True)
    print(f"recorded {time.time() - t0:.0f} s")
    print(f"simulator video: {sim_path} ({rec.frames} frames @ {args.fps} fps)")
    if not args.no_dashboard_video:
        print(f"dashboard video: {dash_path}")


if __name__ == "__main__":
    main()
