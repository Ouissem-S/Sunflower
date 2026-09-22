"""Camera capture helpers."""
import time

import cv2

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


def open_camera(index=0, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """Open a camera device and configure its resolution."""
    cap = cv2.VideoCapture(index)
    cap.set(3, width)  # Set camera width
    cap.set(4, height)  # Set camera height
    return cap


def read_frame(cap, max_attempts=5, retry_delay=0.2):
    """Read a frame from the camera, retrying briefly on failure.

    Returns (success, frame); success is False if all attempts failed.
    """
    success, img = cap.read()
    attempt = 0

    # If the camera fails to load, retry up to `max_attempts` times with a delay between attempts
    while not success and attempt < max_attempts:
        time.sleep(retry_delay)
        success, img = cap.read()
        attempt += 1

    return success, img
