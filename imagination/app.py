"""Main application loop: capture frames, track hands, and render effects."""
import math

import cv2

from imagination.camera import open_camera, read_frame
from imagination.effects import draw_imagination_text, draw_rainbow, draw_sparkles
from imagination.hand_tracking import (
    create_hands_detector,
    draw_finger_tips,
    draw_hand_landmarks,
    is_hand_raised,
)

# Bringing both thumbs closer than this (in pixels) triggers the rainbow effect
PINCH_DISTANCE_THRESHOLD = 40


def run():
    cap = open_camera()

    with create_hands_detector() as hands:
        while True:
            success, img = read_frame(cap)

            # If the camera still fails after retries, display an error and exit
            if not success:
                print("Failed to read frame")
                break

            # Flip the camera horizontally to create a mirror effect (left/right match the user's view)
            img = cv2.flip(img, 1)

            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            left_thumb = None
            right_thumb = None

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    draw_hand_landmarks(img, hand_landmarks)
                    draw_finger_tips(img, hand_landmarks, w, h)

                    label = handedness.classification[0].label
                    thumb_tip = hand_landmarks.landmark[4]

                    if label == "Left" and is_hand_raised(hand_landmarks):
                        left_thumb = thumb_tip
                    elif label == "Right" and is_hand_raised(hand_landmarks):
                        right_thumb = thumb_tip

                if left_thumb and right_thumb:
                    lx, ly = left_thumb.x * w, left_thumb.y * h
                    rx, ry = right_thumb.x * w, right_thumb.y * h

                    distance = math.hypot(rx - lx, ry - ly)

                    cv2.putText(
                        img,
                        f"Distance: {int(distance)}",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

                    if distance < PINCH_DISTANCE_THRESHOLD:
                        mx = int((lx + rx) / 2)
                        my = int((ly + ry) / 2) - 60

                        draw_rainbow(img, (mx, my))
                        draw_sparkles(img, (mx, my))
                        draw_imagination_text(img, (30, 100))

            cv2.imshow("Image", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
