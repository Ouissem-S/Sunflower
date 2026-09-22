"""Visual overlay effects drawn on top of the camera feed."""
import math
import random
import time

import cv2

# Letter colors for the "IMAGINATION" caption, cycling red -> violet like the meme
IMAGINATION_COLORS = [
    (0, 0, 255),      # red
    (0, 128, 255),    # orange
    (0, 255, 255),    # yellow
    (0, 255, 0),      # green
    (255, 50, 50),    # blue
    (238, 130, 238),  # violet
]


def draw_rainbow(img, center, base_radius=120, band_gap=16, thickness=16, alpha=0.45):
    """Draw a semi-transparent rainbow arc centered above `center`."""
    overlay = img.copy()
    colors = [
        (238, 130, 238),  # violet
        (255, 50, 50),    # blue
        (0, 255, 0),      # green
        (0, 255, 255),    # yellow
        (0, 128, 255),    # orange
        (0, 0, 255),      # red
    ]

    for i, color in enumerate(colors):
        radius = base_radius - (i * band_gap)
        cv2.ellipse(overlay, center, (radius, radius), 0, 180, 360, color, thickness, cv2.LINE_AA)

    # glossy shine: a thin white arc near the inner edge
    inner_radius = base_radius - (len(colors) - 1) * band_gap + 4
    cv2.ellipse(overlay, center, (inner_radius, inner_radius), 0, 200, 340, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_sparkles(img, center, spread_radius=140, count=20):
    """Scatter small white sparkle dots above `center`."""
    x, y = center
    for _ in range(count):
        angle = math.radians(random.uniform(180, 360))  # only scatter over the top half
        radius = random.uniform(spread_radius * 0.5, spread_radius * 1.15)
        sx = int(x + radius * math.cos(angle))
        sy = int(y + radius * math.sin(angle))
        brightness = random.randint(200, 255)
        cv2.circle(img, (sx, sy), random.choice([1, 2, 3]), (brightness, brightness, brightness), -1, cv2.LINE_AA)


def draw_imagination_text(img, origin, text="IMAGINATION", font_scale=1.3, thickness=3, wave_amplitude=10):
    """Draw `text` letter by letter in rainbow colors, each bobbing on its own
    phase - the wavy "IMAGINATION" caption from the SpongeBob meme."""
    x, y = origin
    t = time.time()

    for i, char in enumerate(text):
        color = IMAGINATION_COLORS[i % len(IMAGINATION_COLORS)]
        offset_y = int(wave_amplitude * math.sin(t * 4 + i * 0.6))
        (char_w, _), _ = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        cv2.putText(
            img,
            char,
            (x, y + offset_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        x += char_w + 2
