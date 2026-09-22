"""Hand detection and landmark drawing helpers built on MediaPipe Hands."""
import cv2
import mediapipe as mp  # type: ignore[import-untyped]

mp_hands = mp.solutions.hands  # Creates the hand detector
mp_draw = mp.solutions.drawing_utils  # Draw hand landmarks and connections

FINGER_TIP_IDS = {
    "Thumb": 4,
    "Index": 8,
    "Middle": 12,
    "Ring": 16,
    "Pinky": 20,
}


def create_hands_detector(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7):
    """Create a configured MediaPipe Hands detector (use as a context manager)."""
    return mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def is_hand_raised(hand_landmarks):
    """A hand counts as "raised" when its middle fingertip is above the wrist
    and near the top of the frame."""
    wrist = hand_landmarks.landmark[0]
    middle_tip = hand_landmarks.landmark[12]
    return middle_tip.y < wrist.y and middle_tip.y < 0.13


def draw_hand_landmarks(img, hand_landmarks):
    """Draw the full hand skeleton (landmarks + connections)."""
    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)


def draw_finger_tips(img, hand_landmarks, w, h):
    """Label and highlight each fingertip on the frame."""
    for name, tip_id in FINGER_TIP_IDS.items():
        landmark = hand_landmarks.landmark[tip_id]
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.putText(
            img,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
