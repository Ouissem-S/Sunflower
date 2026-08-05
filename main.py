import cv2
import mediapipe as mp  # type: ignore[import-untyped]
import time



mp_hands = mp.solutions.hands # Creates the hand detector
mp_draw = mp.solutions.drawing_utils # Draw hand landmarks and connections

cap = cv2.VideoCapture(0) # Open camera 0
cap.set(3, 1280) # Set camera width to 1280px
cap.set(4, 720) # Set camera height to 720px

def main():
    while True:
        success, img = cap.read()
        attempt = 0

        # If the camera fails to load, retry up to 5 times with a 0.2s delay between attempts
        while not success and attempt < 5:
            time.sleep(0.2)
            success, img = cap.read()
            attempt += 1
        
        # If the camera still fails after retries, display an error and exit
        if not success:
            print("Failed to read frame")
            break
        
        # Flip the camera horizontally to create a mirror effect (left/right match the user's view)
        img = cv2.flip(img, 1)
        
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()


