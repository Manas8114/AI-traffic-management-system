import cv2

# Test local webcam
cap = cv2.VideoCapture("http://192.168.1.7:8080/video")
if not cap.isOpened():
    print("Error: Could not open local webcam.")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Webcam Test', frame)
        cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
