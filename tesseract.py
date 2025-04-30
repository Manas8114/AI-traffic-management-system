import cv2
import numpy as np
import pytesseract
import re
import time
import os
from datetime import datetime

# Configuration
VIDEO_SOURCE = 0  # Use 0 for webcam, or provide a file path for video file
OUTPUT_FOLDER = "captured_plates"
CONFIDENCE_THRESHOLD = 70  # Minimum confidence for plate detection
SAVE_FRAMES = True  # Whether to save frames with detected plates
LOG_FILE = "plate_log.txt"

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Set up Tesseract path - change this to your tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class LicensePlateRecognizer:
    def __init__(self):
        # Load the cascade for license plate detection
        self.plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
        
        self.detected_plates = set()
        self.last_detection = {}  # To prevent duplicates in short time
        
        # Set up logging
        self.log_file = open(LOG_FILE, "a")
        self.log_file.write(f"\n--- New Session Started: {datetime.now()} ---\n")
    
    def cleanup(self):
        self.log_file.close()
    
    def preprocess_plate_img(self, plate_img):
        """Preprocess the license plate image for better OCR results"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        
        _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((1, 1), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        
        return thresh
    
    def clean_plate_text(self, text):
        """Clean and validate the extracted license plate text"""
        text = text.strip().replace('\n', '').replace(' ', '')
        
        text = re.sub(r'[^A-Za-z0-9]', '', text)
        
        text = text.upper()
        
        if len(text) < 4 or len(text) > 10:
            return None
            
        return text
    
    def extract_plate_text(self, plate_img):
        """Extract text from license plate image using OCR"""
        processed_img = self.preprocess_plate_img(plate_img)
        
        config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(processed_img, config=config)
        
        plate_text = self.clean_plate_text(text)
        
        return plate_text
    
    def save_plate(self, frame, plate_img, plate_text):
        """Save detected plate image and log the detection"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if SAVE_FRAMES:
            frame_filename = f"{OUTPUT_FOLDER}/frame_{plate_text}_{timestamp}.jpg"
            cv2.imwrite(frame_filename, frame)
            
            plate_filename = f"{OUTPUT_FOLDER}/plate_{plate_text}_{timestamp}.jpg"
            cv2.imwrite(plate_filename, plate_img)
        
        log_entry = f"[{timestamp}] Detected plate: {plate_text}\n"
        print(log_entry.strip())
        self.log_file.write(log_entry)
        self.log_file.flush()
    
    def is_new_detection(self, plate_text):
        """Check if this is a new plate detection or a duplicate"""
        current_time = time.time()
        
        # If we've never seen this plate, or it's been a while since we last saw it
        if plate_text not in self.last_detection or (current_time - self.last_detection[plate_text]) > 10:
            self.last_detection[plate_text] = current_time
            if plate_text not in self.detected_plates:
                self.detected_plates.add(plate_text)
                return True
            return False  # We've seen it before, but not recently
        
        # We've seen this plate recently
        return False
    
    def process_frame(self, frame):
        """Process a video frame to detect and recognize license plates"""
        # Create a copy of the frame to draw on
        result_frame = frame.copy()
        
        # Convert to grayscale for plate detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect license plates
        plates = self.plate_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in plates:
            # Draw rectangle around the plate
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract the plate region
            plate_img = frame[y:y+h, x:x+w]
            
            # Get plate text using OCR
            plate_text = self.extract_plate_text(plate_img)
            
            if plate_text:
                # Display text above the rectangle
                cv2.putText(result_frame, plate_text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Check if this is a new detection
                if self.is_new_detection(plate_text):
                    self.save_plate(frame, plate_img, plate_text)
        
        return result_frame

def main():
    # Initialize the recognizer
    recognizer = LicensePlateRecognizer()
    
    try:
        # Open video source
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
        
        print(f"License Plate Recognition started. Press 'q' to quit.")
        print(f"Saving detected plates to: {OUTPUT_FOLDER}")
        
        while True:
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break
            
            # Process the frame
            result_frame = recognizer.process_frame(frame)
            
            # Display the result
            cv2.imshow('License Plate Recognition', result_frame)
            
            # Check for user quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        recognizer.cleanup()
        print("License Plate Recognition stopped.")

if __name__ == "__main__":
    main()