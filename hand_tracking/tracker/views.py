from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
import cv2
import numpy as np
import threading
import time

class HandTracker:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.hand_position = {'x': 0, 'y': 0, 'detected': False}
        self.lock = threading.Lock()
        self.frame_width = 640
        self.frame_height = 480
        
    def start_camera(self):
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.is_running = True
            
    def stop_camera(self):
        self.is_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
    
    def detect_hand_improved(self, frame):
        """
        Improved hand detection using multiple color ranges and better preprocessing
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Method 1: HSV color space (multiple ranges for different skin tones)
        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        
        lower_skin2 = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin2 = np.array([20, 255, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        # Method 2: YCrCb color space (more robust for skin detection)
        lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        mask3 = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine all masks
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Dilate to make hand more visible
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Apply Gaussian blur
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the mask for debugging
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        if contours:
            # Filter contours by area
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 3000]
            
            if valid_contours:
                # Find the largest contour
                max_contour = max(valid_contours, key=cv2.contourArea)
                area = cv2.contourArea(max_contour)
                
                if area > 3000:  # Minimum area threshold
                    # Calculate convex hull
                    hull = cv2.convexHull(max_contour)
                    
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(max_contour)
                    
                    # Calculate centroid
                    M = cv2.moments(max_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Draw visualizations
                        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
                        cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)
                        cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
                        
                        # Update hand position
                        height, width = frame.shape[:2]
                        with self.lock:
                            self.hand_position = {
                                'x': cx / width,
                                'y': cy / height,
                                'detected': True
                            }
                        
                        # Display information
                        cv2.putText(frame, f"Hand Position: ({cx}, {cy})", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Area: {int(area)}", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, "HAND DETECTED", (10, 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        return frame, True
        
        # No hand detected
        with self.lock:
            self.hand_position['detected'] = False
        
        cv2.putText(frame, "NO HAND DETECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Show your hand to camera", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame, False
    
    def get_hand_position(self):
        with self.lock:
            return self.hand_position.copy()
    
    def generate_frames(self):
        self.start_camera()
        
        while self.is_running:
            success, frame = self.camera.read()
            if not success:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hand with improved algorithm
            frame, detected = self.detect_hand_improved(frame)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame = buffer.tobytes()
            
            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Global tracker instance
tracker = HandTracker()

def index(request):
    return render(request, 'index.html')

def video_feed(request):
    return StreamingHttpResponse(
        tracker.generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def get_hand_position(request):
    """API endpoint to get current hand position"""
    position = tracker.get_hand_position()
    return JsonResponse(position)