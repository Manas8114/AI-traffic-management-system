import numpy as np
import cv2
from ultralytics import YOLO
from pymongo import MongoClient
from datetime import datetime, timedelta
import threading
from queue import Queue, Empty
import logging
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pytesseract
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from flask import Flask, render_template, Response, jsonify, request, Blueprint
from flask_cors import CORS
from cryptography.fernet import Fernet
import requests
from twilio.rest import Client
import uuid
import yaml
import signal
import sys
import argparse
import atexit
import gc
import psutil
import socket
import asyncio
import aiohttp
import hashlib
from pathlib import Path
import traceback
from typing import Literal
import webbrowser
from flask_swagger_ui import get_swaggerui_blueprint

__version__ = "2.1.0"

def load_config(config_path="config.yml"):
    """Load configuration from YAML file with environment variable substitution"""
    try:
        if not os.path.exists(config_path):
            logger.warning(f"Config file '{config_path}' not found, using default configuration")
            return get_default_config()

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        required_sections = ["logging", "traffic", "cameras", "detection", "prediction", "incidents", "system"]
        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing section '{section}' in config, using defaults")
                config[section] = get_default_config()[section]

        for section, values in config.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        env_var = value[2:-1]
                        config[section][key] = os.getenv(env_var, "")

        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return get_default_config()

def get_default_config():
    """Return default configuration values"""
    return {
        "logging": {"level": "INFO", "file": "traffic_system.log", "rotate_size_mb": 10, "backup_count": 5},
        "traffic": {"lane_divisions": 3, "min_green_time": 10, "yellow_duration": 3, "red_duration": 2},
        "cameras": {"urls": [], "reconnect_interval": 5, "frame_skip": 2},
        "detection": {"model_path": "yolov8n.pt", "confidence": 0.35, "speed_limit": 50, "tracking_history": 50},
        "prediction": {"update_interval": 60, "model_save_interval": 3600, "min_samples": 100},
        "incidents": {"stopped_threshold": 10, "congestion_threshold": 0.7, "alert_cooldown": 300},
        "system": {"worker_threads": 4, "max_queue_size": 100, "memory_limit_mb": 1024},
        "api": {"rate_limit": 100, "cache_timeout": 5},
        "security": {"enable_encryption": True, "jwt_expiry": 3600}
    }

def setup_logging(config):
    """Configure rotating log handler"""
    from logging.handlers import RotatingFileHandler

    log_level = getattr(logging, config.get("logging", {}).get("level", "INFO"))
    log_file = config.get("logging", {}).get("file", "traffic_system.log")
    max_bytes = config.get("logging", {}).get("rotate_size_mb", 10) * 1024 * 1024
    backup_count = config.get("logging", {}).get("backup_count", 5)
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            handler,
            logging.StreamHandler()
        ]
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    return logging.getLogger(__name__)

load_dotenv()
logger = logging.getLogger(__name__)

class MemoryManager:
    """Memory management utilities"""
    @staticmethod
    def get_memory_usage_mb():
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    @staticmethod
    def check_memory_usage(limit_mb=1024):
        """Check if memory usage exceeds limit and force garbage collection if needed"""
        usage_mb = MemoryManager.get_memory_usage_mb()
        if usage_mb > limit_mb:
            logger.warning(f"Memory usage high: {usage_mb:.2f}MB. Running garbage collection.")
            gc.collect()
            return True
        return False

class SecurityManager:
    """Handle security operations including encryption and API authentication"""
    def __init__(self, config):
        self.enable_encryption = config.get("security", {}).get("enable_encryption", True)
        self.jwt_expiry = config.get("security", {}).get("jwt_expiry", 3600)
        self.api_keys = {}
        try:
            self.encryption_key = os.getenv("ENCRYPTION_KEY") or Fernet.generate_key()
            self.cipher = Fernet(self.encryption_key)
            logger.info("Encryption initialized successfully")
        except Exception as e:
            logger.error(f"Encryption setup failed: {e}")
            self.cipher = None

    def generate_api_key(self, user_id=None):
        """Generate a new API key for a user or system component"""
        if user_id is None:
            user_id = str(uuid.uuid4())

        api_key = hashlib.sha256(os.urandom(32)).hexdigest()
        self.api_keys[api_key] = {
            "user_id": user_id,
            "created": datetime.now(),
            "expires": datetime.now() + timedelta(seconds=self.jwt_expiry)
        }

        return api_key

    def validate_api_key(self, api_key):
        """Validate an API key"""
        if not api_key:
            return False

        key_data = self.api_keys.get(api_key)
        if not key_data:
            return False

        if datetime.now() > key_data["expires"]:
            self.api_keys.pop(api_key, None)
            return False
        return True

    def encrypt_data(self, data):
        """Encrypt sensitive data"""
        if not self.enable_encryption or not self.cipher:
            return data
        try:
            if isinstance(data, dict):
                data = json.dumps(data).encode('utf-8')
            elif isinstance(data, str):
                data = data.encode('utf-8')
            return self.cipher.encrypt(data)
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            return data

    def decrypt_data(self, encrypted_data):
        """Decrypt encrypted data"""
        if not self.enable_encryption or not self.cipher:
            return encrypted_data
        try:
            return self.cipher.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return None

class VehicleTracker:
    """Tracks vehicle movement, speed and stores history"""
    def __init__(self, max_history=100, cleanup_interval=60):
        self.vehicle_history = defaultdict(deque)  # Use deque for better performance
        self.max_history = max_history
        self.last_cleanup = time.time()
        self.cleanup_interval = cleanup_interval
        self.vehicle_metadata = {}  # Store additional data about vehicles
        self.lock = threading.RLock()  # Thread-safe operations

    def update(self, track_id, x, y, cls=None, confidence=None):
        """Update position history for a tracked vehicle"""
        with self.lock:
            timestamp = time.time()
            # Update position history
            if track_id not in self.vehicle_history:
                self.vehicle_history[track_id] = deque(maxlen=self.max_history)
            self.vehicle_history[track_id].append((x, y, timestamp))
            # Update metadata
            if track_id not in self.vehicle_metadata:
                self.vehicle_metadata[track_id] = {
                    "first_seen": timestamp,
                    "class": cls,
                    "confidence": confidence,
                    "speeds": deque(maxlen=10),  # Keep recent speed measurements
                    "last_updated": timestamp
                }
            else:
                self.vehicle_metadata[track_id]["class"] = cls
                self.vehicle_metadata[track_id]["confidence"] = confidence
                self.vehicle_metadata[track_id]["last_updated"] = timestamp
            # Periodically clean up old vehicles
            if timestamp - self.last_cleanup > self.cleanup_interval:
                self._cleanup_history()

    def get_speed(self, track_id, time_window=3, pixels_per_meter=10):
        """
        Calculate vehicle speed in pixels/second over the given time window

        Args:
            track_id: Vehicle track ID
            time_window: Time window in seconds
            pixels_per_meter: Conversion factor for pixels to meters

        Returns:
            Speed in pixels/second
        """
        with self.lock:
            history = list(self.vehicle_history.get(track_id, []))
            if len(history) < 2:
                return 0.0
            # Find points within time window
            current_time = history[-1][2]
            window_start = current_time - time_window
            window_points = [p for p in history if p[2] >= window_start]
            if len(window_points) < 2:
                return 0.0
            # Calculate displacement
            start_x, start_y = window_points[0][0], window_points[0][1]
            end_x, end_y = window_points[-1][0], window_points[-1][1]
            dx = end_x - start_x
            dy = end_y - start_y
            distance = np.sqrt(dx**2 + dy**2)
            # Calculate time difference
            dt = window_points[-1][2] - window_points[0][2]
            if dt <= 0:
                return 0.0
            speed = distance / dt
            # Store speed in metadata
            if track_id in self.vehicle_metadata:
                self.vehicle_metadata[track_id]["speeds"].append(speed)
            return speed

    def get_avg_speed(self, track_id):
        """Get average speed from recent measurements"""
        with self.lock:
            if track_id not in self.vehicle_metadata:
                return 0.0
            speeds = self.vehicle_metadata[track_id]["speeds"]
            if not speeds:
                return 0.0
            return sum(speeds) / len(speeds)

    def get_direction(self, track_id):
        """Calculate movement direction (angle in degrees)"""
        with self.lock:
            history = list(self.vehicle_history.get(track_id, []))
            if len(history) < 5:  # Need sufficient history for reliable direction
                return None
            # Use recent points to determine direction
            recent = history[-5:]
            # Calculate average movement vector
            dx = recent[-1][0] - recent[0][0]
            dy = recent[-1][1] - recent[0][1]
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                return 0  # No significant movement
            # Calculate angle in degrees (0° is East, 90° is North)
            angle = np.degrees(np.arctan2(-dy, dx))  # Negative dy because y-axis is inverted in images
            # Convert to 0-360 range
            if angle < 0:
                angle += 360
            return angle

    def is_stopped(self, track_id, threshold=5.0, samples=10):
        """Check if vehicle is stopped (moving less than threshold)"""
        with self.lock:
            history = list(self.vehicle_history.get(track_id, []))
            if len(history) < samples:
                return False
            recent = history[-samples:]
            avg_x = sum(p[0] for p in recent) / len(recent)
            avg_y = sum(p[1] for p in recent) / len(recent)

            max_deviation = max(
                np.sqrt((p[0] - avg_x)**2 + (p[1] - avg_y)**2) for p in recent)
            

            return max_deviation < threshold

    def get_travel_time(self, track_id):
        """Get total tracking time for vehicle in seconds"""
        with self.lock:
            if track_id not in self.vehicle_metadata:
                return 0
            first_seen = self.vehicle_metadata[track_id]["first_seen"]
            last_seen = self.vehicle_metadata[track_id]["last_updated"]
            return last_seen - first_seen

    def get_trajectory(self, track_id, max_points=20):
        """Get trajectory points for visualization"""
        with self.lock:
            history = list(self.vehicle_history.get(track_id, []))
            if not history:
                return []
            if len(history) <= max_points:
                return [(int(p[0]), int(p[1])) for p in history]
            else:
                stride = len(history) // max_points
                return [(int(p[0]), int(p[1])) for p in history[::stride]]

    def _cleanup_history(self):
        """Remove vehicles that haven't been updated recently"""
        with self.lock:
            current_time = time.time()
            inactive_threshold = self.cleanup_interval
            inactive_ids = []
            for tid, metadata in self.vehicle_metadata.items():
                if current_time - metadata["last_updated"] > inactive_threshold:
                    inactive_ids.append(tid)
            for tid in inactive_ids:
                self.vehicle_history.pop(tid, None)
                self.vehicle_metadata.pop(tid, None)
            self.last_cleanup = current_time
            if inactive_ids:
                logger.debug(f"Cleaned up {len(inactive_ids)} inactive vehicles")

class EnhancedVehicleDetector:
    """Advanced vehicle detection and tracking with violation checking"""
    def __init__(self, model_path, confidence=0.35):
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model not found at {model_path}, using YOLO default")
                self.model = YOLO("yolov8n.pt")
            else:
                self.model = YOLO(model_path)
            self.tracker = VehicleTracker()
            self.violation_history = []
            self.confidence = confidence
            self.lane_boundaries = []
            self.speed_limit = CONFIG.get("detection", {}).get("speed_limit", 50)
            self.license_plate_detector = None

            self.classes_of_interest = {
                "car": {"priority": 1, "color": (0, 255, 0)},
                "truck": {"priority": 2, "color": (0, 255, 255)},
                "bus": {"priority": 2, "color": (255, 255, 0)},
                "motorcycle": {"priority": 1, "color": (0, 165, 255)},
                "bicycle": {"priority": 1, "color": (255, 0, 0)},
                "ambulance": {"priority": 10, "color": (255, 0, 0)},
                "police": {"priority": 8, "color": (0, 0, 255)},
                "fire truck": {"priority": 9, "color": (255, 0, 0)},
                "person": {"priority": 3, "color": (255, 165, 0)}
            }

            try:
                pytesseract.get_tesseract_version()
                self.ocr_available = True
            except:
                self.ocr_available = False
                logger.warning("Tesseract OCR not available, license plate detection disabled")

            logger.info(f"Vehicle detector initialized with model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize vehicle detector: {e}")
            # Fallback to default model
            self.model = YOLO("yolov8n.pt")
            self.tracker = VehicleTracker()
            self.violation_history = []
            self.confidence = 0.25

    def set_lane_boundaries(self, frame_width, lane_divisions=None):
        """Setup lane boundaries based on frame width"""
        divisions = lane_divisions or CONFIG.get("traffic", {}).get("lane_divisions", 3)

        self.lane_boundaries = [
            (i * frame_width / divisions, (i+1) * frame_width / divisions)
            for i in range(divisions)
        ]

        logger.debug(f"Lane boundaries set: {self.lane_boundaries}")

    def detect_license_plate(self, vehicle_crop):
        """Detect and recognize license plate"""
        if not self.ocr_available:
            return None

        try:
            if self.license_plate_detector is None:
                self.license_plate_detector = True
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            plate_candidates = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                area = w * h

                if 100 < area < 10000 and 2 < aspect_ratio < 6:
                    plate_candidates.append((x, y, w, h))
            if plate_candidates:
                x, y, w, h = max(plate_candidates, key=lambda rect: rect[2] * rect[3])
                plate_img = gray[y:y+h, x:x+w]
                plate_text = pytesseract.image_to_string(plate_img, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                plate_text = ''.join(c for c in plate_text if c.isalnum())
                if len(plate_text) >= 3:
                    return plate_text
            return None
        except Exception as e:
            logger.error(f"License plate detection error: {e}")
            return None

    def detect_and_track(self, frame):
        try:
            if frame is None:
                return defaultdict(int), defaultdict(int), False, None, []
            processed_frame = self._preprocess_frame(frame)
            if not self.lane_boundaries and processed_frame is not None:
                self.set_lane_boundaries(processed_frame.shape[1])
            results = self.model.track(processed_frame, persist=True, conf=self.confidence)
            annotated_frame = processed_frame.copy()
            lane_densities = defaultdict(int)
            vehicle_counts = defaultdict(int)
            emergency_present = False
            violations = []
            detected_objects = []
            if not results or not hasattr(results[0], 'boxes') or not results[0].boxes:
                return vehicle_counts, lane_densities, emergency_present, annotated_frame, violations
            for result in results:
                if not result.boxes:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = np.arange(len(boxes)) + int(time.time() * 1000) % 10000
                for idx, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
                    if idx >= len(track_ids):
                        continue
                    track_id = track_ids[idx]
                    cls = self.model.names[class_id]
                    if cls not in self.classes_of_interest:
                        continue
                    x1, y1, x2, y2 = box
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    detected_objects.append({
                        "track_id": track_id,
                        "class": cls,
                        "confidence": float(conf),
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "center": [float(x_center), float(y_center)]
                    })
                    vehicle_counts[cls] += 1
                    self.tracker.update(track_id, x_center, y_center, cls, float(conf))
                    lane_idx = 0
                    for i, (start, end) in enumerate(self.lane_boundaries):
                        if start <= x_center < end:
                            lane_idx = i
                            break
                    lane_densities[lane_idx] += 1
                    if cls in ["ambulance", "police", "fire truck"]:
                        emergency_present = True
                    speed = self.tracker.get_speed(track_id)
                    direction = self.tracker.get_direction(track_id)
                    if self._check_violations(track_id, cls, speed, x_center, y_center):
                        violations.append({
                            "type": "speed",
                            "track_id": track_id,
                            "vehicle_type": cls,
                            "speed": float(speed),
                            "position": [float(x_center), float(y_center)],
                            "timestamp": datetime.now().isoformat()
                        })
                    if cls in ["car", "truck", "bus"] and track_id % 10 == 0:
                        try:
                            vehicle_crop = processed_frame[int(y1):int(y2), int(x1):int(x2)]
                            if vehicle_crop.size > 0:
                                plate_text = self.detect_license_plate(vehicle_crop)
                                if plate_text:
                                    if track_id in self.tracker.vehicle_metadata:
                                        self.tracker.vehicle_metadata[track_id]["plate"] = plate_text
                        except Exception as e:
                            logger.debug(f"License plate detection failed: {e}")
                    self._annotate_detection(
                        annotated_frame, box, track_id, cls, conf, speed,
                        direction, self.tracker.is_stopped(track_id)
                    )
            return vehicle_counts, lane_densities, emergency_present, annotated_frame, violations
        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=True)
            return defaultdict(int), defaultdict(int), False, frame, []

    def _preprocess_frame(self, frame):
        """Apply preprocessing to improve detection"""
        try:
            max_dim = 1280
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            return frame
        except Exception as e:
            logger.error(f"Frame preprocessing error: {e}")
            return frame

    def _check_violations(self, track_id, cls, speed, x, y):
        """Check for traffic violations"""
        if speed > self.speed_limit:
            return True
        return False

    def _annotate_detection(self, frame, box, track_id, cls, conf, speed, direction, is_stopped):
        """Draw detection and tracking information on frame"""
        if cls in self.classes_of_interest:
            color = self.classes_of_interest[cls]["color"]
        else:
            color = ((track_id * 123) % 255, (track_id * 85) % 255, (track_id * 201) % 255)
        if is_stopped:
            r, g, b = color
            color = (0, 0, 255)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} #{track_id} ({conf:.2f})"
        if speed > 0:
            label += f" {speed:.1f}px/s"
        if direction is not None:
            label += f" {direction:.0f}°"

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        trajectory = self.tracker.get_trajectory(track_id, max_points=10)
        if len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                cv2.line(frame, trajectory[i], trajectory[i + 1], color, 2)

class TrafficPredictor:
    """LSTM-based traffic prediction with adaptive learning"""

    def __init__(self, model_path=None):
        self.sequence_length = 30
        self.feature_columns = ["time_of_day", "day_of_week", "density", "weather", "special_event"]
        self.min_samples_for_prediction = CONFIG.get("prediction", {}).get("min_samples", 100)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._load_or_build_model(model_path)
        self.history = []
        self.update_interval = CONFIG.get("prediction", {}).get("update_interval", 60)
        self.model_save_interval = CONFIG.get("prediction", {}).get("model_save_interval", 3600)
        self.last_training = 0
        self.last_save = 0
        self.training_lock = threading.Lock()
        self.model_dir = "models"

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _load_or_build_model(self, model_path):
        """Load existing model or build new one"""
        if model_path and os.path.exists(model_path):
            try:
                logger.info(f"Loading traffic prediction model from {model_path}")
                return load_model(model_path)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")

        logger.info("Building new LSTM prediction model")
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, len(self.feature_columns))),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        logger.info(f"Model built with {model.count_params()} parameters")
        return model

    def add_data_point(self, data):
        """Add new traffic data point to history"""
        timestamp = datetime.now()
        time_of_day = timestamp.hour + timestamp.minute / 60.0
        day_of_week = timestamp.weekday()
        density = data.get("density", 0.0)
        weather = data.get("weather", 0.0)
        special_event = data.get("special_event", 0.0)
        data_point = {
            "timestamp": timestamp,
            "time_of_day": time_of_day / 24.0,
            "day_of_week": day_of_week / 6.0,
            "density": density,
            "weather": weather,
            "special_event": special_event
        }
        self.history.append(data_point)
        current_time = time.time()
        if len(self.history) >= self.min_samples_for_prediction and (
            current_time - self.last_training > self.update_interval):
            self._schedule_training()
        if current_time - self.last_save > self.model_save_interval:
            self._save_model()

    def predict_future(self, minutes_ahead=15, current_data=None):
        """Predict future traffic conditions"""
        if len(self.history) < self.min_samples_for_prediction:
            logger.warning("Not enough data for prediction")
            return None

        try:
            recent_data = self.history[-self.sequence_length:]
            if len(recent_data) < self.sequence_length:
                padding = [recent_data[0]] * (self.sequence_length - len(recent_data))
                recent_data = padding + recent_data
            X = np.array([[
                d["time_of_day"],
                d["day_of_week"],
                d["density"],
                d["weather"],
                d["special_event"]
            ] for d in recent_data])
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = X_scaled.reshape(1, self.sequence_length, len(self.feature_columns))
            predicted_density = self.model.predict(X_scaled)[0][0]
            prediction_time = datetime.now() + timedelta(minutes=minutes_ahead)
            return {
                "predicted_time": prediction_time.isoformat(),
                "predicted_density": float(predicted_density),
                "confidence": self._calculate_confidence(len(self.history))
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return None

    def _calculate_confidence(self, data_points):
        """Calculate prediction confidence based on data size"""
        base_confidence = 0.5
        data_factor = min(1.0, data_points / (self.min_samples_for_prediction * 5))
        return base_confidence + (0.5 * data_factor)

    def _schedule_training(self):
        """Schedule model training in background thread"""
        if self.training_lock.acquire(blocking=False):
            try:
                training_thread = threading.Thread(target=self._train_model)
                training_thread.daemon = True
                training_thread.start()
            except Exception as e:
                logger.error(f"Failed to schedule training: {e}")
                self.training_lock.release()

    def _train_model(self):
        """Train model with available data"""
        try:
            logger.info("Starting model training...")
            start_time = time.time()
            if len(self.history) < self.min_samples_for_prediction:
                logger.warning("Not enough data for training")
                return
            X, y = [], []

            for i in range(len(self.history) - self.sequence_length):
                sequence = self.history[i:i+self.sequence_length]
                target = self.history[i+self.sequence_length]["density"]
                sequence_features = np.array([[
                    d["time_of_day"],
                    d["day_of_week"],
                    d["density"],
                    d["weather"],
                    d["special_event"]
                ] for d in sequence])
                X.append(sequence_features)
                y.append(target)
            if not X:
                logger.warning("No training sequences could be created")
                return
            X = np.array(X)
            y = np.array(y)
            X_reshaped = X.reshape(-1, len(self.feature_columns))
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
            self.model.fit(
                X_scaled, y,
                epochs=5,
                batch_size=32,
                verbose=0
            )
            training_time = time.time() - start_time
            logger.info(f"Model training completed in {training_time:.2f}s")
            self.last_training = time.time()

            if time.time() - self.last_save > self.model_save_interval:
                self._save_model()
        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
        finally:
            self.training_lock.release()

    def _save_model(self):
        """Save model to disk"""
        try:
            filename = f"traffic_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            filepath = os.path.join(self.model_dir, filename)
            self.model.save(filepath)
            self.last_save = time.time()
            logger.info(f"Model saved to {filepath}")
            self._cleanup_old_models()
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def _cleanup_old_models(self, keep_last=5):
        """Remove old model files, keeping the most recent ones"""
        try:
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith("traffic_model_") and f.endswith(".h5")]
            if len(model_files) <= keep_last:
                return
            model_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.model_dir, f)))
            for f in model_files[:-keep_last]:
                file_path = os.path.join(self.model_dir, f)
                os.remove(file_path)
                logger.debug(f"Removed old model file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up old models: {e}")

class IncidentDetector:
    """Detect traffic incidents and anomalies"""
    def __init__(self, config=None):
        self.stopped_threshold = config.get("incidents", {}).get("stopped_threshold", 10)
        self.congestion_threshold = config.get("incidents", {}).get("congestion_threshold", 0.7)
        self.alert_cooldown = config.get("incidents", {}).get("alert_cooldown", 300)
        self.recent_incidents = {}
        self.incident_history = []
        self.last_alert_time = defaultdict(float)

    def detect_incidents(self, vehicle_tracker, lane_densities, emergency_present=False):
        """
        Detect traffic incidents based on vehicle tracking data

        Args:
            vehicle_tracker: VehicleTracker instance
            lane_densities: Dictionary of lane densities
            emergency_present: Flag for emergency vehicle presence

        Returns:
            List of detected incidents
        """
        incidents = []
        current_time = time.time()
        stopped_vehicles = self._detect_stopped_vehicles(vehicle_tracker)
        for vehicle_id, data in stopped_vehicles.items():
            incident_type = "stopped_vehicle"
            incident_id = f"{incident_type}_{vehicle_id}"
            if current_time - self.last_alert_time[incident_id] < self.alert_cooldown:
                continue
            incident = {
                "id": incident_id,
                "type": incident_type,
                "vehicle_id": vehicle_id,
                "vehicle_type": data["type"],
                "location": data["location"],
                "duration": data["duration"],
                "timestamp": datetime.now().isoformat()
            }

            incidents.append(incident)
            self.recent_incidents[incident_id] = incident
            self.last_alert_time[incident_id] = current_time
        congested_lanes = self._detect_congestion(lane_densities)
        for lane_id, density in congested_lanes.items():
            incident_type = "congestion"
            incident_id = f"{incident_type}_lane_{lane_id}"
            if current_time - self.last_alert_time[incident_id] < self.alert_cooldown:
                continue
            incident = {
                "id": incident_id,
                "type": incident_type,
                "lane_id": lane_id,
                "density": density,
                "timestamp": datetime.now().isoformat()
            }

            incidents.append(incident)
            self.recent_incidents[incident_id] = incident
            self.last_alert_time[incident_id] = current_time
        if incidents:
            self.incident_history.extend(incidents)
            if len(self.incident_history) > 1000:
                self.incident_history = self.incident_history[-1000:]

        return incidents

    def _detect_stopped_vehicles(self, vehicle_tracker):
        """Detect vehicles that have stopped unusually"""
        stopped_vehicles = {}
        for track_id, metadata in vehicle_tracker.vehicle_metadata.items():
            if vehicle_tracker.get_travel_time(track_id) < self.stopped_threshold:
                continue
            if not vehicle_tracker.is_stopped(track_id):
                continue
            vehicle_type = metadata.get("class", "unknown")
            history = list(vehicle_tracker.vehicle_history.get(track_id, []))
            if not history:
                continue
            last_pos = history[-1]
            location = (last_pos[0], last_pos[1])
            duration = time.time() - metadata.get("last_updated", time.time())
            stopped_vehicles[track_id] = {
                "type": vehicle_type,
                "location": location,
                "duration": duration
            }
        return stopped_vehicles

    def _detect_congestion(self, lane_densities):
        """Detect congested lanes"""
        congested_lanes = {}
        for lane_id, count in lane_densities.items():
            if count > self.congestion_threshold:
                congested_lanes[lane_id] = count
        return congested_lanes

    def get_incident_summary(self):
        """Return summary of recent incidents"""
        current_time = time.time()
        active_incidents = {}
        for incident_id, incident in self.recent_incidents.items():
            last_alert = self.last_alert_time.get(incident_id, 0)
            if current_time - last_alert < self.alert_cooldown:
                active_incidents[incident_id] = incident
        return {
            "active_count": len(active_incidents),
            "active_incidents": list(active_incidents.values()),
            "total_detected": len(self.incident_history)
        }

class CameraManager:
    """Handle video camera streams"""
    def __init__(self, camera_urls=None):
        self.cameras = {}
        self.streams = {}
        self.frame_queues = {}  # Queue for each camera
        self.active = True
        self.reconnect_interval = CONFIG.get("cameras", {}).get("reconnect_interval", 5)
        self.frame_skip = CONFIG.get("cameras", {}).get("frame_skip", 2)
        if camera_urls:
            for idx, url in enumerate(camera_urls):
                self.add_camera(idx, url)

    def add_camera(self, camera_id, url):
        """Add a new camera source"""
        if camera_id in self.cameras:
            logger.warning(f"Camera ID {camera_id} already exists, replacing")
        self.cameras[camera_id] = {
            "url": url,
            "status": "initializing",
            "last_frame_time": 0,
            "fps": 0,
            "frame_count": 0,
            "reconnect_attempts": 0
        }
        self.frame_queues[camera_id] = Queue(maxsize=10)
        threading.Thread(
            target=self._camera_worker,
            args=(camera_id, url),
            daemon=True
        ).start()
        logger.info(f"Added camera {camera_id} with URL: {url}")

    def remove_camera(self, camera_id):
        """Remove a camera source"""
        if camera_id in self.cameras:
            self.cameras[camera_id]["status"] = "removing"
            # Wait for thread to notice and clean up
            time.sleep(0.5)
            self.cameras.pop(camera_id, None)
            self.streams.pop(camera_id, None)
            self.frame_queues.pop(camera_id, None)
            logger.info(f"Removed camera {camera_id}")

    def get_frame(self, camera_id, timeout=1):
        """Get the latest frame from a camera"""
        if camera_id not in self.frame_queues:
            return None
        try:
            return self.frame_queues[camera_id].get(timeout=timeout)
        except Empty:
            return None

    def get_status(self, camera_id=None):
        """Get status of cameras"""
        if camera_id is not None:
            return self.cameras.get(camera_id)
        return self.cameras

    def _camera_worker(self, camera_id, url):
        """Worker thread to read frames from camera"""
        frame_count = 0
        while self.active and camera_id in self.cameras:
            try:
                if self.cameras[camera_id]["status"] == "removing":
                    break
                if camera_id not in self.streams or self.streams[camera_id] is None:
                    logger.info(f"Connecting to camera {camera_id}: {url}")
                    self.cameras[camera_id]["status"] = "connecting"
                    stream = cv2.VideoCapture(url)
                    if not stream.isOpened():
                        raise Exception(f"Failed to open camera stream: {url}")
                    self.streams[camera_id] = stream
                    self.cameras[camera_id]["status"] = "streaming"
                    self.cameras[camera_id]["reconnect_attempts"] = 0
                    logger.info(f"Camera {camera_id} connected successfully")
                ret, frame = self.streams[camera_id].read()
                if not ret:
                    raise Exception("Failed to read frame")
                frame_count += 1
                if frame_count % self.frame_skip != 0:
                    continue
                current_time = time.time()
                elapsed = current_time - self.cameras[camera_id].get("last_frame_time", current_time)
                if elapsed > 0:
                    self.cameras[camera_id]["fps"] = 1 / elapsed
                self.cameras[camera_id]["last_frame_time"] = current_time
                self.cameras[camera_id]["frame_count"] += 1
                if self.frame_queues[camera_id].full():
                    try:
                        self.frame_queues[camera_id].get_nowait()
                    except Empty:
                        pass
                self.frame_queues[camera_id].put(frame)
            except Exception as e:
                logger.error(f"Camera {camera_id} error: {e}")
                self.cameras[camera_id]["status"] = "error"
                self.cameras[camera_id]["reconnect_attempts"] += 1
                if camera_id in self.streams and self.streams[camera_id] is not None:
                    self.streams[camera_id].release()
                    self.streams[camera_id] = None
                reconnect_delay = min(
                    self.reconnect_interval * self.cameras[camera_id]["reconnect_attempts"],
                    60  # Max 60 seconds between retries
                )
                logger.info(f"Reconnecting to camera {camera_id} in {reconnect_delay}s (attempt {self.cameras[camera_id]['reconnect_attempts']})")
                time.sleep(reconnect_delay)

    def shutdown(self):
        """Clean shutdown of all camera streams"""
        self.active = False
        for camera_id, stream in self.streams.items():
            if stream is not None:
                stream.release()
        logger.info("All camera streams released")

class AlertManager:
    """Manage incident alerts and notifications"""
    def __init__(self, config=None):
        self.alert_methods = []
        self.alert_history = []
        self.notification_cooldown = 300
        self.last_notification = {}
        self._initialize_twilio()
        self._initialize_email()

    def _initialize_twilio(self):
        """Initialize Twilio SMS service if configured"""
        try:
            twilio_sid = os.getenv("TWILIO_SID")
            twilio_token = os.getenv("TWILIO_TOKEN")
            twilio_from = os.getenv("TWILIO_FROM")

            if twilio_sid and twilio_token and twilio_from:
                self.twilio_client = Client(twilio_sid, twilio_token)
                self.twilio_from = twilio_from
                self.alert_methods.append("sms")
                logger.info("Twilio SMS notifications initialized")
            else:
                self.twilio_client = None
                logger.debug("Twilio not configured, SMS alerts disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Twilio: {e}")
            self.twilio_client = None

    def _initialize_email(self):
        """Initialize email notifications if configured"""
        try:
            smtp_server = os.getenv("SMTP_SERVER")
            smtp_port = os.getenv("SMTP_PORT")
            smtp_user = os.getenv("SMTP_USER")
            smtp_pass = os.getenv("SMTP_PASS")

            if all([smtp_server, smtp_port, smtp_user, smtp_pass]):
                import smtplib
                from email.message import EmailMessage

                self.email_config = {
                    "server": smtp_server,
                    "port": int(smtp_port),
                    "user": smtp_user,
                    "pass": smtp_pass,
                    "from": smtp_user
                }
                self.alert_methods.append("email")
                logger.info("Email notifications initialized")
            else:
                self.email_config = None
                logger.debug("SMTP not configured, email alerts disabled")
        except Exception as e:
            logger.error(f"Failed to initialize email: {e}")
            self.email_config = None

    def send_alert(self, incident, recipients=None, methods=None):
        """
        Send alert about incident

        Args:
            incident: Incident data
            recipients: Dict with keys 'email' and 'phone' containing lists of recipients
            methods: List of alert methods to use (default: all available)
        """
        if not recipients:
            return
        if not methods:
            methods = self.alert_methods
        current_time = time.time()
        incident_id = incident.get("id", str(hash(str(incident))))
        if incident_id in self.last_notification:
            time_since_last = current_time - self.last_notification[incident_id]
            if time_since_last < self.notification_cooldown:
                logger.debug(f"Alert for incident {incident_id} suppressed (cooldown: {time_since_last:.1f}s)")
                return
        alert_text = self._format_alert_message(incident)
        results = {}
        if "sms" in methods and self.twilio_client and "phone" in recipients:
            results["sms"] = self._send_sms_alert(alert_text, recipients["phone"])
        if "email" in methods and self.email_config and "email" in recipients:
            results["email"] = self._send_email_alert(alert_text, recipients["email"], incident)
        alert_record = {
            "incident": incident,
            "timestamp": datetime.now().isoformat(),
            "methods": methods,
            "recipients": recipients,
            "results": results
        }
        self.alert_history.append(alert_record)
        self.last_notification[incident_id] = current_time
        return alert_record

    def _format_alert_message(self, incident):
        """Format an incident into an alert message"""
        incident_type = incident.get("type", "unknown")
        if incident_type == "stopped_vehicle":
            vehicle_type = incident.get("vehicle_type", "unknown")
            duration = incident.get("duration", 0)
            return f"ALERT: Stopped {vehicle_type} detected for {duration:.1f} seconds"
        elif incident_type == "congestion":
            lane_id = incident.get("lane_id", "unknown")
            density = incident.get("density", 0)
            return f"ALERT: Traffic congestion detected in lane {lane_id} (density: {density})"
        else:
            return f"ALERT: Traffic incident detected: {incident}"

    def _send_sms_alert(self, message, recipients):
        """Send SMS alert via Twilio"""
        if not self.twilio_client:
            return {"status": "error", "message": "Twilio not configured"}
        results = []
        for phone in recipients:
            try:
                sms = self.twilio_client.messages.create(
                    body=message,
                    from_=self.twilio_from,
                    to=phone
                )
                results.append({"phone": phone, "status": "sent", "sid": sms.sid})
                logger.info(f"SMS alert sent to {phone}")
            except Exception as e:
                results.append({"phone": phone, "status": "error", "error": str(e)})
                logger.error(f"Failed to send SMS to {phone}: {e}")
        return results

    def _send_email_alert(self, message, recipients, incident):
        """Send email alert via SMTP"""
        if not self.email_config:
            return {"status": "error", "message": "Email not configured"}
        import smtplib
        from email.message import EmailMessage
        results = []
        try:
            msg = EmailMessage()
            msg["Subject"] = f"Traffic Alert: {incident.get('type', 'Incident')}"
            msg["From"] = self.email_config["from"]
            html_content = f"""
            <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; }}
                        .alert {{ background-color: #ffebee; padding: 15px; border-left: 5px solid #f44336; }}
                        .details {{ margin-top: 15px; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    </style>
                </head>
                <body>
                    <div class="alert">
                        <h2>Traffic Alert</h2>
                        <p>{message}</p>
                    </div>
                    <div class="details">
                        <h3>Incident Details</h3>
                        <table>
            """
            for key, value in incident.items():
                html_content += f"<tr><th>{key}</th><td>{value}</td></tr>"
            html_content += """
                        </table>
                    </div>
                </body>
            </html>
            """

            msg.set_content(message)
            msg.add_alternative(html_content, subtype="html")
            with smtplib.SMTP(self.email_config["server"], self.email_config["port"]) as server:
                server.starttls()
                server.login(self.email_config["user"], self.email_config["pass"])
                for email in recipients:
                    try:
                        msg["To"] = email
                        server.send_message(msg)
                        results.append({"email": email, "status": "sent"})
                        logger.info(f"Email alert sent to {email}")
                    except Exception as e:
                        results.append({"email": email, "status": "error", "error": str(e)})
                        logger.error(f"Failed to send email to {email}: {e}")
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
            results.append({"status": "error", "error": str(e)})
        return results

class DatabaseManager:
    """Handle data storage and retrieval"""
    def __init__(self, config=None):
        self.db_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.db_name = os.getenv("MONGODB_DB", "traffic")
        self.client = None
        self.db = None
        self._connect()

    def _connect(self):
        """Establish database connection"""
        try:
            self.client = MongoClient(self.db_uri)
            self.db = self.client[self.db_name]
            logger.info(f"Connected to database: {self.db_name}")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            self.db = None

    def store_traffic_data(self, data):
        """Store traffic analysis data"""
        if self.db is None:
            self._connect()
            if self.db is None:
                return False
        try:
            if "timestamp" not in data:
                data["timestamp"] = datetime.now()
            result = self.db.traffic_data.insert_one(data)
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to store traffic data: {e}")
            return None

    def store_incident(self, incident):
        """Store incident data"""
        if self.db is None:
            self._connect()
            if self.db is None:
                return False
        try:
            if "timestamp" not in incident:
                incident["timestamp"] = datetime.now()
            result = self.db.incidents.insert_one(incident)
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to store incident: {e}")
            return None

    def get_recent_traffic_data(self, hours=24, limit=1000):
        """Get recent traffic data"""
        if self.db is None:
            self._connect()
            if self.db is None:
                return []
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            cursor = self.db.traffic_data.find(
                {"timestamp": {"$gte": start_time}},
                sort=[("timestamp", -1)],
                limit=limit
            )
            return list(cursor)
        except Exception as e:
            logger.error(f"Failed to retrieve traffic data: {e}")
            return []

    def get_incidents(self, hours=24, limit=100):
        """Get recent incidents"""
        if self.db is None:
            self._connect()
            if self.db is None:
                return []
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            cursor = self.db.incidents.find(
                {"timestamp": {"$gte": start_time}},
                sort=[("timestamp", -1)],
                limit=limit
            )
            return list(cursor)
        except Exception as e:
            logger.error(f"Failed to retrieve incidents: {e}")
            return []

    def get_incident_stats(self, days=7):
        """Get incident statistics for the specified time period"""
        if self.db is None:
            self._connect()
            if self.db is None:
                return {}
        try:
            start_time = datetime.now() - timedelta(days=days)
            pipeline = [
                {"$match": {"timestamp": {"$gte": start_time}}},
                {"$group": {
                    "_id": "$type",
                    "count": {"$sum": 1},
                    "latest": {"$max": "$timestamp"}
                }}
            ]
            results = self.db.incidents.aggregate(pipeline)
            stats = {item["_id"]: {"count": item["count"], "latest": item["latest"]}
                     for item in results}
            return stats
        except Exception as e:
            logger.error(f"Failed to get incident stats: {e}")
            return {}

    def store_traffic_light_status(self, traffic_light_status):
        """Store traffic light status"""
        if self.db is None:
            self._connect()
            if self.db is None:
                return False
        try:
            if "timestamp" not in traffic_light_status:
                traffic_light_status["timestamp"] = datetime.now()
            result = self.db.traffic_light_statuses.insert_one(traffic_light_status)
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to store traffic light status: {e}")
            return None

    def get_traffic_light_statuses(self):
        """Get traffic light statuses"""
        if self.db is None:
            self._connect()
            if self.db is None:
                return []
        try:
            cursor = self.db.traffic_light_statuses.find().sort("timestamp", -1)
            return list(cursor)
        except Exception as e:
            logger.error(f"Failed to retrieve traffic light statuses: {e}")
            return []

    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("Database connection closed")

class TrafficController:
    """Main traffic control system"""
    def __init__(self, config=None):
        self.config = config or {}
        self.running = False
        self.processing_queue = Queue(maxsize=CONFIG.get("system", {}).get("max_queue_size", 100))
        self.camera_manager = None
        self.detector = None
        self.predictor = None
        self.incident_detector = None
        self.alert_manager = None
        self.db_manager = None
        self.worker_threads = []
        self.security_manager = None
        self.last_stats_update = 0
        self.system_stats = {}
        self.traffic_light_controller = None

        # Current traffic state
        self.current_state = {
            "vehicle_counts": defaultdict(int),
            "lane_densities": defaultdict(int),
            "active_incidents": [],
            "emergency_present": False,
            "last_update": datetime.now(),
            "traffic_lights": defaultdict(str)
        }

        # Shared state lock
        self.state_lock = threading.RLock()

    def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing traffic control system")
            self.security_manager = SecurityManager(self.config)
            model_path = self.config.get("detection", {}).get("model_path", "yolov8n.pt")
            confidence = self.config.get("detection", {}).get("confidence", 0.35)
            self.detector = EnhancedVehicleDetector(model_path, confidence)
            camera_urls = self.config.get("cameras", {}).get("urls", [])
            self.camera_manager = CameraManager(camera_urls)
            self.predictor = TrafficPredictor()
            self.incident_detector = IncidentDetector(self.config)
            self.alert_manager = AlertManager(self.config)
            self.db_manager = DatabaseManager(self.config)

            # Initialize traffic light controller
            api_url = self.config.get("traffic_lights", {}).get("api_url")
            api_key = self.config.get("traffic_lights", {}).get("api_key")
            self.traffic_light_controller = TrafficLightController(api_url, api_key)

            # Register cleanup handler
            atexit.register(self.shutdown)

            return True
        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            return False

    def control_traffic_lights(self):
        # Example logic to control traffic lights based on current state
        with self.state_lock:
            lane_densities = self.current_state["lane_densities"]
            emergency_present = self.current_state["emergency_present"]

            # Example decision logic
            if emergency_present:
                # Change all lights to green for emergency vehicles
                for intersection_id in self.current_state["traffic_lights"].keys():
                    self.traffic_light_controller.change_light_color(intersection_id, "green")
            else:
                # Change lights based on lane densities
                for lane_id, density in lane_densities.items():
                    intersection_id = f"intersection_{lane_id}"
                    if density > 0.7:  # High density
                        self.traffic_light_controller.change_light_color(intersection_id, "green")
                    else:
                        self.traffic_light_controller.change_light_color(intersection_id, "red")

    def _process_frame(self, camera_id, frame):
        """Process a single camera frame"""
        try:
            vehicle_counts, lane_densities, emergency_present, annotated_frame, violations = \
                self.detector.detect_and_track(frame)
            incidents = self.incident_detector.detect_incidents(
                self.detector.tracker,
                lane_densities,
                emergency_present
            )
            timestamp = datetime.now()
            with self.state_lock:
                self.current_state["vehicle_counts"] = dict(vehicle_counts)
                self.current_state["lane_densities"] = dict(lane_densities)
                self.current_state["emergency_present"] = emergency_present
                self.current_state["active_incidents"] = incidents
                self.current_state["last_update"] = timestamp
                self.current_state["camera_id"] = camera_id

            # Control traffic lights based on the current state
            self.control_traffic_lights()

            total_vehicles = sum(vehicle_counts.values())
            avg_density = sum(lane_densities.values()) / max(len(lane_densities), 1)
            prediction_data = {
                "density": avg_density,
                "vehicle_count": total_vehicles,
                "emergency_present": 1 if emergency_present else 0,
                "weather": 0,
                "special_event": 0
            }
            self.predictor.add_data_point(prediction_data)
            if timestamp.second % 10 == 0:  # Every 10 seconds
                db_data = {
                    "timestamp": timestamp,
                    "camera_id": camera_id,
                    "vehicle_counts": dict(vehicle_counts),
                    "lane_densities": dict(lane_densities),
                    "emergency_present": emergency_present,
                    "incidents": incidents
                }
                threading.Thread(
                    target=self.db_manager.store_traffic_data,
                    args=(db_data,),
                    daemon=True
                ).start()
            if incidents:
                self._handle_incidents(incidents)
        except Exception as e:
            logger.error(f"Frame processing error: {e}", exc_info=True)

    def _handle_incidents(self, incidents):
        """Handle detected incidents"""
        for incident in incidents:
            self.db_manager.store_incident(incident)
            recipients = {
                "email": ["traffic-alerts@example.com"],
                "phone": ["+14129912633"]
            }
            self.alert_manager.send_alert(incident, recipients)

    def start(self):
        """Start traffic control system"""
        if self.running:
            logger.warning("System already running")
            return False
        logger.info("Starting traffic control system")
        self.running = True
        num_workers = self.config.get("system", {}).get("worker_threads", 4)
        for i in range(num_workers):
            thread = threading.Thread(
                target=self._process_frames_worker,
                name=f"worker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        stats_thread = threading.Thread(
            target=self._system_stats_worker,
            name="stats-monitor",
            daemon=True
        )
        stats_thread.start()
        self.worker_threads.append(stats_thread)
        logger.info(f"System started with {num_workers} worker threads")
        return True

    def stop(self):
        """Stop traffic control system"""
        logger.info("Stopping traffic control system")
        self.running = False
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=5)
        logger.info("System stopped")

    def shutdown(self):
        """Clean shutdown of the system"""
        logger.info("Shutting down traffic control system")
        self.running = False
        if self.camera_manager:
            self.camera_manager.shutdown()
        if self.db_manager:
            self.db_manager.close()
        logger.info("Traffic system shutdown complete")

    def get_system_stats(self):
        """Get system statistics"""
        self.system_stats["memory_usage_mb"] = MemoryManager.get_memory_usage_mb()
        self.system_stats["queue_size"] = self.processing_queue.qsize()
        self.system_stats["worker_threads"] = len([t for t in self.worker_threads if t.is_alive()])
        return self.system_stats

    def get_current_state(self):
        """Get current traffic state"""
        with self.state_lock:
            time_since_update = (datetime.now() - self.current_state["last_update"]).total_seconds()
            if time_since_update > 10:  # Only predict if data is stale
                prediction = self.predictor.predict_future(minutes_ahead=15)
                if prediction:
                    self.current_state["prediction"] = prediction
            return dict(self.current_state)

    def _process_frames_worker(self):
        """Worker thread to process camera frames"""
        logger.info(f"Worker thread {threading.current_thread().name} started")

        while self.running:
            try:
                camera_status = self.camera_manager.get_status()
                if not camera_status:
                    time.sleep(1)
                    continue
                for camera_id, status in camera_status.items():
                    if status["status"] != "streaming":
                        continue
                    frame = self.camera_manager.get_frame(camera_id, timeout=0.1)
                    if frame is None:
                        continue
                    self._process_frame(camera_id, frame)
                MemoryManager.check_memory_usage(
                    self.config.get("system", {}).get("memory_limit_mb", 1024)
                )
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                time.sleep(1)

    def _system_stats_worker(self):
        """Worker thread to monitor system statistics"""
        logger.info("System stats monitoring started")
        while self.running:
            try:
                stats = {
                    "timestamp": datetime.now().isoformat(),
                    "memory_usage_mb": MemoryManager.get_memory_usage_mb(),
                    "cpu_percent": psutil.cpu_percent(),
                    "thread_count": threading.active_count(),
                    "queue_size": self.processing_queue.qsize()
                }
                if self.camera_manager:
                    camera_status = self.camera_manager.get_status()
                    stats["cameras"] = len(camera_status)
                    stats["active_cameras"] = sum(1 for s in camera_status.values()
                                                 if s["status"] == "streaming")
                self.system_stats = stats
                if stats["memory_usage_mb"] > self.config.get("system", {}).get("memory_limit_mb", 1024) * 0.9:
                    logger.warning(f"High memory usage: {stats['memory_usage_mb']:.1f}MB")
            except Exception as e:
                logger.error(f"Stats monitoring error: {e}")
            time.sleep(5)

class TrafficLightController:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def change_light_color(self, intersection_id, color):
        url = f"{self.api_url}/intersections/{intersection_id}/lights"
        payload = {"color": color}
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            logger.info(f"Changed light color at intersection {intersection_id} to {color}")
        except Exception as e:
            logger.error(f"Failed to change light color: {e}")

class APIServer:
    """REST API for traffic system"""
    def __init__(self, traffic_controller, config=None):
        self.app = Flask(__name__)
        self.blueprint = Blueprint('api', __name__)
        self.traffic_controller = traffic_controller
        self.config = config or {}
        CORS(self.app)
        self._setup_routes()
        self.app.register_blueprint(self.blueprint, url_prefix='/api')
        self._setup_error_handlers()
        @self.app.route('/traffic-lights', methods=['GET'])
        def traffic_lights_status():
            return render_template_string('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Traffic Light Status</title>
                    <style>
                        .status-box {
                            margin: 20px;
                            padding: 20px;
                            border: 2px solid #333;
                        }
                        .light-status {
                            margin: 10px 0;
                            padding: 8px;
                            width: 200px;
                        }
                    </style>
                </head>
                <body>
                    <h1>Traffic Light Status Monitor</h1>
                    <div class="status-box">
                        {% for intersection, status in traffic_lights.items() %}
                        <div>
                            <label>{{ intersection }}:</label>
                            <input type="text"
                                   class="light-status"
                                   value="{{ status|upper }}"
                                   style="background-color: {{ status }}"
                                   readonly>
                        </div>
                        {% endfor %}
                    </div>
                </body>
                </html>
            ''', traffic_lights=self.traffic_controller.current_state["traffic_lights"])

    def _setup_routes(self):
        """Set up API routes"""
        self.blueprint.route('/status', methods=['GET'])(self.get_system_status)
        self.blueprint.route('/traffic', methods=['GET'])(self.get_traffic_state)
        self.blueprint.route('/incidents', methods=['GET'])(self.get_incidents)
        self.blueprint.route('/cameras', methods=['GET'])(self.get_cameras)
        self.blueprint.route('/cameras/<camera_id>', methods=['GET'])(self.get_camera)
        self.blueprint.route('/cameras/<camera_id>/stream', methods=['GET'])(self.stream_camera)
        self.blueprint.route('/predict', methods=['GET'])(self.get_prediction)
        self.blueprint.route('/control/add_camera', methods=['POST'])(self.add_camera)
        self.blueprint.route('/control/remove_camera', methods=['POST'])(self.remove_camera)
        self.blueprint.route('/traffic_lights', methods=['GET'])(self.get_traffic_lights)
        self.blueprint.route('/traffic_lights/<intersection_id>', methods=['PUT'])(self.change_traffic_light)

    def _setup_error_handlers(self):
        """Set up API error handlers"""
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({"error": "Not found"}), 404
        @self.app.errorhandler(500)
        def server_error(error):
            return jsonify({"error": "Server error"}), 500

    def get_system_status(self):
        """Get system status"""
        stats = self.traffic_controller.get_system_stats()
        response = {
            "status": "running" if self.traffic_controller.running else "stopped",
            "version": __version__,
            "uptime": time.time() - self.traffic_controller.system_stats.get("start_time", time.time()),
            "stats": stats
        }
        return jsonify(response)

    def get_traffic_state(self):
        """Get current traffic state"""
        state = self.traffic_controller.get_current_state()
        for key, value in state.items():
            if isinstance(value, defaultdict):
                state[key] = dict(value)
        return jsonify(state)

    def get_incidents(self):
        """Get active and recent incidents"""
        active = self.traffic_controller.incident_detector.get_incident_summary()
        hours = request.args.get('hours', default=24, type=int)
        limit = request.args.get('limit', default=100, type=int)
        historical = self.traffic_controller.db_manager.get_incidents(hours, limit)
        for incident in historical:
            if '_id' in incident:
                incident['_id'] = str(incident['_id'])
        response = {
            "active": active,
            "historical": historical
        }
        return jsonify(response)

    def get_cameras(self):
        """Get all camera information"""
        cameras = self.traffic_controller.camera_manager.get_status()
        filtered_cameras = {}
        for cam_id, data in cameras.items():
            filtered_cameras[str(cam_id)] = {
                "status": data["status"],
                "fps": data.get("fps", 0),
                "frame_count": data.get("frame_count", 0),
                "last_frame_time": data.get("last_frame_time", 0)
            }
        return jsonify(filtered_cameras)

    def get_camera(self, camera_id):
        """Get specific camera information"""
        try:
            camera_id = int(camera_id)
        except ValueError:
            return jsonify({"error": "Invalid camera ID"}), 400
        camera = self.traffic_controller.camera_manager.get_status(camera_id)
        if not camera:
            return jsonify({"error": "Camera not found"}), 404
        filtered_camera = {
            "status": camera["status"],
            "fps": camera.get("fps", 0),
            "frame_count": camera.get("frame_count", 0),
            "last_frame_time": camera.get("last_frame_time", 0)
        }
        return jsonify(filtered_camera)

    def stream_camera(self, camera_id):
        """Stream camera feed as MJPEG"""
        try:
            camera_id = int(camera_id)
        except ValueError:
            return jsonify({"error": "Invalid camera ID"}), 400

        def generate_frames():
            while True:
                frame = self.traffic_controller.camera_manager.get_frame(camera_id)
                if frame is None:
                    time.sleep(0.1)
                    continue
                if request.args.get('annotate', 'false').lower() == 'true':
                    vehicle_counts, lane_densities, emergency_present, annotated_frame, violations = \
                        self.traffic_controller.detector.detect_and_track(frame)
                    if annotated_frame is not None:
                        frame = annotated_frame
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)

        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def get_prediction(self):
        """Get traffic prediction"""
        minutes = request.args.get('minutes', default=15, type=int)
        prediction = self.traffic_controller.predictor.predict_future(minutes_ahead=minutes)
        if prediction is None:
            return jsonify({
                "error": "Insufficient data for prediction",
                "required_samples": self.traffic_controller.predictor.min_samples_for_prediction,
                "current_samples": len(self.traffic_controller.predictor.history)
            }), 400
        return jsonify(prediction)

    def add_camera(self):
        """Add a new camera"""
        if not request.json or 'url' not in request.json:
            return jsonify({"error": "Missing camera URL"}), 400
        url = request.json['url']
        camera_id = request.json.get('id')
        if camera_id is None:
            existing_cameras = self.traffic_controller.camera_manager.get_status()
            camera_id = max(existing_cameras.keys(), default=0) + 1
        self.traffic_controller.camera_manager.add_camera(camera_id, url)
        return jsonify({
            "status": "success",
            "message": f"Camera added with ID {camera_id}",
            "camera_id": camera_id
        })

    def remove_camera(self):
        """Remove a camera"""
        if not request.json or 'id' not in request.json:
            return jsonify({"error": "Missing camera ID"}), 400
        try:
            camera_id = int(request.json['id'])
        except ValueError:
            return jsonify({"error": "Invalid camera ID"}), 400
        camera = self.traffic_controller.camera_manager.get_status(camera_id)
        if not camera:
            return jsonify({"error": "Camera not found"}), 404
        self.traffic_controller.camera_manager.remove_camera(camera_id)
        return jsonify({
            "status": "success",
            "message": f"Camera {camera_id} removed"
        })

    def get_traffic_lights(self):
        """Get traffic light status"""
        with self.traffic_controller.state_lock:
            traffic_lights = self.traffic_controller.current_state["traffic_lights"]
        return jsonify(traffic_lights)

    def change_traffic_light(self, intersection_id):
        """Change traffic light color"""
        if not request.json or 'color' not in request.json:
            return jsonify({"error": "Missing color"}), 400
        color = request.json['color']
        if color not in ["red", "yellow", "green"]:
            return jsonify({"error": "Invalid color"}), 400
        self.traffic_controller.traffic_light_controller.change_light_color(intersection_id, color)
        return jsonify({"status": "success", "message": f"Traffic light at intersection {intersection_id} changed to {color}"})

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the API server"""
        self.app.run(host=host, port=port, debug=debug)

def main():
    parser = argparse.ArgumentParser(description="Traffic Monitoring System")
    parser.add_argument("-c", "--config", type=str, default="config.yml", help="Configuration file")
    parser.add_argument("-p", "--port", type=int, default=5000, help="API server port")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode with sample video")
    parser.add_argument("--quickstart", action="store_true", help="Display quickstart guide")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()
    global CONFIG
    CONFIG = load_config(args.config)
    global logger
    logger = setup_logging(CONFIG)
    logger.info(f"Traffic Monitoring System v{__version__} starting up")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"OpenCV version: {cv2.__version__}")
    controller = TrafficController(CONFIG)
    if not controller.initialize():
        logger.error("Failed to initialize traffic controller")
        return 1
    if not controller.start():
        logger.error("Failed to start traffic controller")
        return 1
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        controller.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    api_server = APIServer(controller, CONFIG)
    logger.info(f"API server starting on port {args.port}")
    api_server.run(port=args.port, debug=args.debug)
    return 0

if __name__ == "__main__":
    sys.exit(main())
