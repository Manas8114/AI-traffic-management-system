# Advanced Traffic Monitoring System

This system uses computer vision and machine learning to monitor traffic, detect incidents, and optimize traffic flow.

## Features

- Real-time vehicle detection and tracking
- Traffic density analysis
- Incident detection (stopped vehicles, congestion)
- Adaptive traffic light control
- Traffic flow prediction
- Web-based monitoring dashboard

## Requirements

- Python 3.8+
- OpenCV
- YOLOv8
- TensorFlow
- MongoDB (optional)
- Twilio (optional, for alerts)

## Installation

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Download YOLOv8 model: `yolo download yolov8n.pt` 
4. Create `.env` file with your configuration
5. Run the system: `python traffic_system.py`

## Configuration

Edit `config.yml` to customize:
- Camera sources
- Traffic lane settings
- Detection thresholds
- System parameters

## Web Interface

Access the dashboard at `http://localhost:5000` when the system is running. Few things i need to be added in dashboard color of light and cameras with no in case of multiple cameras.