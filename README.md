# Traffic Object Detection and Counting

## Overview

This project detects and counts vehicles in a traffic video using YOLO and OpenCV.

The system detects:

* Cars
* Buses
* Bikes/Motorcycles

Trucks are excluded from the counting process.

## Features

* Real-time-style vehicle detection from video
* YOLO-based object detection
* ByteTrack-based object tracking
* Vehicle ID tracking
* Line-crossing based vehicle counting
* Detection confidence displayed on the video
* Output video generated with bounding boxes and counts

## Project Structure

```text
traffic-object-counter/
│
├── main.py
├── README.md
├── requirements.txt
├── input/
│   └── traffic.mp4
└── output/
    └── result.mp4
```

## Technologies Used

* Python
* YOLO
* Ultralytics
* OpenCV
* ByteTrack

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd traffic-object-counter
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the input video

Place the traffic video inside:

```text
input/traffic.mp4
```

### 4. Run the project

```bash
python main.py
```

The processed video will be generated in:

```text
output/result.mp4
```

## Approach

The input traffic video is processed frame by frame using a YOLO object detection model.

The model detects relevant vehicle classes and assigns tracking IDs using ByteTrack. A virtual counting line is placed across the road. When a tracked vehicle crosses this line, it is counted once.

The system counts cars, bikes/motorcycles, and buses while excluding trucks.

## Output

The output video contains:

* Bounding boxes around detected vehicles
* Vehicle class labels
* Tracking IDs
* Confidence scores
* Counting line
* Vehicle counts

## Demo Video

https://drive.google.com/file/d/1cObml-pPXFII8Dml85tgg3hiJmy37m5m/view?usp=drive_link

## Author

Bhavya
