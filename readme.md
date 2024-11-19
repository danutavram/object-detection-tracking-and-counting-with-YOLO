Vehicle Detection, Tracking and Counting with YOLO 

Code Explanation

main_tracker.py

    Handles the main vehicle detection loop:
    Loads the YOLO model.
    Reads the input video and processes each frame.
    Detects vehicles in each frame and tracks them.
    Updates vehicle counts when vehicles cross entry or exit lines.
    Outputs a processed video showing tracked vehicles and counts.

model_loader.py

    Loads the YOLO model using the ultralytics YOLO library.
    vehicle_detector.py
    Detects vehicles in each frame by running the YOLO model on the video frame.
    Filters detections to focus on specific vehicle types (e.g., cars, trucks).

tracker.py

    Tracks the detected vehicles using their center points across multiple frames.
    Manages the tracking logic, including assigning IDs to vehicles and handling vehicle crossing states (entry, exit).

video_manager.py

    Handles video I/O: Opening the input video file, writing the processed video, and releasing resources.

Example of Entry and Exit Lines Visualization

    In each frame, the system will display:

        Yellow line for entry.
        Red line for exit.
        Vehicle counts for both entry and exit at the top of the video.

Results

    This project is capable of real-time vehicle detection and tracking in video files. It successfully counts the number of vehicles entering and exiting a designated area, and displays visual feedback such as bounding boxes, vehicle type, and entry/exit counts.

Acknowledgements

    YOLO (You Only Look Once): Pre-trained object detection model used for vehicle detection.
    OpenCV: Used for video processing and visualization.
    Ultralytics: Provides an implementation of the YOLO model.
