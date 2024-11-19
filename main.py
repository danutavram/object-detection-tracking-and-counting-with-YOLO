# main.py
from main_tracker import real_time_detection

def main():
    video_path = "video0.mp4"
    output_path = "output_video10.mp4"
    real_time_detection(video_path, output_path)

if __name__ == "__main__":
    main()
