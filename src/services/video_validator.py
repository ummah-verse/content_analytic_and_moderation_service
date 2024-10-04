import numpy as np
import io
import cv2

def extract_frames(video_path, interval=1):
    """
    Extract frames from video at every given second (interval).
    """
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    frames = []
    success, frame = video_capture.read()
    count = 0
    while success:
        # Ambil frame pada setiap interval detik
        if count % int(fps * interval) == 0:
            frames.append(frame)
        success, frame = video_capture.read()
        count += 1
    video_capture.release()
    return frames