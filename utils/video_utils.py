import cv2
import os
import glob
import queue
import threading
import tempfile
import time
import signal
import traceback
from typing import List, Tuple, Optional
import numpy as np

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(ouput_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()
    
def process_video(processor = None, video_source: str = 0, output_video: Optional[str] = "output.mp4", 
                  batch_size: int = 30, skip_seconds: int = 0) -> None:
    """
    Process a video file or stream, capturing, processing, and displaying frames.

    Args:
        processor (AbstractVideoProcessor): Object responsible for processing frames.
        video_source (str, optional): Video source (default is "0" for webcam).
        output_video (Optional[str], optional): Path to save the output video or None to skip saving.
        batch_size (int, optional): Number of frames to process at once.
        skip_seconds (int, optional): Seconds to skip at the beginning of the video.
    """
    from annotation import AbstractVideoProcessor  # Lazy import

    if processor is not None and not isinstance(processor, AbstractVideoProcessor):
        raise ValueError("The processor must be an instance of AbstractVideoProcessor.")
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_to_skip = int(skip_seconds * fps)

    # Skip the first 'frames_to_skip' frames
    for _ in range(frames_to_skip):
        cap.read()  # Simply read and discard the frames

    frame_queue = queue.Queue(maxsize=100)
    processed_queue = queue.Queue(maxsize=100)
    stop_event = threading.Event()
    
    def signal_handler(signum, frame):
        """Signal handler to initiate shutdown on interrupt."""

        print("Interrupt received, initiating shutdown...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)