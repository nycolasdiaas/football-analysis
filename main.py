from utils import read_video, save_video
from trackers import Tracker
import cv2

def main():
    # Read video
    video_frames = read_video('input_videos/troca de passes.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')
    
    # Get objects
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    # Save video
    save_video(output_video_frames, 'output_videos/output_videos.avi')
    
if __name__ == '__main__':
    main()