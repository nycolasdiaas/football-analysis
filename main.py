from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

from utils import process_video
# from tracking import ObjectTracker, KeypointsTracker
# from club_assignment import ClubAssigner, Club
# from ball_to_player_assignment import BallToPlayerAssigner
from annotation import FootballVideoProcessor

import cv2
import numpy as np

def main():
    # Read video
    video_frames = read_video('input_videos/troca de passes.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')
    
    # Get objects
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    
    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.teams_colors[team]

    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)
    
    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)
    
    # Save video
    save_video(output_video_frames, 'output_videos/output_videos.avi')
    
    # 6. Initialize the video processor
    # This processor will handle every task needed for analysis.
    processor = FootballVideoProcessor(obj_tracker,                                   # Created ObjectTracker object
                                       kp_tracker,                                    # Created KeypointsTracker object
                                       club_assigner,                                 # Created ClubAssigner object
                                       ball_player_assigner,                          # Created BallToPlayerAssigner object
                                       top_down_keypoints,                            # Created Top-Down keypoints numpy array
                                       field_img_path='input_videos/field_2d_v2.png', # Top-Down field image path
                                       save_tracks_dir='output_videos',               # Directory to save tracking information.
                                       draw_frame_num=True                            # Whether or not to draw current frame number on 
                                                                                      #the output video.
                                       )
    
    # 7. Process the video
    # Specify the input video path and the output video path. 
    # The batch_size determines how many frames are processed in one go.
    process_video(processor,                                # Created FootballVideoProcessor object
                  video_source='input_videos/video2.mp4', # Video source (in this case video file path)
                  output_video='output_videos/testx.mp4',    # Output video path (Optional)
                  batch_size=10                             # Number of frames to process at once
                  )
    
if __name__ == '__main__':
    main()