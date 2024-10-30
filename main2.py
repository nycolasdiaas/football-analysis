# main

from utils import read_video, save_video 
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
import pickle

def main():
    # Lê o vídeo
    # video_frames = read_video('input_videos/troca de passes.mp4')
    video_frames = read_video('input_videos/3_0_input.mp4')

    # Inicializa o Tracker
    tracker = Tracker('models/best.pt')
    
    # Obtém os objetos rastreados
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path='stubs/track_stubs_with_teams.pkl')
    # Interpola posições da bola
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Atribui os times aos jogadores
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    # Atribui o time para cada jogador em cada frame
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            # Obtém o time com base na cor uniforme
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.teams_colors[team]

    # Atribuição de posse de bola
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        # Define posse de bola e controla a sequência de quem está com a bola
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control = np.array(team_ball_control)
    
    # Salva o dicionário atualizado com times no arquivo .pkl
    with open('stubs/track_stubs_with_teams.pkl', 'wb') as f:
        pickle.dump(tracks, f)

    # Desenha o resultado
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    # Salva o vídeo
    save_video(output_video_frames, 'output_videos/output_videos.avi')
    
if __name__ == '__main__':
    main()
