# main.py

import numpy as np
from utils import read_video, save_video, get_settings  # Importando get_settings para obter as configurações
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from view_transformer import ViewTransformer, CameraMovementEstimator
import pickle

def main():
    # Configurações
    settings = get_settings()

    # Lê o vídeo
    video_frames = read_video(settings.INPUT_VIDEO_PATH + settings.INPUT_VIDEO_NAME)

    # Inicializa o Tracker
    tracker = Tracker('models/best.pt')
    
    # Obtém os objetos rastreados
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs_with_teams.pkl')

    # Interpola posições da bola
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Atribui os times aos jogadores
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    # Atribui o time para cada jogador em cada frame
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in enumerate(player_track):  # Assumindo que player_track é uma lista
            # Verifique se track é um dicionário e tem a chave 'bbox'
            if isinstance(track, dict) and 'bbox' in track:
                # Obtém o time com base na cor uniforme
                team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.teams_colors[team]
            else:
                print(f"Track não é um dicionário ou não tem 'bbox': {track}")

    # Atribuição de posse de bola
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        # Define posse de bola e controla a sequência de quem está com a bola
        if assigned_player != -1 and 'team' in tracks['players'][frame_num][assigned_player]:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)

    # Inicializa o estimador de movimento da câmera
    camera_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement = camera_estimator.get_camera_movement(video_frames)

    # Definindo keypoints fictícios, já que não foram definidos
    top_down_keypoints = np.array([[0, 0], [105, 0], [105, 68], [0, 68]])  # Exemplo de keypoints
    filter = np.array([True] * len(top_down_keypoints))  # Usando todos os keypoints

    # Inicializa o ViewTransformer
    view_transformer = ViewTransformer(top_down_keypoints)

    # Transforme os dados de rastreamento para a visão top-down
    transformed_tracks = []
    for player_track in tracks['players']:
        transformed_player_tracks = view_transformer.transform(player_track, None, filter)
        transformed_tracks.append(transformed_player_tracks)

    # Salva o dicionário atualizado com times no arquivo .pkl
    with open('stubs/track_stubs_with_teams.pkl', 'wb') as f:
        pickle.dump(tracks, f)

    # Desenha o resultado
    output_video_frames = tracker.draw_annotations(video_frames, transformed_tracks, team_ball_control)
    
    # Salva o vídeo
    save_video(output_video_frames, 'output_videos/new_output_videos.avi')

if __name__ == '__main__':
    main()
