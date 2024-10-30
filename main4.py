import os
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
import pickle

def main():
    # Inicializa o Tracker
    tracker = Tracker('models/best.pt')

    # Caminho para os vídeos segmentados
    video_folder = 'input_videos/parts'
    video_files = sorted(os.listdir(video_folder))  # Obtém a lista de vídeos e ordena
    
    all_output_frames = []  # Lista para armazenar todos os frames de saída
    file_name = None
    for video_file in video_files:
        # Lê o vídeo
        video_frames = read_video(os.path.join(video_folder, video_file))
        file_name = video_file.replace('.mp4', '')
        # Obtém os objetos rastreados
        tracks = tracker.get_object_tracks(video_frames, read_from_stub=False,
                                           stub_path='stubs/track_stubs_with_teams.pkl')
        
        # Interpola posições da bola
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        # Atribui os times aos jogadores
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
        
        # Atribui o time para cada jogador em cada frame
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.teams_colors[team]

        # Atribuição de posse de bola
        player_assigner = PlayerBallAssigner()
        team_ball_control = []
        for frame_num, player_track in enumerate(tracks['players']):
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
            
            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

        team_ball_control = np.array(team_ball_control)

        # Desenha o resultado para o vídeo atual
        output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
        
        # Armazena os frames de saída
        all_output_frames.extend(output_video_frames)  # Adiciona os frames do vídeo atual à lista total

        # Limpa a memória após processar cada vídeo
        del video_frames, tracks
        import gc
        gc.collect()  # Coleta de lixo para liberar memória

    # Salva o vídeo final
    save_video(all_output_frames, f'output_videos/final_output_{file_name}.avi')

if __name__ == '__main__':
    main()
