from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
import pickle

class MatchProcessor:
    def __init__(self, video_path, model_path, stub_path, output_video_path):
        """Inicializa o processador de partidas com caminhos de vídeo, modelo e saídas."""
        self.video_frames = read_video(video_path)
        self.tracker = Tracker(model_path)
        self.team_assigner = TeamAssigner()
        self.player_assigner = PlayerBallAssigner()
        self.stub_path = stub_path
        self.output_video_path = output_video_path
        self.tracks = None
        self.team_ball_control = []

    def load_tracks(self):
        """Carrega ou cria as informações de rastreamento para os objetos no vídeo."""
        self.tracks = self.tracker.get_object_tracks(
            self.video_frames,
            read_from_stub=False,
            stub_path=self.stub_path
        )
        self.tracks["ball"] = self.tracker.interpolate_ball_positions(self.tracks["ball"])

    def assign_teams(self):
        """Atribui o time para cada jogador com base na cor do uniforme."""
        initial_frame = self.video_frames[0]
        self.team_assigner.assign_team_color(initial_frame, self.tracks['players'][0])
        
        for frame_num, player_track in enumerate(self.tracks['players']):
            for player_id, track in player_track.items():
                team = self.team_assigner.get_player_team(
                    self.video_frames[frame_num], track['bbox'], player_id)
                track['team'] = team
                track['team_color'] = self.team_assigner.teams_colors[team]

    def assign_ball_possession(self):
        """Atribui a posse de bola ao jogador mais próximo em cada frame."""
        for frame_num, player_track in enumerate(self.tracks['players']):
            ball_bbox = self.tracks['ball'][frame_num][1]['bbox']
            assigned_player = self.player_assigner.assign_ball_to_player(player_track, ball_bbox)
            
            # Define posse de bola e controla a sequência de quem está com a bola
            if assigned_player != -1:
                player_track[assigned_player]['has_ball'] = True
                self.team_ball_control.append(player_track[assigned_player]['team'])
            else:
                # Mantém o último time com a posse
                self.team_ball_control.append(
                    self.team_ball_control[-1] if self.team_ball_control else 0
                )
        self.team_ball_control = np.array(self.team_ball_control)

    def save_tracks(self):
        """Salva as informações de rastreamento atualizadas em um arquivo .pkl."""
        with open(self.stub_path, 'wb') as f:
            pickle.dump(self.tracks, f)

    def annotate_and_save_video(self):
        """Desenha as anotações e salva o vídeo final com as anotações."""
        output_video_frames = self.tracker.draw_annotations(
            self.video_frames, self.tracks, self.team_ball_control)
        save_video(output_video_frames, self.output_video_path)

    def process(self):
        """Executa todas as etapas do processamento do vídeo."""
        self.load_tracks()
        self.assign_teams()
        self.assign_ball_possession()
        self.save_tracks()
        self.annotate_and_save_video()


if __name__ == '__main__':
    processor = MatchProcessor(
        video_path='input_videos/output_video_000.mp4',
        model_path='models/best.pt',
        stub_path='stubs/track_stubs_with_teams.pkl',
        output_video_path='output_videos/new_output_videos.avi'
    )
    processor.process()
