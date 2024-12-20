from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import pickle
import os
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width
from sort import Sort  # Importando o SORT

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = Sort()  # Substituindo ByteTrack pelo SORT
    
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
            
    def detect_frames(self, frames):
        batch_size = 100
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.10)
            detections += detections_batch
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)
        
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            
            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Prepara as detecções para o SORT
            detections_to_sort = []
            for i, (bbox, class_id) in enumerate(zip(detection_supervision.xyxy, detection_supervision.class_id)):
                detections_to_sort.append((bbox[0], bbox[1], bbox[2], bbox[3], cls_names[class_id]))  # x1, y1, x2, y2, class

            # Rastreamento usando o SORT
            track_bbs_ids = self.tracker.update(np.array(detections_to_sort))
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for track in track_bbs_ids:
                bbox = track[:4]  # x1, y1, x2, y2
                track_id = int(track[4])  # ID do rastreamento

                # Verifica a classe do objeto
                if track_id in detection_supervision.class_id:
                    cls_id = detection_supervision.class_id[track_id]
                    if cls_names[cls_id] == "player":
                        tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    elif cls_names[cls_id] == "referee":
                        tracks["referees"][frame_num][track_id] = {"bbox": bbox}
            
            # Adicionar lógica para rastreamento da bola
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
                
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
                
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        x_center, y_center = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.34 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
                
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control, player_dict, ball_position):
        # Desenha um retângulo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # Obtém o número de vezes que cada time teve o controle da bola
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        # Evita divisão por zero
        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames > 0:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames
        else:
            team_1, team_2 = 0, 0

        # Exibe a porcentagem de posse de bola
        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        # Identifica o jogador mais próximo da bola
        closest_player = None
        min_distance = float('inf')

        # Calcular a posição central da bola
        ball_center = get_center_of_bbox(ball_position)

        for track_id, player in player_dict.items():
            player_bbox = player["bbox"]
            player_center = get_center_of_bbox(player_bbox)
            distance = np.linalg.norm(np.array(ball_center) - np.array(player_center))

            if distance < min_distance:
                min_distance = distance
                closest_player = track_id

        # Adiciona informação sobre o jogador mais próximo
        if closest_player is not None:
            cv2.putText(frame, f"Closest Player: {closest_player}", (1400, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        last_ball_position = None  # Variável para armazenar a última posição da bola
        last_team_ball_control = None  # Variável para armazenar a última posse de bola

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            
            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255)) 
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0, 0, 255))
        
            # Draw Referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255), track_id)
            
            # Draw ball
            if ball_dict:  # Verifique se o dicionário da bola não está vazio
                for track_id, ball in ball_dict.items():
                    frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))
                    
                # Obtem a bbox da bola
                ball_position = ball_dict.get(1, {}).get('bbox', None)
                if ball_position is not None:
                    last_ball_position = ball_position  # Atualiza a posição da bola
                    last_team_ball_control = team_ball_control[frame_num]  # Atualiza a posse de bola
                else:
                    # Se a bola não for detectada, use a última posição conhecida
                    print(f"Bola não detectada no frame {frame_num}. Usando a última posição conhecida.")
            
            else:
                # Se a bola não foi detectada, mantenha o último valor de posse
                print(f"Bola não detectada no frame {frame_num}. Usando a última posição conhecida.")
                ball_position = last_ball_position  # Mantém a última posição
                team_ball_control[frame_num] = last_team_ball_control  # Mantém a posse de bola
                
            # Chama a função de controle da bola usando a última posição
            if last_ball_position is not None:
                frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, player_dict, last_ball_position)

            output_video_frames.append(frame)
            
        return output_video_frames
