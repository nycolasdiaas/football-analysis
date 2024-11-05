from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.teams_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=5, random_state=42)
        kmeans.fit(image_2d)
        return kmeans
    
    def get_player_color(self, frame, bbox):
        try:
            image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            top_half_image = image[: int(image.shape[0] / 2), :]
            kmeans = self.get_clustering_model(top_half_image)
            
            labels = kmeans.labels_
            clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
            corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
            non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
            player_cluster = 1 - non_player_cluster
            player_color = kmeans.cluster_centers_[player_cluster]
            
            return player_color
        except Exception as e:
            print(f"Erro ao obter cor do jogador: {e}")
            return np.array([0, 0, 0])  # Retorna uma cor padrão em caso de falha
    
    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_info in player_detections.items():
            bbox = player_info['bbox']
            player_color = self.get_player_color(frame, bbox)
            if player_color is not None:
                player_colors.append(player_color)
        
        if len(player_colors) >= 2:
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
            kmeans.fit(player_colors)
            self.kmeans = kmeans
            self.teams_colors[1] = kmeans.cluster_centers_[0]
            self.teams_colors[2] = kmeans.cluster_centers_[1]
        else:
            print("Aviso: cores de times insuficientes para criar modelo KMeans.")

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        if self.kmeans is None or player_color is None:
            print("Aviso: modelo KMeans não inicializado ou cor do jogador não detectada.")
            return 0  # Retorna um valor padrão para indicar que o time não foi determinado
        
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1
        self.player_team_dict[player_id] = team_id
        return team_id
