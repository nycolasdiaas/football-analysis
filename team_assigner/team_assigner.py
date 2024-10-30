from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.teams_colors = {}
        self.player_team_dict = {}
        self.kmeans = None
        
    def get_clustering_model(self, image):
        # Redimensiona a imagem para um array 2D
        image_2d = image.reshape(-1, 3)
        
        # Realiza o K-means com 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=5, random_state=42)
        kmeans.fit(image_2d)

        return kmeans
    
    def get_player_color(self, frame, bbox):
        # Extrai a imagem da região de bbox
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # Pega apenas a metade superior do jogador para evitar cores de chuteiras e gramado
        top_half_image = image[: int(image.shape[0] / 2), :]
        
        # Obtém o modelo de clustering para a cor do jogador
        kmeans = self.get_clustering_model(top_half_image)

        # Determina a cor do jogador com base no cluster dominante (excluindo bordas)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        
        # Determina qual cluster é o do jogador e qual é o fundo (gramado)
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        
        player_color = kmeans.cluster_centers_[player_cluster]
        
        return player_color
    
    def assign_team_color(self, frame, player_detections):
        # Extrai as cores médias de cada jogador na imagem inicial
        player_colors = []
        for _, player_info in player_detections.items():
            bbox = player_info['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        # Usa K-means para distinguir as cores dos times com base nos jogadores
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(player_colors)
        
        # Guarda as cores dos times
        self.kmeans = kmeans
        self.teams_colors[1] = kmeans.cluster_centers_[0]
        self.teams_colors[2] = kmeans.cluster_centers_[1]
    
    def get_player_team(self, frame, player_bbox, player_id):
        # Verifica se o time do jogador já foi atribuído
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        # Obtém a cor do jogador e atribui ao time correspondente
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1
        
        # Armazena o time para o jogador e retorna
        self.player_team_dict[player_id] = team_id
        
        return team_id
