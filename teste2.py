import json
import numpy as np
import matplotlib.pyplot as plt
import imageio

# Função para desenhar o campo de futebol
def draw_pitch(ax):
    plt.xlim(0, 105)
    plt.ylim(0, 68)
    ax.set_aspect('equal')
    
    # Desenhar linhas do campo
    ax.plot([0, 0, 105, 105, 0], [0, 68, 68, 0, 0], color="black")  # Limites do campo

# Função para criar um GIF a partir do JSON
def create_gif_from_json(json_file, output_gif):
    # Carregar dados do JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Lista para armazenar os frames
    frames = []

    # Processar cada frame
    for frame in data['frames']:
        # Criar uma nova figura
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        draw_pitch(ax)

        # Desenhar jogadores
        for player_id, details in frame['players'].items():
            bbox = details['bbox']
            x_center = (bbox[0] + bbox[2]) / 2  # Centro x
            y_center = (bbox[1] + bbox[3]) / 2  # Centro y

            # Normalizar para as dimensões do campo
            x_normalized = x_center / 1280 * 105
            y_normalized = y_center / 720 * 68
            
            print(f"Jogador ID: {player_id}, Posição: ({x_normalized}, {y_normalized})")  # Debugging
            
            # Verificar se as coordenadas estão dentro dos limites
            if 0 <= x_normalized <= 105 and 0 <= y_normalized <= 68:
                color = (1, 0, 0) if details['team'] == 1 else (0, 0, 1)  # Vermelho ou Azul
                ax.scatter(x_normalized, y_normalized, color=color, s=100)

        # Desenhar a bola
        if 'ball' in frame:
            ball_bbox = frame['ball']['bbox']
            ball_x = (ball_bbox[0] + ball_bbox[2]) / 2
            ball_y = (ball_bbox[1] + ball_bbox[3]) / 2

            # Normalizar para as dimensões do campo
            ball_x_normalized = ball_x / 1280 * 105
            ball_y_normalized = ball_y / 720 * 68

            print(f"Bola Posição: ({ball_x_normalized}, {ball_y_normalized})")  # Debugging

            # Verificar se a posição da bola está dentro dos limites
            if 0 <= ball_x_normalized <= 105 and 0 <= ball_y_normalized <= 68:
                ax.scatter(ball_x_normalized, ball_y_normalized, color='yellow', s=50)

        # Salvar o frame como imagem temporária
        plt.axis('off')
        plt.savefig('temp_frame.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Adicionar a imagem à lista de frames do GIF
        frames.append(imageio.imread('temp_frame.png'))

    # Criar o GIF
    imageio.mimsave(output_gif, frames, duration=0.1)  # Ajuste a duração conforme necessário

# Caminho para o arquivo JSON e nome do arquivo de saída do GIF
json_file_path = 'output_tracking_data.json'  # Substitua pelo seu arquivo JSON
output_gif_path = 'output_video.gif'  # Nome do arquivo de saída

# Criar GIF a partir do JSON
create_gif_from_json(json_file_path, output_gif_path)

print(f"GIF criado com sucesso em {output_gif_path}!")
