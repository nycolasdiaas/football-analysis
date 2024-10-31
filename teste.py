import pickle
import json

def create_json_from_tracking(pkl_file, output_json):
    # Carregar dados do arquivo .pkl
    with open(pkl_file, 'rb') as f:
        tracks = pickle.load(f)

    frames_data = []

    # Iterar sobre os frames no tracking
    for frame in tracks['players']:
        frame_data = {
            "players": {},
            "ball": {}
        }
        
        # Adicionar jogadores
        for player_id, details in frame.items():
            if 'bbox' in details and 'team' in details:
                # Converter np.int64 para int
                player_id_int = int(player_id)  # Convertendo a chave
                # Garantir que o bbox é uma lista de int
                bbox = [int(coord) for coord in details['bbox']]
                frame_data['players'][player_id_int] = {
                    "bbox": bbox,
                    "team": int(details['team'])  # Converter o time para int
                }
        
        # Adicionar bola
        if 'ball' in tracks and len(tracks['ball']) > 0:
            ball_data = tracks['ball'][0]  # Assumindo que a bola é a mesma para todos os frames
            if 1 in ball_data:
                frame_data['ball'] = {
                    "bbox": [int(coord) for coord in ball_data[1]['bbox']]
                }
        
        frames_data.append(frame_data)

    # Criar o JSON
    json_data = {
        "frames": frames_data
    }

    # Salvar o JSON em um arquivo
    with open(output_json, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

# Caminho para o arquivo .pkl e nome do arquivo JSON de saída
pkl_file_path = 'stubs/track_stubs_with_teams.pkl'  # Caminho do seu arquivo .pkl
output_json_path = 'output_tracking_data.json'  # Nome do arquivo JSON de saída

# Criar JSON a partir do tracking
create_json_from_tracking(pkl_file_path, output_json_path)

print(f"Dados salvos em {output_json_path} com sucesso!")
