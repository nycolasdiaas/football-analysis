import pickle

# Objeto que será salvo
data = {'chave': 'valor', 'outro': [1, 2, 3]}

# Abrindo o arquivo para escrita em modo binário
with open('stubs/track_stubs.pkl', 'rb') as file:
    data = pickle.load(file)
    
print(data)