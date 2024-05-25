!pip install pandas
import numpy as np
import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.inspector import show
import dimod
import dwave.inspector as ins

# Configuración del solver
solver_type = 'Advantage2_prototype2.3'

# Tu token de D-Wave
dwave_token = "DEV-951435db241ce06d7f8f8ce960e5c44c5475f72d"

# Inicialización del solver con la configuración especificada
solver = DWaveSampler(solver=solver_type, token=dwave_token)

# Parámetros del modelo
params = {
    'J': 1.0,  # Interacción entre piedras adyacentes.
    'mu': 0.5,  # Factor de libertades
    'lambda_5': 1.0,  # Teorema 5
    'lambda_7': 1.0,  # Teorema 7
    'lambda_9': 1.0,  # Teorema 9
    'lambda_11': 1.0,  # Teorema 11
    'lambda_12': 1.0,  # Teorema 12
    'lambda_15': 1.0,  # Teorema 15
    'lambda_16': 1.0,  # Teorema 16
    'lambda_17': 1.0,  # Teorema 17
    'size': 9,  # Tamaño del tablero.
}

# Definir el estado inicial del tablero para un tablero de 9x9.
board = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, -1, 0],
    [0, 1, 0, -1, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Funciones auxiliares para calcular libertades y componentes conexas
def calculate_liberties(board, x, y):
    liberties = 0
    if x > 0 and board[x-1, y] == 0: liberties += 1
    if x < params['size']-1 and board[x+1, y] == 0: liberties += 1
    if y > 0 and board[x, y-1] == 0: liberties += 1
    if y < params['size']-1 and board[x, y+1] == 0: liberties += 1
    return liberties

def calculate_shared_liberties(board, x, y):
    shared_liberties = 0
    if x > 0 and board[x-1, y] == 0: shared_liberties += 1
    if x < params['size']-1 and board[x+1, y] == 0: shared_liberties += 1
    if y > 0 and board[x, y-1] == 0: shared_liberties += 1
    if y < params['size']-1 and board[x, y+1] == 0: shared_liberties += 1
    return shared_liberties

def dfs(board, x, y, visited, group):
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if (cx, cy) not in visited:
            visited.add((cx, cy))
            group.append((cx, cy))
            for nx, ny in [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]:
                if 0 <= nx < params['size'] and 0 <= ny < params['size'] and board[nx, ny] == board[x, y] and (nx, ny) not in visited:
                    stack.append((nx, ny))
    return group

def calculate_connected_components(board, x, y):
    visited = set()
    components = []
    for i in range(params['size']):
        for j in range(params['size']):
            if board[i, j] == board[x, y] and (i, j) not in visited:
                group = []
                group = dfs(board, i, j, visited, group)
                components.append(group)
    return components

def calculate_exterior_liberties(board, components):
    liberties = 0
    for group in components:
        for x, y in group:
            liberties += calculate_liberties(board, x, y)
    return liberties

def dfs_eyes(board, x, y, visited, color):
    stack = [(x, y)]
    eye_group = []
    while stack:
        cx, cy = stack.pop()
        if (cx, cy) not in visited:
            visited.add((cx, cy))
            eye_group.append((cx, cy))
            for nx, ny in [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]:
                if 0 <= nx < params['size'] and 0 <= ny < params['size']:
                    if board[nx, ny] == 0 and (nx, ny) not in visited:
                        stack.append((nx, ny))
                    elif board[nx, ny] == color:
                        continue
                    else:
                        return [], False
    return eye_group, True

def calculate_two_eyes(board, x, y):
    visited = set()
    color = board[x, y]
    eye_count = 0

    for i in range(params['size']):
        for j in range(params['size']):
            if board[i, j] == 0 and (i, j) not in visited:
                eye_group, is_eye = dfs_eyes(board, i, j, visited, color)
                if is_eye:
                    eye_count += 1
                if eye_count >= 2:
                    return True
    return False

def dfs_network(board, x, y, visited, color):
    stack = [(x, y)]
    network_group = []
    while stack:
        cx, cy = stack.pop()
        if (cx, cy) not in visited:
            visited.add((cx, cy))
            network_group.append((cx, cy))
            for nx, ny in [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]:
                if 0 <= nx < params['size'] and 0 <= ny < params['size']:
                    if board[nx, ny] == color and (nx, ny) not in visited:
                        stack.append((nx, ny))
                    elif board[nx, ny] != color and board[nx, ny] != 0:
                        return False
    return True

def calculate_network(board, x, y):
    visited = set()
    color = board[x, y]

    for i in range(params['size']):
        for j in range(params['size']): 
            if board[i, j] == color and (i, j) not in visited:
                if not dfs_network(board, i, j, visited, color):
                    return False
    return True

def dfs_semeai(board, x, y, visited, color):
    stack = [(x, y)]
    group = []
    liberties = set()
    while stack:
        cx, cy = stack.pop()
        if (cx, cy) not in visited:
            visited.add((cx, cy))
            group.append((cx, cy))
            for nx, ny in [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]:
                if 0 <= nx < params['size'] and 0 <= ny < params['size']:
                    if board[nx, ny] == color and (nx, ny) not in visited:
                        stack.append((nx, ny))
                    elif board[nx, ny] == 0:
                        liberties.add((nx, ny))
    return group, liberties

def calculate_semeai(board, x, y):
    visited = set()
    color = board[x, y]
    opponent_color = -color

    my_group, my_liberties = dfs_semeai(board, x, y, visited, color)
    opponent_groups = []
    for lib in my_liberties:
        ox, oy = lib
        if board[ox, oy] == opponent_color and (ox, oy) not in visited:
            opp_group, opp_liberties = dfs_semeai(board, ox, oy, visited, opponent_color)
            opponent_groups.append((opp_group, opp_liberties))

    for opp_group, opp_liberties in opponent_groups:
        if len(opp_liberties) <= len(my_liberties):
            return True
    return False

# Calcular el Hamiltoniano total
def hamiltonian_total(board):
    H = 0
    for x in range(params['size']):
        for y in range(params['size']):
            if board[x, y] != 0:
                # Término atómico
                if x > 0: H -= params['J'] * board[x, y] * board[x-1, y]
                if x < params['size']-1: H -= params['J'] * board[x, y] * board[x+1, y]
                if y > 0: H -= params['J'] * board[x, y] * board[x, y-1]
                if y < params['size']-1: H -= params['J'] * board[x, y] * board[x, y+1]
                
                # Libertades
                liberties = calculate_liberties(board, x, y)
                H -= params['mu'] * liberties * board[x, y]
                
                # Teorema 5: Formación de Ojo
                if calculate_two_eyes(board, x, y):
                    H -= params['lambda_5']
                
                # Teorema 7: Red
                if calculate_network(board, x, y):
                    H += params['lambda_7']
                
                # Teorema 9: Semeai
                if calculate_semeai(board, x, y):
                    H += params['lambda_9']
                
                # Teorema 11: Conectividad de Grupos
                components = calculate_connected_components(board, x, y)
                if len(components) >= 2:
                    H -= params['lambda_11']
                
                # Teorema 12: Libertades Conjuntas
                shared_liberties = calculate_shared_liberties(board, x, y)
                H -= params['lambda_12'] * shared_liberties
                
                # Teorema 15: Expansión de Territorio
                if liberties > 0:
                    H -= params['lambda_15']
                
                # Teorema 16: Reducción Estratégica
                if liberties < 3:
                    H += params['lambda_16']
                
                # Teorema 17: Confinamiento de Libertades
                exterior_liberties = calculate_exterior_liberties(board, components)
                if exterior_liberties == 0:
                    H += params['lambda_17']
    return H

# Crear variables de Spin para cada posición en el tablero
spins = {(i, j): i*params['size']+j for i in range(params['size']) for j in range(params['size'])}

# Configurar h_values de acuerdo con el estado inicial del tablero
h_values = {spins[(i, j)]: board[i, j] for i in range(params['size']) for j in range(params['size'])}

# Construir el Hamiltoniano Ising basado en el Hamiltoniano total calculado
H = hamiltonian_total(board)

# Crear el modelo binario
model = dimod.BinaryQuadraticModel(h_values, {}, H, dimod.SPIN)

# Crear un sampler
sampler = EmbeddingComposite(solver)
sampler.parameters['verbose'] = True

# Muestrear el modelo
response = sampler.sample(model)

# Convertir la respuesta a un DataFrame de Pandas
df = pd.DataFrame(response.data())

# Imprimir todo el DataFrame
print(df)

# Imprimir la solución con la menor energía
print(response.first)

# Mostrar el inspector
ins.show(response)

# Get the first sample
first_sample = response.first.sample

# Map the spins back to their positions on the board
board_state = np.zeros((params['size'], params['size']))
for (i, j), spin in spins.items():
    board_state[i, j] = first_sample[spin]

print(board_state)

