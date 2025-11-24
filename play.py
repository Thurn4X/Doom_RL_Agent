import torch
import time
import numpy as np
from src.wrapper import DoomEnv
from src.model import DoomCNN

# --- CONFIGURATION ---
# Change ce chemin pour tester différents cerveaux !
MODEL_PATH = "models/doom_dqn_vec_50000.pth" 
NUM_EPISODES = 10

# Détection automatique (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def play():
    print(f"Chargement du modèle : {MODEL_PATH}")
    
    # 1. On lance le jeu avec l'écran visible
    env = DoomEnv(render=True)
    
    # 2. On recrée l'architecture du cerveau (vide)
    num_actions = env.action_space.n
    model = DoomCNN((1, 84, 84), num_actions).to(device)
    
    # 3. On charge les poids entraînés à l'intérieur
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval() # Mode évaluation (fige les couches comme le Dropout s'il y en avait)
        print("Modèle chargé avec succès !")
    except FileNotFoundError:
        print(f"Erreur : Le fichier {MODEL_PATH} n'existe pas.")
        return

    # 4. Boucle de jeu
    for i in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        
        print(f"\n--- Épisode {i+1} ---")
        
        while not done:
            # Préparation de l'image pour PyTorch (exactement comme dans train.py)
            state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            
            # Décision de l'IA (Sans Epsilon ! On prend juste le max)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.max(1)[1].item()
            
            # Action dans le jeu
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Ralenti pour que tu aies le temps de savourer (0.05s)
            # Enlève ça si tu veux voir la vitesse réelle de l'IA
            time.sleep(0.05)
            
        print(f"Score final : {total_reward}")
        time.sleep(1.0)

    env.close()

if __name__ == "__main__":
    play()