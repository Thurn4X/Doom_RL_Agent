import torch
import time
import numpy as np
import glob  # Pour scanner les dossiers
import os
import re    # Pour extraire le numéro de l'épisode dans le nom du fichier

from src.wrapper import DoomEnv
from src.model import DoomCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sorted_checkpoints():
    """
    Récupère tous les fichiers .pth dans models/ et les trie par numéro d'épisode.
    Exemple : doom_dqn_ep50.pth avant doom_dqn_ep125.pth
    """
    # 1. On récupère la liste brute des fichiers
    # On suppose que le script est lancé depuis la racine Doom_RL
    files = glob.glob("models/*.pth")
    
    if not files:
        print("⚠️ Aucun modèle trouvé dans le dossier 'models/' !")
        return []

    # 2. Fonction pour extraire le numéro "50" de "doom_dqn_ep50.pth"
    def extract_episode_num(filename):
        # On cherche "ep" suivi de chiffres
        match = re.search(r"ep(\d+)", filename)
        if match:
            return int(match.group(1))
        return 0 # Si pas de numéro, on le met au début

    # 3. On trie la liste en utilisant cette fonction clé
    sorted_files = sorted(files, key=extract_episode_num)
    
    print(f"📂 Modèles trouvés (Triés) : {[os.path.basename(f) for f in sorted_files]}")
    return sorted_files

def watch_evolution():
    # Récupération automatique
    checkpoints = get_sorted_checkpoints()
    if not checkpoints:
        return

    print("🎬 Lancement de la séquence d'évolution...")
    
    # On ouvre la fenêtre Doom UNE SEULE FOIS
    env = DoomEnv(render=True)
    num_actions = env.action_space.n
    
    model = DoomCNN((1, 84, 84), num_actions).to(device)
    
    for model_path in checkpoints:
        print(f"\n--- CHARGEMENT DU CERVEAU : {model_path} ---")
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        except Exception as e:
            print(f"⚠️ Erreur lecture fichier : {model_path} ({e})")
            continue
            
        # On joue UN épisode
        state = env.reset()
        done = False
        total_reward = 0
        
        time.sleep(1.0) # Titre
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            
            with torch.no_grad():
                # Mode Greedy (Max) - On teste la performance pure
                action = model(state_tensor).max(1)[1].item()
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
            time.sleep(0.04) # Vitesse de visionnage

            # --- EFFET FREEZE FRAME ---
            if done:
                if total_reward > 0: 
                    print(f"🎯 BOOM ! (+{total_reward})")
                    time.sleep(2.0) 
                else:
                    print(f"💀 Raté ({total_reward}).")
                    time.sleep(0.5)
            # --------------------------
            
        print(f"🏁 Résultat : {total_reward}")
        time.sleep(1.0)

    print("\nSéquence terminée !")
    env.close()

if __name__ == "__main__":
    watch_evolution()