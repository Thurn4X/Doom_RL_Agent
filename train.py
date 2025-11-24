import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import time

# Nos modules "maison"
from src.wrapper import DoomEnv
from src.model import DoomCNN
from src.memory import ReplayMemory

# --- HYPERPARAMÈTRES (Les réglages de l'IA) ---
LEARNING_RATE = 0.0001   # Vitesse d'apprentissage (Standard DQN)
GAMMA = 0.99              # Importance du futur (0.99 = très prévoyant)
BATCH_SIZE = 64           # Nombre d'images qu'on revoit à chaque entraînement
BUFFER_SIZE = 10000       # Taille de la mémoire
EPSILON_START = 1.0       # 100% hasard au début
EPSILON_END = 0.01        # On finit à 10% de hasard (pour explorer un peu toujours)
EPSILON_DECAY = 0.99    # Vitesse de réduction du hasard
NUM_EPISODES = 1000       # Nombre de parties à jouer
SAVE_INTERVAL = 50        # Sauvegarder le cerveau tous les 50 épisodes

# Choix automatique : Carte Graphique (cuda) ou Processeur (cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entraînement lancé sur : {device}")

def train():
    # 1. Initialisation
    env = DoomEnv(render=False) # False pour aller plus vite (pas de fenêtre)
    num_actions = env.action_space.n
    
    # Création du cerveau et de la mémoire
    policy_net = DoomCNN((1, 84, 84), num_actions).to(device) # Le réseau qui apprend
    target_net = DoomCNN((1, 84, 84), num_actions).to(device) # Une copie stable pour les maths
    target_net.load_state_dict(policy_net.state_dict())       # On les synchronise au début
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(BUFFER_SIZE)
    
    epsilon = EPSILON_START
    
    # 2. Boucle principale (Les Épisodes)
    for episode in range(NUM_EPISODES):
        state = env.reset()
        # Conversion numpy (84,84,1) -> Tensor PyTorch (1,1,84,84)
        # On passe les canaux en premier (transpose) et on ajoute la dimension de batch (unsqueeze)
        state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        
        total_reward = 0
        done = False
        
        while not done:

            # --- VERIFICATION EXPRESS (A supprimer après) ---
            if episode == 0 and total_reward == 0: # On le fait juste une fois au début
                print(f"📍 Les données sont sur : {state.device}")
                print(f"🧠 Le réseau est sur : {next(policy_net.parameters()).device}")
            # ----------------------------------------------

            # ... reste du code ...
            # --- A. SÉLECTION DE L'ACTION (Epsilon-Greedy) ---
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample() # Hasard
            else:
                with torch.no_grad():
                    # Le réseau renvoie les Q-values pour les 3 actions. On prend la plus grande (.max)
                    # t.max(1) renvoie (valeur, index). On veut l'index [1].
                    action_idx = policy_net(state).max(1)[1].item()
            
            # --- B. EXÉCUTION DANS LE JEU ---
            next_state_np, reward, done, _ = env.step(action_idx)
            total_reward += reward
            
            # Traitement du next_state comme pour le state
            if not done:
                next_state = torch.tensor(next_state_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            else:
                next_state = None # Fin de l'épisode
            
            # --- C. MÉMORISATION ---
            # On stocke tout dans la mémoire
            memory.push(state, action_idx, reward, next_state, done)
            
            state = next_state
            
            # --- D. APPRENTISSAGE (Le cœur du DQN) ---
            if len(memory) >= BATCH_SIZE:
                optimize_model(policy_net, target_net, memory, optimizer)
        
        # --- FIN DE L'ÉPISODE ---
        # Réduire epsilon (Moins de hasard, plus de cerveau)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # Mettre à jour le "Target Net" de temps en temps (stabilité)
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        print(f"Episode {episode}/{NUM_EPISODES} | Reward: {total_reward:.1f} | Epsilon: {epsilon:.3f}")
        
        # On définit les moments clés : 25%, 50%, 75%, 100% d'entrainement
        milestones = [
            int(NUM_EPISODES * 0.25), 
            int(NUM_EPISODES * 0.50), 
            int(NUM_EPISODES * 0.75), 
            NUM_EPISODES - 1
        ]

        # On sauvegarde si on est sur un milestone OU si c'est le tout début (pour avoir le modèle "nul")
        if episode == 50 or episode in milestones:
            save_path = f"models/doom_dqn_ep{episode}.pth"
            torch.save(policy_net.state_dict(), save_path)
            print(f"💾 Checkpoint sauvegardé : {save_path}")

    print("Entraînement terminé !")
    env.close()

def optimize_model(policy_net, target_net, memory, optimizer):
    """C'est ici qu'on applique les maths (Bellman Equation)"""
    # 1. On pioche 64 souvenirs au hasard
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
    
    # Conversion en Tensors groupés (Batch)
    # torch.cat permet d'empiler les images les unes sur les autres
    state_batch = torch.cat(states)
    action_batch = torch.tensor(actions, device=device).unsqueeze(1) # Forme (64, 1)
    reward_batch = torch.tensor(rewards, device=device)
    
    # Pour next_states, attention : certains sont None (si done=True)
    # On crée un masque pour gérer ça proprement
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in next_states if s is not None])
    
    # 2. Calcul de Q(s, a) : Ce que le réseau a prédit
    # gather(1, action_batch) permet de ne garder que la valeur de l'action qui a vraiment été jouée
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # 3. Calcul de V(s') : La valeur du prochain état (selon le Target Net)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        
    # 4. Calcul de la cible (Target) via Bellman : Reward + Gamma * Valeur_Futur
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # 5. Calcul de la perte (Loss) : Différence entre Prédiction et Cible
    # On utilise Huber Loss (SmoothL1Loss) car elle est plus stable que MSE pour le DQN
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # 6. Optimisation (Backpropagation)
    optimizer.zero_grad() # On efface les anciens gradients
    loss.backward()       # On calcule les nouveaux gradients
    
    # Gradient Clipping (Astuce de pro pour éviter que l'IA ne "explose" si l'erreur est trop grosse)
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    
    optimizer.step()      # On met à jour les poids du réseau

if __name__ == "__main__":
    train()