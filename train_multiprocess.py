import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import time

from src.wrapper import DoomEnv
from src.model import DoomCNN
from src.memory import ReplayMemory
from src.multiprocessing_env import VecEnv # Notre nouveau module

# --- HYPERPARAMÈTRES VECTORISÉS ---
LEARNING_RATE = 0.0001
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 20000       # Plus grand car on collecte plus vite
EPSILON_START = 1.0
EPSILON_END = 0.01        
EPSILON_DECAY = 0.99      
NUM_ENVS = 8              # NOMBRE DE JEUX EN PARALLÈLE (Boost CPU)
TOTAL_STEPS = 50000       # On ne compte plus en épisodes mais en "steps" globaux
SAVE_INTERVAL_STEPS = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env():
    # Fonction helper pour créer un environnement
    def _thunk():
        return DoomEnv(render=False)
    return _thunk

def train():
    print(f"🚀 ENTRAÎNEMENT VECTORISÉ ({NUM_ENVS} envs) SUR : {device}")
    
    # 1. Création du VecEnv
    envs = [make_env() for _ in range(NUM_ENVS)]
    env = VecEnv(envs)
    
    num_actions = env.action_space.n
    
    policy_net = DoomCNN((1, 84, 84), num_actions).to(device)
    target_net = DoomCNN((1, 84, 84), num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(BUFFER_SIZE)
    
    epsilon = EPSILON_START
    steps_done = 0
    
    # Initialisation de tous les environnements
    states = env.reset() # Renvoie un tableau de (8, 84, 84, 1)
    
    try:
        while steps_done < TOTAL_STEPS:
            # --- A. SÉLECTION DE L'ACTION (Batch) ---
            # On doit choisir une action pour CHACUN des 8 environnements
            actions = []
            for i in range(NUM_ENVS):
                if np.random.random() < epsilon:
                    actions.append(np.random.randint(0, num_actions))
                else:
                    # On prépare l'état i pour le GPU
                    state_tensor = torch.tensor(states[i], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                    with torch.no_grad():
                        actions.append(policy_net(state_tensor).max(1)[1].item())
            
            # --- B. EXÉCUTION PARALLÈLE ---
            next_states, rewards, dones, _ = env.step(actions)
            
            # --- C. MÉMORISATION ---
            for i in range(NUM_ENVS):
                # On stocke chaque expérience individuelle
                memory.push(states[i], actions[i], rewards[i], next_states[i] if not dones[i] else None, dones[i])
            
            states = next_states
            steps_done += NUM_ENVS # On a fait 8 pas d'un coup
            
            # --- D. APPRENTISSAGE ---
            if len(memory) >= BATCH_SIZE:
                optimize_model(policy_net, target_net, memory, optimizer)
            
            # Mise à jour Epsilon et Target Net
            if steps_done % 100 == 0: # On decay moins souvent car ça va très vite
                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
                
            if steps_done % 1000 == 0:
                target_net.load_state_dict(policy_net.state_dict())
                # Affichage du progrès
                avg_reward = np.mean(rewards) # Moyenne sur les 8 envs à l'instant T
                print(f"Steps: {steps_done}/{TOTAL_STEPS} | Avg Reward Batch: {avg_reward:.5f} | Epsilon: {epsilon:.3f}")

            # Sauvegarde
            if steps_done % SAVE_INTERVAL_STEPS == 0:
                save_path = f"models/doom_dqn_vec_{steps_done}.pth"
                torch.save(policy_net.state_dict(), save_path)
                print(f"💾 Modèle sauvegardé : {save_path}")

    except KeyboardInterrupt:
        print("Arrêt manuel.")
    finally:
        env.close()
        print("Fermeture des environnements.")

def optimize_model(policy_net, target_net, memory, optimizer):
    # ... (Exactement la même fonction que dans ton train.py précédent)
    # Copie-colle ta fonction optimize_model ici !
    # Je la remets pour être complet :
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
    state_batch = torch.cat([torch.tensor(s, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0 for s in states])
    action_batch = torch.tensor(actions, device=device).unsqueeze(1)
    reward_batch = torch.tensor(rewards, device=device)
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0 for s in next_states if s is not None])
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if __name__ == "__main__":
    train()