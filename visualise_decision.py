import torch
import time
import cv2
import numpy as np
from src.wrapper import DoomEnv
from src.model import DoomCNN

# --- CONFIGURATION ---
MODEL_PATH = "models/doom_dqn_vec_30000.pth" 
NUM_EPISODES = 3

# --- CHOIX DU SCENARIO POUR L'AFFICHAGE ---
# Mets True si tu testes le Scénario 2 (Tourelle), False si Scénario 1 (Basic)
SCENARIO_2_MODE = True 

if SCENARIO_2_MODE:
    action_names = ["TOURNE GAUCHE", "TOURNE DROITE", "TIRER"]
else:
    action_names = ["GAUCHE", "DROITE", "TIRER"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_bars(image, q_values, action_names):
    h, w = image.shape[:2]
    stats_width = 350 # Un peu plus large pour "TOURNE GAUCHE"
    canvas = np.zeros((h, w + stats_width, 3), dtype=np.uint8)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    canvas[:, :w] = image
    
    q_min, q_max = min(q_values), max(q_values)
    
    for i, (q, name) in enumerate(zip(q_values, action_names)):
        y = 50 + i * 60
        is_chosen = (i == np.argmax(q_values))
        
        # Couleur : Vert (Choisi), Rouge (Rejeté)
        # Nuance : Si la valeur est très basse (<0), rouge foncé
        color = (0, 255, 0) if is_chosen else (0, 0, 255)
        
        # Calcul de la barre (normalisée pour l'affichage)
        if q_max == q_min: 
            norm_val = 0
        else:
            norm_val = (q - q_min) / (q_max - q_min)
            
        bar_len = int(norm_val * 200) + 10
        
        cv2.rectangle(canvas, (w + 10, y), (w + 10 + bar_len, y + 30), color, -1)
        cv2.putText(canvas, f"{name}: {q:.5f}", (w + 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return canvas

def visualize_brain():
    print(f"Analyse du cerveau : {MODEL_PATH}")
    
    # Assure-toi que wrapper.py est configuré sur le bon scénario avant de lancer !
    env = DoomEnv(render=True)
    model = DoomCNN((1, 84, 84), env.action_space.n).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except FileNotFoundError:
        print("Fichier modèle introuvable.")
        return

    for ep in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            
            with torch.no_grad():
                q_values_tensor = model(state_tensor)
                q_values = q_values_tensor.cpu().numpy()[0]
                action = np.argmax(q_values)

            # Visualisation
            agent_view = cv2.resize(state.squeeze(), (400, 400), interpolation=cv2.INTER_NEAREST)
            final_frame = draw_bars(agent_view, q_values, action_names)
            
            cv2.imshow("Analyse des Decisions (Q-Values)", final_frame)
            cv2.waitKey(1)

            state, reward, done, _ = env.step(action)
            total_reward += reward
            time.sleep(0.05)
            
            if done:
                print(f"Score Final : {total_reward}")
                if total_reward > 0:
                    print("VICTOIRE !")
                    time.sleep(1)

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_brain()