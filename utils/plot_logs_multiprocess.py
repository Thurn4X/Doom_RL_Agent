import matplotlib.pyplot as plt
import re
import numpy as np
import os

def plot_learning_curve_multiprocess(log_file_path, window_size=10):
    steps = []
    rewards = []
    epsilons = []
    
    # 1. Lecture et Parsing
    if not os.path.exists(log_file_path):
        print(f"❌ Erreur : Le fichier {log_file_path} n'existe pas.")
        return

    print(f"Lecture des logs multiprocess : {log_file_path}...")
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Regex adaptée au format vectorisé :
            # "Steps: 1000/50000 | Avg Reward Batch: -0.17 | Epsilon: 0.951"
            match = re.search(r"Steps:\s*(\d+)/\d+\s*\|\s*Avg Reward Batch:\s*(-?\d+\.?\d*)\s*\|\s*Epsilon:\s*(\d+\.?\d*)", line)
            
            if match:
                steps.append(int(match.group(1)))
                rewards.append(float(match.group(2)))
                epsilons.append(float(match.group(3)))
    
    if not steps:
        print("⚠️ Aucune donnée trouvée ! Vérifie que le fichier contient bien des lignes 'Avg Reward Batch'.")
        return

    # 2. Lissage (Moyenne Mobile)
    def moving_average(data, window):
        if len(data) < window: return data
        return np.convolve(data, np.ones(window), 'valid') / window

    smoothed_rewards = moving_average(rewards, window_size)
    # On ajuste les steps pour qu'ils aient la même taille que la courbe lissée
    # (La convolution "mange" le début de la liste)
    adjusted_steps = steps[len(steps) - len(smoothed_rewards):]
    
    # 3. Création du Graphique
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Axe 1 (Gauche) : Reward
    color = 'tab:blue'
    ax1.set_xlabel('Global Steps (Pas d\'entraînement)')
    ax1.set_ylabel('Récompense Moyenne (sur 8 agents)', color=color)
    
    # On affiche les points bruts en très transparent pour voir la variance
    ax1.plot(steps, rewards, color=color, alpha=0.15, label='Brut')
    # On affiche la moyenne mobile en solide
    ax1.plot(adjusted_steps, smoothed_rewards, color=color, linewidth=2, label=f'Moyenne mobile ({window_size} pts)')
    
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Axe 2 (Droite) : Epsilon
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Epsilon (Exploration)', color=color)
    ax2.plot(steps, epsilons, color=color, linestyle='--', alpha=0.7, label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('Scénario 2 Expérience 5')
    fig.tight_layout()
    
    # Sauvegarde
    output_file = "learning_curve_multiprocess.png"
    plt.savefig(output_file)
    print(f"✅ Graphique généré : {output_file}")
    plt.show()

if __name__ == "__main__":
    # Adapte le chemin selon où tu mets ton fichier log
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(current_dir, "..", "training_logs", "training_logs_scenario2_entrainement5.txt")
    
    plot_learning_curve_multiprocess(log_path)