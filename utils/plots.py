import matplotlib.pyplot as plt
import re
import numpy as np
import os

def plot_learning_curve(log_file_path, window_size=50):
    rewards = []
    epsilons = []
    
    # 1. Lecture et Parsing du fichier texte
    if not os.path.exists(log_file_path):
        print(f"Erreur : Le fichier {log_file_path} n'existe pas.")
        return

    print("Lecture des logs...")
    with open(log_file_path, 'r') as f:
        for line in f:
            # On cherche le motif "Reward: [nombre]" avec une Regex
            # Cela permet de trouver le chiffre même s'il y a du texte autour
            reward_match = re.search(r"Reward:\s*(-?\d+\.?\d*)", line)
            epsilon_match = re.search(r"Epsilon:\s*(\d+\.?\d*)", line)
            
            if reward_match and epsilon_match:
                rewards.append(float(reward_match.group(1)))
                epsilons.append(float(epsilon_match.group(1)))
    
    if not rewards:
        print("Aucune donnée trouvée ! Vérifie le format de ton fichier texte.")
        return

    # 2. Calcul de la Moyenne Mobile (Smoothing)
    # C'est ça qui rend la courbe "propre"
    def moving_average(data, window):
        return np.convolve(data, np.ones(window), 'valid') / window

    # On coupe les X premiers points car la moyenne mobile a besoin de données
    smoothed_rewards = moving_average(rewards, window_size)
    
    # 3. Création du Graphique
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Axe 1 (Gauche) : Le Reward
    color = 'tab:blue'
    ax1.set_xlabel('Épisodes')
    ax1.set_ylabel('Récompense Moyenne', color=color)
    ax1.plot(smoothed_rewards, color=color, linewidth=2, label=f'Moyenne mobile ({window_size} ep)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Axe 2 (Droite) : Epsilon
    # On superpose Epsilon pour montrer que "Moins de hasard = Meilleur score"
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Epsilon (Exploration)', color=color)
    # On doit ajuster la taille car 'smoothed_rewards' est plus court que 'epsilons'
    ax2.plot(epsilons[window_size-1:], color=color, linestyle='--', alpha=0.5, label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Apprentissage de l\'Agent Doom (DQN)')
    fig.tight_layout()
    
    # Sauvegarde
    output_file = "learning_curve.png"
    plt.savefig(output_file)
    print(f"✅ Graphique généré : {output_file}")
    plt.show()

if __name__ == "__main__":
    # On suppose que le fichier log est à la racine
    # Il faut remonter de utils/ vers la racine
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(current_dir, "..", "training_logs", "training_logs_entrainement3.txt")
    
    plot_learning_curve(log_path)