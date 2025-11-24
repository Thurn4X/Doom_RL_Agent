import re
import os

def analyze_accuracy(log_file_path, chunk_size=100):
    if not os.path.exists(log_file_path):
        print(f"❌ Fichier introuvable : {log_file_path}")
        return

    print(f"📊 Analyse du fichier : {log_file_path}\n")

    rewards = []
    
    # 1. Extraction des données
    with open(log_file_path, 'r') as f:
        for line in f:
            # Regex pour attraper le chiffre après "Reward:"
            match = re.search(r"Reward:\s*(-?\d+\.?\d*)", line)
            if match:
                rewards.append(float(match.group(1)))
    
    total_episodes = len(rewards)
    if total_episodes == 0:
        print("Aucune donnée trouvée.")
        return

    # 2. Calcul Global
    # On considère Victoire si Reward > 0
    total_wins = sum(1 for r in rewards if r > 0)
    global_accuracy = (total_wins / total_episodes) * 100
    
    print(f"--- RÉSULTATS GLOBAUX ({total_episodes} épisodes) ---")
    print(f"🏆 Victoires totales : {total_wins}")
    print(f"📉 Défaites totales : {total_episodes - total_wins}")
    print(f"✅ Précision Globale : {global_accuracy:.2f}%\n")

    # 3. Calcul par Tranches (pour montrer la progression)
    print(f"--- PROGRESSION (Par tranches de {chunk_size} épisodes) ---")
    
    for i in range(0, total_episodes, chunk_size):
        chunk = rewards[i : i + chunk_size]
        wins_in_chunk = sum(1 for r in chunk if r > 0)
        acc_in_chunk = (wins_in_chunk / len(chunk)) * 100
        
        start = i
        end = min(i + chunk_size, total_episodes)
        
        # Barre de progression visuelle
        bar_len = int(acc_in_chunk / 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        
        print(f"Episodes {start:03d}-{end:03d} : {bar} {acc_in_chunk:6.2f}% de réussite")

if __name__ == "__main__":
    # Chemin vers ton fichier de logs (à la racine)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(current_dir, "..", "training_logs","training_logs_entrainement4.txt")
    
    analyze_accuracy(log_path)