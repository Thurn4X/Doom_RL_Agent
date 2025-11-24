import re
import os
import numpy as np

def analyze_accuracy_multiprocess(log_file_path, chunk_size=10):
    if not os.path.exists(log_file_path):
        print(f"❌ Fichier introuvable : {log_file_path}")
        return

    print(f"📊 Analyse Multiprocess du fichier : {log_file_path}\n")

    rewards = []
    steps = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Extraction du Reward Batch
            match = re.search(r"Steps:\s*(\d+).*Avg Reward Batch:\s*(-?\d+\.?\d*)", line)
            if match:
                steps.append(int(match.group(1)))
                rewards.append(float(match.group(2)))
    
    total_entries = len(rewards)
    if total_entries == 0:
        print("Aucune donnée trouvée.")
        return

    # --- ANALYSE ---
    # Note : Ici "Reward" est une MOYENNE de 8 agents.
    # Si Reward > 0.5, ça veut dire que la majorité des 8 agents ont gagné.
    # C'est une estimation de la tendance, pas un compte exact épisode par épisode.
    
    positive_batches = sum(1 for r in rewards if r > 0)
    global_trend = (positive_batches / total_entries) * 100
    
    print(f"--- RÉSULTATS GLOBAUX ({total_entries} points de mesure) ---")
    print(f"Points de mesure positifs (Victoire moyenne) : {positive_batches}")
    print(f"Tendance globale de réussite : {global_trend:.2f}%\n")

    # Calcul par tranches de Steps
    print(f"--- PROGRESSION (Par tranches de logs) ---")
    
    # On découpe la liste par paquets (chunks)
    for i in range(0, total_entries, chunk_size):
        chunk_rewards = rewards[i : i + chunk_size]
        chunk_steps = steps[i : i + chunk_size]
        
        if not chunk_steps: continue
        
        start_step = chunk_steps[0]
        end_step = chunk_steps[-1]
        
        # Moyenne du score sur cette tranche
        avg_score = np.mean(chunk_rewards)
        
        # Estimation visuelle
        # Si score > 0.5 : Vert (Très bon)
        # Si score > -0.5 : Jaune (Moyen)
        # Si score < -0.5 : Rouge (Mauvais)
        if avg_score > 0.5:
            status = "🟢 EXCELLENT (Sniper)"
        elif avg_score > 0:
            status = "🟢 BON (Victoires fréquentes)"
        elif avg_score > -1.0:
            status = "🟡 MOYEN (En progrès)"
        elif avg_score > -3.0:
            status = "🟠 PACIFISTE (Ne tire pas)"
        else:
            status = "🔴 ÉCHEC (Tire à côté)"

        print(f"Steps {start_step:06d}-{end_step:06d} | Moyenne: {avg_score:5.2f} | {status}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Modifie le nom du fichier ici si besoin
    log_path = os.path.join(current_dir, "..", "training_logs", "training_logs_scenario2_entrainement2.txt")
    
    analyze_accuracy_multiprocess(log_path)