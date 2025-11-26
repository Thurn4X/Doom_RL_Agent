from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from doom_env import DoomEnv
import os

# Paramètres
CONFIG_PATH = "doom_config.cfg"
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"

# Création des dossiers
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # 1. Création de l'environnement
    print("Initialisation de l'environnement Doom...")
    env = DoomEnv(CONFIG_PATH)
    
    # Vérification que l'environnement respecte les standards Gym
    # check_env(env) # Décommenter pour debug
    
    # 2. Création du modèle PPO
    # CnnPolicy est nécessaire car nous utilisons des images en entrée
    print("Création du modèle PPO...")
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=LOG_DIR,
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=64,
        gamma=0.99
    )

    # 3. Entraînement
    # Pour le premier niveau complet, il faut BEAUCOUP de pas (ex: 1 million)
    # Pour tester que le code marche, mets 10000. Pour vrais résultats -> 1000000+
    TIMESTEPS = 100000 
    print(f"Lancement de l'entraînement pour {TIMESTEPS} steps...")
    
    model.learn(total_timesteps=TIMESTEPS)

    # 4. Sauvegarde
    model.save(f"{MODEL_DIR}/ppo_doom_first_level")
    print("Modèle sauvegardé !")
    
    env.close()

if __name__ == "__main__":
    main()