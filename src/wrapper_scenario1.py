import gym
from gym import spaces
import numpy as np
from vizdoom import DoomGame
import cv2 # OpenCV pour redimensionner l'image
import os

class DoomEnv(gym.Env):
    def __init__(self, render=False):
        super(DoomEnv, self).__init__()
        
        # 1. Démarrage du moteur Doom
        self.game = DoomGame()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- CONFIGURATION (Scenario 1: Basic) ---
        config_path = os.path.join(current_dir, "..", "assets", "basic.cfg")
        scenario_path = os.path.join(current_dir, "..", "assets", "basic.wad")
        
        self.game.load_config(config_path)
        self.game.set_doom_scenario_path(scenario_path)

        # On affiche la fenêtre seulement si on le demande (pour aller vite en entraînement)
        self.game.set_window_visible(render)
        self.game.init()
        
        # 2. Définition de l'espace d'observation (Ce que voit l'IA)
        # On va réduire l'image à 84x84 pixels en Noir & Blanc (1 canal)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )
        
        # 3. Définition de l'espace d'action
        # Dans Basic, on a 3 actions : [Gauche, Droite, Tirer]
        self.action_space = spaces.Discrete(3)
        
        # Matrice d'identité pour convertir le chiffre de l'IA en boutons Doom
        # 0 -> [1, 0, 0] (Gauche)
        # 1 -> [0, 1, 0] (Droite)
        # 2 -> [0, 0, 1] (Tirer)
        self.actions_list = np.identity(3, dtype=int).tolist()

    def step(self, action):
        # L'IA donne un chiffre (ex: 2), on le convertit en boutons [0, 0, 1]
        # On répète l'action pendant 4 frames (Frame Skipping)
        reward = self.game.make_action(self.actions_list[action], 4)
                
        # 1. Normalisation : On ramène +100 à +1.0 pour la stabilité du réseau
        reward = reward / 100.0
        
        reward = self.game.make_action(self.actions_list[action], 4)
        
        # --- REWARD SHAPING ULTIME ---
        
        # 1. Le temps ne doit pas stresser l'agent
        if reward > 50: # C'est un KILL (+100 dans le jeu)
            reward = 1.0 
        else:
            # C'est juste du temps qui passe (-1 dans le jeu)
            reward = -0.001 # Très faible pénalité d'existence
            
        # 2. Pénalité de tir raté "Douce mais ferme"
        if action == 2 and reward < 0.5: # Si je tire et je ne tue pas
            reward -= 0.1
            
        # Récupérer le nouvel état
        done = self.game.is_episode_finished()
        
        if done:
            state = np.zeros((84, 84, 1), dtype=np.uint8)
        else:
            state = self.game.get_state().screen_buffer
            state = self.preprocess(state)
            
        info = {}
        return state, reward, done, info

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.preprocess(state)

    def preprocess(self, img):
        # Transformation de l'image pour l'IA
        img = np.moveaxis(img, 0, -1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        img = np.reshape(img, (84, 84, 1))
        return img

    def close(self):
        self.game.close()