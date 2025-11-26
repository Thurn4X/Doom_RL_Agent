import gymnasium as gym
from gymnasium import spaces
import vizdoom as vzd
import numpy as np
import cv2

class DoomEnv(gym.Env):
    def __init__(self, config_file_path):
        super(DoomEnv, self).__init__()
        
        # Initialisation de ViZDoom
        self.game = vzd.DoomGame()
        self.game.load_config(config_file_path)
        self.game.set_window_visible(False) # Mettre à True pour voir l'entraînement (plus lent)
        self.game.init()
        
        # Définition de l'espace d'observation (Image)
        # On va réduire l'image en 100x60 Grayscale pour simplifier le calcul
        self.observation_shape = (60, 100, 1) 
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.observation_shape, dtype=np.uint8
        )
        
        # Définition de l'espace d'action
        # Nous avons 4 boutons dans le config, mais on peut combiner des touches
        # Ici on utilise un index simple pour chaque bouton
        self.action_space = spaces.Discrete(self.game.get_available_buttons_size())
        
        # Variables pour le calcul de la récompense
        self.last_health = 100
        self.last_killcount = 0

    def step(self, action):
        # Créer un vecteur d'actions booléennes (ex: [0, 0, 1, 0])
        actions = np.zeros(self.game.get_available_buttons_size())
        actions[action] = 1
        
        # Effectuer l'action et avancer de 4 frames (frame skipping pour accélérer)
        reward = self.game.make_action(list(actions), 4)
        
        # Récupérer l'état
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        
        info = {}
        img = np.zeros(self.observation_shape, dtype=np.uint8)

        if state:
            # Traitement de l'image (Screen buffer)
            screen = state.screen_buffer
            # Conversion en niveau de gris et redimensionnement
            img = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (100, 60))
            img = np.expand_dims(img, axis=-1)

            # --- Shaping de la récompense (Personnalisation) ---
            # ViZDoom donne une récompense de base, mais on peut l'améliorer
            game_vars = state.game_variables
            health = game_vars[0]
            killcount = game_vars[1]
            
            # Bonus pour tuer des monstres
            if killcount > self.last_killcount:
                reward += 50
            
            # Pénalité pour perte de vie
            if health < self.last_health:
                reward -= 10
                
            self.last_health = health
            self.last_killcount = killcount

        return img, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        self.last_health = 100
        self.last_killcount = 0
        
        state = self.game.get_state()
        screen = state.screen_buffer
        img = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (100, 60))
        img = np.expand_dims(img, axis=-1)
        
        return img, {}

    def close(self):
        self.game.close()