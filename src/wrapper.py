import gym
from gym import spaces
import numpy as np
from vizdoom import DoomGame, Button 
import cv2
import os

class DoomEnv(gym.Env):
    def __init__(self, render=False):
        super(DoomEnv, self).__init__()
        
        self.game = DoomGame()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Configuration : Defend The Center
        config_path = os.path.join(current_dir, "..", "assets", "defend_the_center.cfg")
        scenario_path = os.path.join(current_dir, "..", "assets", "defend_the_center.wad")
        
        self.game.load_config(config_path)
        self.game.set_doom_scenario_path(scenario_path)
        
        # Fix des boutons (Indispensable)
        self.game.set_available_buttons([Button.TURN_LEFT, Button.TURN_RIGHT, Button.ATTACK])
        
        self.game.set_window_visible(render)
        self.game.init()
        
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )
        
        self.action_space = spaces.Discrete(3)
        
        # Actions : Rotation + Tir
        self.actions_list = [
            [1, 0, 0], # Tourner Gauche
            [0, 1, 0], # Tourner Droite
            [0, 0, 1]  # Tirer
        ]

    def step(self, action):
        reward = self.game.make_action(self.actions_list[action], 4)
        
        # --- REWARD SHAPING "RADAR" ---
        
        # 1. Normalisation du Kill
        if reward > 0:
            reward = 5.0  # +5.0 ! C'est la fête.
        else:
            reward = -0.01 # Le temps coûte cher, il faut se dépêcher.
            
        # 2. Le Coût de la Munition (Systématique)
        # Chaque tir coûte un peu, qu'il touche ou non.
        if action == 2:
            reward -= 0.05
            
        # ------------------------------
        
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
        img = np.moveaxis(img, 0, -1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        img = np.reshape(img, (84, 84, 1))
        return img

    def close(self):
        self.game.close()