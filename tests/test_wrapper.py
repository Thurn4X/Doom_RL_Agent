import sys
import os
import time


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.wrapper import DoomEnv 

# Création de l'environnement custom
env = DoomEnv(render=True)

print("Test de l'environnement...")
state = env.reset()

# Vérification technique
print(f"Forme de l'image renvoyée : {state.shape}")

for i in range(10):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    
    print(f"Step {i}: Action={action}, Reward={reward}, Done={done}")
    time.sleep(0.2)
    
    if done:
        env.reset()

env.close()