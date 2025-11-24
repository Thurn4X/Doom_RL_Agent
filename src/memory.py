import random
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        """
        Initialise la mémoire avec une taille maximale.
        Si la mémoire est pleine, les souvenirs les plus anciens sont supprimés.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Enregistre une transition dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Récupère un lot (batch) aléatoire de souvenirs pour l'entraînement.
        """
        batch = random.sample(self.memory, batch_size)
        # MAGIE PYTHON 'zip(*batch)' transpose la liste de tuples en un tuple de listes, le format dont PyTorch a besoin.
        state, action, reward, next_state, done = zip(*batch)
        
        return state, action, reward, next_state, done

    def __len__(self):
        """Renvoie le nombre d'éléments actuels dans la mémoire"""
        return len(self.memory)