import torch
import torch.nn as nn
import torch.nn.functional as F

class DoomCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DoomCNN, self).__init__()
        
        c, h, w = input_shape
        
        # --- PARTIE VISION (Inchangée - Nature CNN) ---
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calcul de la taille aplatie
        self.fc_input_dim = self.feature_size(input_shape)
        
        # --- PARTIE DUELING (Le Changement) ---
        # Au lieu d'une seule tête, on en a deux.
        
        # 1. Value Stream (V) : Estime la valeur de l'état (Est-ce que je vais gagner ?)
        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1) # Une seule valeur : "Ça va bien" ou "Ça va mal"
        )
        
        # 2. Advantage Stream (A) : Estime l'intérêt de chaque action
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions) # Une valeur par action
        )

    def forward(self, x):
        # Vision
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        # Séparation en deux flux
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Recombinaison (Formule Dueling DQN)
        # Q(s,a) = V(s) + (A(s,a) - Moyenne(A))
        # Cela force le réseau à apprendre V et A séparément
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

    def feature_size(self, input_shape):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape)))).view(1, -1).size(1)