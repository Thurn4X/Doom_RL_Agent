import sys
import os
import torch


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model import DoomCNN

def test_cnn():
    print("Test du modèle CNN (Cerveau)...")

    # 1. On simule une image venant de Doom 
    # Format: (Batch=1, Channel=1 (Gris), Height=84, Width=84)
    dummy_input = torch.zeros(1, 1, 84, 84)

    # 2. On crée le cerveau
    # On veut 3 actions en sortie (Gauche, Droite, Tirer)
    model = DoomCNN(input_shape=(1, 84, 84), num_actions=3)

    print("\nArchitecture du réseau :")
    print(model)

    # 3. On fait passer l'image dans le réseau (Forward pass)
    output = model(dummy_input)

    print("\nRésultat du test :")
    print(f"Entrée : Image (1, 84, 84)")
    print(f"Sortie : {output.shape}") 
    print(f"Valeurs : {output}")

    # Vérification automatique
    if output.shape == (1, 3):
        print("\n✅ SUCCÈS : Le modèle sort bien 3 valeurs pour les 3 actions.")
    else:
        print(f"\n❌ ÉCHEC : Mauvaise dimension de sortie. Attendu (1, 3), Reçu {output.shape}")

if __name__ == "__main__":
    test_cnn()