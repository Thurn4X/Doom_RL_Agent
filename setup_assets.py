import urllib.request
import os

# On définit le dossier de destination
# (Important pour que wrapper.py les trouve)
assets_dir = "assets"

# Création du dossier s'il n'existe pas
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)
    print(f"Dossier '{assets_dir}' créé.")

# Les URLs officielles
base_url = "https://github.com/mwydmuch/ViZDoom/raw/master/scenarios/"

# Liste des fichiers à récupérer (Scenario 1 et 2)
files = [
    "basic.wad", 
    "basic.cfg",
    "defend_the_center.wad", 
    "defend_the_center.cfg"
]

print("Mise à jour des fichiers du scénario...")

for filename in files:
    url = base_url + filename
    dest_path = os.path.join(assets_dir, filename)
    
    if not os.path.exists(dest_path):
        print(f"Téléchargement de {filename}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"✅ {filename} téléchargé dans {assets_dir}/")
        except Exception as e:
            print(f"❌ Erreur sur {filename} : {e}")
    else:
        print(f"ℹ️ {filename} existe déjà.")

print("\nTout est prêt ! Tu peux lancer l'entraînement.")