from vizdoom import DoomGame
import time
import random
import os  # <--- Très important

def test_environment():
    print("Initialisation du jeu (Test Brut)...")
    game = DoomGame()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "..", "assets", "basic.cfg")
    scenario_path = os.path.join(current_dir, "..", "assets", "basic.wad")
    


    try:
        game.load_config(config_path)
        game.set_doom_scenario_path(scenario_path)
        game.set_window_visible(True)
        game.init()
    except Exception as e:
        print(f"ERREUR: Impossible de lancer Doom.\n{e}")
        return

    actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    print("Lancement de 3 épisodes...")
    for i in range(3):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            action = random.choice(actions)
            reward = game.make_action(action)
            time.sleep(0.02)
        
        print(f"Episode {i+1} terminé.")
        time.sleep(0.5)

    game.close()
    print("Test réussi !")

if __name__ == "__main__":
    test_environment()