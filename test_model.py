from agent import DQNAgent
from tetris import Tetris

model_name = "best_model.keras"
tetris = Tetris()

agent = DQNAgent(
        discount = 0.98,
        state_size = 4,
        batch_size = 512, # 128
        buffer_size = 1000,
        epsilon = 0,
        epsilon_min = 0.01,
        n_neurons = [32, 32],
        activations = ['relu', 'relu', 'linear'],
        loss_fun = 'mse',
        optimizer = 'adam',
        epsilon_stop_episode = 1500,
        replay_start_size = 1000,
        trained_model=model_name)
done = False

while True:
  if done:
    print(f"Final Score: {tetris.get_score()}")
    tetris.reset()
    
  next_states = tetris.get_next_states()
  best_state = agent.get_best_state(next_states.values())
  best_action = None
  for action, state in next_states.items():
      if state == best_state:
          best_action = action
          break
  tetris.rotate_piece(best_action[1])
  reward, done = tetris.play(best_action[0], render=True)

