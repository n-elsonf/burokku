from agent import DQNAgent
from tetris import Tetris
from heuristic import TetrisAI
from tqdm import tqdm
import numpy as np
import sys


class AgentParam:
    def __init__(self):
        self.discount = 0.98
        self.state_size = 4
        self.batch_size = 512 # 128
        self.buffer_size = 1000
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.n_neurons = [32, 32]
        self.activations = ['relu', 'relu', 'linear']
        self.loss_fun = 'mse'
        self.optimizer = 'adam'
        self.epsilon_stop_episode = 1500
        self.max_steps = 1000
        self.episodes = 2000
        self.epochs = 1
        self.replay_start_size = 1000

def run_dqn(agentparam: AgentParam, save_model):
    tetris = Tetris()
    scores = []

    agent = DQNAgent(
        discount=agentparam.discount,
        state_size=agentparam.state_size,
        batch_size=agentparam.batch_size,
        buffer_size=agentparam.buffer_size,
        epsilon=agentparam.epsilon,
        epsilon_min=agentparam.epsilon_min,
        n_neurons=agentparam.n_neurons,
        activations=agentparam.activations,
        loss_fun=agentparam.loss_fun,
        optimizer=agentparam.optimizer,
        epsilon_stop_episode=agentparam.epsilon_stop_episode,
        replay_start_size=agentparam.replay_start_size,
    )

    best_score = 0
    target_score = 10000

    for episode in tqdm(range(agentparam.episodes)):
        curr_state = tetris.reset()
        done = False
        step = 0

        print(f"Episode {episode + 1}/{agentparam.episodes}")
        
        while not done and (agentparam.max_steps != 0 or step < agentparam.max_steps):
            next_states = tetris.get_next_states()
            
            # print(next_states)

            if not next_states:
                print("No next states available. Ending episode.")
                break

            best_state = agent.get_best_state((next_states.values()))
            if best_state is None:
                print("Agent failed to select a valid state.")
                break

            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            # render = False
            # if episode >= 1000:
            #     render = True
            tetris.rotate_piece(best_action[1])
            reward, done = tetris.play(best_action[0], render=True)
            
            agent.remember(curr_state, next_states[best_action], reward, done)
            curr_state = next_states[best_action]
            step += 1

        print(f"Training on episode {episode + 1}")
        print(f"Final score: {tetris.get_score()}")

        scores.append(tetris.get_score())
        agent.train(batch_size=agentparam.batch_size, epochs=agentparam.epochs)

        if save_model and tetris.get_score() >= best_score:
            best_score = tetris.get_score()
            print(f"New best score, saving.")
            agent.model.save("trained_model_v3.keras")

        if save_model and tetris.get_score() >= target_score:
            print(f"Reached score at least 10,000, saving and exiting.")
            agent.model.save("trained_model_v3.keras")
            sys.exit(0)

            
    
    print("scores", scores)



if __name__ == "__main__":
    agentparam = AgentParam()
    run_dqn(agentparam, False)
