import numpy as np
import pygame
import random
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv
from collections import deque
import torch

STUDENT_ID = 'a1850943'
DEGREE = 'UG'

class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'
        else:
            self.mode = mode

        self.storage = deque(maxlen=10000)
        self.network = MLPRegression(input_dim=4, output_dim=2, learning_rate=1e-3)
        self.network2 = MLPRegression(input_dim=4, output_dim=2, learning_rate=1e-3)
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.999
        self.n = 32
        self.discount_factor = 0.99

        self.state_vector_before_action = None
        self.action_before_action = None

        self.train_step_counter = 0

        if load_model_path:
            self.load_model(load_model_path)

    def choose_action(self, state: dict, action_table: dict) -> int:
        state_vector = self.build_state(state)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)

        if self.mode == 'train':
            if random.random() < self.epsilon:
                action = np.random.choice([0, 1])
            else:
                q_values = self.network(state_tensor)
                action = torch.argmax(q_values).item()
        elif self.mode == 'eval':
            q_values = self.network(state_tensor)
            action = torch.argmax(q_values).item()

        if action == 0:
            action = action_table['jump']
        else:
            action = action_table['do_nothing']

        if self.mode == 'train':
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return action

    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        next_state_vector = None

        if self.mode == 'train':
            next_state_vector = self.build_state(state)
            reward = self.reward(state)

            self.storage.append((
                self.state_vector_before_action,
                self.action_before_action,
                next_state_vector,
                reward,
                state['done']
            ))

            if len(self.storage) >= self.n:
                self.train_step()

        self.state_vector_before_action = next_state_vector
        self.action_before_action = 0 if next_state_vector is None else np.argmax(
            self.network(torch.tensor(next_state_vector, dtype=torch.float32).unsqueeze(0)).detach().numpy()
        )

    def train_step(self):
    # Sample a minibatch
        minibatch = random.sample(self.storage, self.n)
        
        # Ensure that the state vectors are not None
        valid_minibatch = [(s, a, ns, r, d) for s, a, ns, r, d in minibatch if s is not None]
        
        if not valid_minibatch:  # If there are no valid states in the minibatch, exit
            return

        # Create X from valid state vectors
        X = np.array([s.flatten() for s, a, ns, r, d in valid_minibatch], dtype=np.float32)
        
        # Get predictions from the network
        Y = self.network.predict(X)
        W = np.zeros_like(Y)
        
        # Loop through the minibatch to calculate the target Q-values
        for idx, (state_before, action_taken, next_state, reward_received, game_over) in enumerate(valid_minibatch):
            if state_before is None or action_taken is None:
                continue
            
            # If game is over, the Q-value is simply the reward received
            if game_over:
                target_q = reward_received
            else:
                # Get the Q-values for the next state
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                best_action_online = torch.argmax(self.network(next_state_tensor)).item()
                next_q_target = self.network2(next_state_tensor)
                target_q = reward_received + self.discount_factor * next_q_target[0, best_action_online].item()

            # Update Y and W for the specific action taken
            if idx < Y.shape[0] and action_taken < Y.shape[1]:
                Y[idx, action_taken] = target_q
                W[idx, action_taken] = 1.0

        # Perform a training step
        self.network.fit_step(X, Y, W)

        # Increment training step counter
        self.train_step_counter += 1
        if self.train_step_counter % 500 == 0:
            self.update_network_model(self.network2, self.network)
       
    def save_model(self, path: str = 'my_model.ckpt'):
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        net_to_update.load_state_dict(net_as_source.state_dict())

    def build_state(self, state: dict):
        bird_y = state['bird_y']
        bird_velocity = state['bird_velocity']
        screen_height = state['screen_height']
        screen_width = state['screen_width']
        pipes = state['pipes']

        if pipes:
            next_pipe = pipes[0]
            pipe_x = next_pipe['x']
            pipe_top = next_pipe['top']
            gap = state['pipe_attributes']['gap']
            gap_center = pipe_top + (gap / 2)
        else:
            pipe_x = screen_width
            gap_center = screen_height / 2

        normalized_bird_height = bird_y / screen_height
        normalized_velocity = bird_velocity / 10
        normalized_distance_to_pipe = (pipe_x - state['bird_x']) / screen_width
        normalized_distance_to_gap = (gap_center - bird_y) / screen_height

        return np.array([normalized_bird_height, normalized_velocity,
                         normalized_distance_to_pipe, normalized_distance_to_gap], dtype=np.float32)

    def reward(self, state: dict):
        done = state['done']
        done_type = state['done_type']
        score = state['score']
        mileage = state['mileage']

        if done:
            if done_type == 'hit_pipe':
                return -50
            elif done_type == 'offscreen':
                return -100
            elif done_type == 'well_done':
                return 100

        reward = 0.1
        if score > 0:
            reward += 1
        reward += mileage * 0.1
        return reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)
    args = parser.parse_args()

    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level, game_length=10)
    agent = MyAgent(show_screen=False)
    episodes = 10000

    for episode in range(episodes):
        env.play(player=agent)

        print("episode number", episode+1)
        print(env.score)
        print(env.mileage)

        agent.save_model(path='best_model_fixed.ckpt')

    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
    agent2 = MyAgent(show_screen=False, load_model_path='best_model_fixed.ckpt', mode='eval')

    episodes = 10
    scores = list()
    for episode in range(episodes):
        env2.play(player=agent2)
        scores.append(env2.score)

    print('chat average score', np.mean(scores))
    print(np.max(scores))
    print(np.mean(scores))