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
        # do not modify these
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'  # mode is either 'train' or 'eval', we will set the mode of your agent to eval mode
        else:
            self.mode = mode

        # modify these
        self.storage = deque(maxlen=10000)  # a data structure of your choice (D in the Algorithm 2)
        # A neural network MLP model which can be used as Q
        self.network = MLPRegression(input_dim=4, output_dim=2, learning_rate=1e-3)
        # network2 has identical structure to network1, network2 is the Q_f
        self.network2 = MLPRegression(input_dim=4, output_dim=2, learning_rate=1e-3)
        # initialise Q_f's parameter by Q's, here is an example
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        self.epsilon = 0.1  # probability ε in Algorithm 2
        self.n = 32  # the number of samples you'd want to draw from the storage each time
        self.discount_factor = 0.99  # γ in Algorithm 2
 
        # do not modify this
        if load_model_path:
            self.load_model(load_model_path)

    def choose_action(self, state: dict, action_table: dict) -> int:
        """
        This function should be called when the agent action is requested.
        Args:
            state: input state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            action: the action code as specified by the action_table
        """
        # following pseudocode to implement this function

        state_vector = self.build_state(state)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)

        if self.mode == 'train':
            if random.random() < self.epsilon:
                a_t = np.random.choice([action_table['jump'], action_table['do_nothing']])
            else:
                q_values = self.network(state_tensor)
                a_t = torch.argmax(q_values).item()

        elif self.mode == 'eval':
            q_values = self.network(state_tensor)
            a_t = torch.argmax(q_values).item()
        
        return a_t

    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        """
        This function should be called to notify the agent of the post-action observation.
        Args:
            state: post-action state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            None
        """
        # following pseudocode to implement this function

        if self.mode == 'train':
            next_state_vector = self.build_state(state)
            reward = self.reward(state)
            if (state['done']):
                a_t_next = 0
            else:
                next_state_tensor = torch.from_numpy(next_state_vector).float().unsqueeze(0)
                q_next = self.network(next_state_tensor)
                a_t_next = np.argmax(q_next.detach().numpy())

            self.storage.append((reward, a_t_next))

    def save_model(self, path: str = 'my_model.ckpt'):
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        """
        Update one MLP model's model parameter by the parameter of another MLP model.
        Args:
            net_to_update: the MLP to be updated
            net_as_source: the MLP to supply the model parameters

        Returns:
            None
        """
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
            pipe_bottom = next_pipe['bottom']
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
                        normalized_distance_to_pipe, normalized_distance_to_gap],  dtype=np.float32)
    
    def reward(self, state: dict):
        done = state['done']
        done_type = state['done_type']
        score = state['score']
        mileage = state['mileage']

        if done:
            if done_type == 'hit_pipe':
                return -100
            elif done_type == 'offscreen':
                return -50
            elif done_type == 'well_done':
                return 100
            
        reward = 0.1

        if score > 0:
            reward+=1

        reward += mileage * 0.01
        return reward

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)

    args = parser.parse_args()

    # bare-bone code to train your agent (you may extend this part as well, we won't run your agent training code)
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=True, level=args.level, game_length=10)
    agent = MyAgent(show_screen=True)
    episodes = 10000
    for episode in range(episodes):
        env.play(player=agent)

        # env.score has the score value from the last play
        # env.mileage has the mileage value from the last play
        print(env.score)
        print(env.mileage)

        # store the best model based on your judgement
        agent.save_model(path='my_model.ckpt')

        # you'd want to clear the memory after one or a few episodes
        ...

        # you'd want to update the fixed Q-target network (Q_f) with Q's model parameter after one or a few episodes
        ...

    # the below resembles how we evaluate your agent
    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
    agent2 = MyAgent(show_screen=False, load_model_path='my_model.ckpt', mode='eval')

    episodes = 10
    scores = list()
    for episode in range(episodes):
        env2.play(player=agent2)
        scores.append(env2.score)

    print(np.max(scores))
    print(np.mean(scores))
