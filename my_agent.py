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

        self.storage = deque(maxlen=20000)  
        self.network = MLPRegression(input_dim=9, output_dim=2, learning_rate=5e-4)  
        self.network2 = MLPRegression(input_dim=9, output_dim=2, learning_rate=5e-4)
        
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        self.epsilon = 1.0  
        self.min_epsilon = 0.01  
        self.epsilon_decay = 0.9997
        self.n = 64 
        self.discount_factor = 0.99  

        self.state_vector_before_action = None
        self.action_before_action = None
        self.last_pipe_passed = 0
        self.last_action_was_jump = 0 
 
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
        state_vector = self.build_state(state)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)

        self.state_vector_before_action = state_vector

        if self.mode == 'eval' and self.last_action_was_jump > 0 and state['bird_velocity'] < -5:
            action_idx = 1  
        else:
            if self.mode == 'train':
                if random.random() < self.epsilon:
                    action_idx = np.random.choice([0, 1])
                else:
                    q_values = self.network(state_tensor)
                    action_idx = torch.argmax(q_values).item()
            else:  
                q_values = self.network(state_tensor)
                action_idx = torch.argmax(q_values).item()

        self.action_before_action = action_idx

        if action_idx == 0:
            action = action_table['jump']
            self.last_action_was_jump = 2 
        else:
            action = action_table['do_nothing']
            self.last_action_was_jump = max(0, self.last_action_was_jump - 1)  
        
        return action

    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        """
        This function should be called to notify the agent of the post-action observation.
        Args:
            state: post-action state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            None
        """
        if self.mode != 'train' or self.state_vector_before_action is None:
            return

        reward = self.reward(state)
        
        next_state_vector = self.build_state(state)
        
        self.storage.append((
            self.state_vector_before_action,
            self.action_before_action,
            next_state_vector,
            reward,
            state['done']
        ))
        
        if len(self.storage) >= self.n:
            minibatch = random.sample(self.storage, self.n)
            
            states = np.array([data[0] for data in minibatch], dtype=np.float32)
            actions = np.array([data[1] for data in minibatch])
            next_states = np.array([data[2] for data in minibatch], dtype=np.float32)
            rewards = np.array([data[3] for data in minibatch])
            dones = np.array([data[4] for data in minibatch])

            current_q_values = self.network.predict(states)
            
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
            next_q_values = self.network2(next_states_tensor).detach().numpy()
            
            targets = np.copy(current_q_values)
            for i in range(self.n):
                if dones[i]:
                    targets[i, actions[i]] = rewards[i]
                else:
                    targets[i, actions[i]] = rewards[i] + self.discount_factor * np.max(next_q_values[i])
            
            weights = np.zeros_like(targets)
            for i in range(self.n):
                weights[i, actions[i]] = 1.0
            
            self.network.fit_step(states, targets, weights)
        
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

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

        pipe_x = screen_width
        pipe_top = 0
        pipe_bottom = screen_height
        gap_center = screen_height / 2
        next_pipe_x = screen_width * 1.5
        next_pipe_gap_center = screen_height / 2

        if pipes:
            next_pipe = pipes[0]
            pipe_x = next_pipe['x']
            pipe_top = next_pipe['top']
            pipe_bottom = next_pipe['bottom']
            gap = pipe_bottom - pipe_top
            gap_center = pipe_top + (gap / 2)
  
            if len(pipes) > 1:
                next_next_pipe = pipes[1]
                next_pipe_x = next_next_pipe['x']
                next_pipe_top = next_next_pipe['top']
                next_pipe_bottom = next_next_pipe['bottom']
                next_pipe_gap = next_pipe_bottom - next_pipe_top
                next_pipe_gap_center = next_pipe_top + (next_pipe_gap / 2)

        normalized_bird_height = bird_y / screen_height
        normalized_velocity = bird_velocity / 10
        normalized_pipe_top = pipe_top / screen_height
        normalized_pipe_bottom = pipe_bottom / screen_height
        normalized_distance_to_pipe = (pipe_x - state['bird_x']) / screen_width
        normalized_distance_to_gap = (bird_y - gap_center) / screen_height
        normalized_next_pipe_x = (next_pipe_x - state['bird_x']) / screen_width
        normalized_next_pipe_gap = (bird_y - next_pipe_gap_center) / screen_height

        normalized_last_jump = self.last_action_was_jump / 2.0

        return np.array([
            normalized_bird_height,
            normalized_velocity,
            normalized_pipe_top,
            normalized_pipe_bottom,
            normalized_distance_to_pipe,
            normalized_distance_to_gap,
            normalized_next_pipe_x,
            normalized_next_pipe_gap,
            normalized_last_jump
        ], dtype=np.float32)

    def reward(self, state: dict):
        done = state['done']
        done_type = state['done_type']
        score = state['score']
        bird_y = state['bird_y']
        bird_velocity = state['bird_velocity']
        screen_height = state['screen_height']
        
        reward = 0.1 
   
        if score > self.last_pipe_passed:
            reward += 1.5 
            self.last_pipe_passed = score
       
        normalized_y = bird_y / screen_height
        if normalized_y < 0.05:
            reward -= 1.0 
        elif normalized_y > 0.95:
            reward -= 1.0  
        elif normalized_y < 0.1 or normalized_y > 0.9:
            reward -= 0.3  
        
        if bird_velocity < -10:
            reward -= 0.8
        elif bird_velocity > 8:
            reward -= 0.5
            
        if state['pipes']:
            next_pipe = state['pipes'][0]
            pipe_gap_center = next_pipe['top'] + (next_pipe['bottom'] - next_pipe['top']) / 2
            distance_to_gap = abs(bird_y - pipe_gap_center) / screen_height

            if distance_to_gap < 0.1:
                reward += 0.3
            elif distance_to_gap < 0.2:
                reward += 0.1
                
            if 0.4 < normalized_y < 0.6 and next_pipe['x'] > state['bird_x']:
                reward += 0.1
                
        if done:
            if done_type == 'hit_pipe':
                return -1.5 
            elif done_type == 'offscreen':
                return -3.0 
            elif done_type == 'well_done':
                return 5.0
                
        return reward

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--episodes', type=int, default=20000)

    args = parser.parse_args()

    # if args.mode == 'train':
    #     env = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level, game_length=10)
    #     agent = MyAgent(show_screen=False)
    #     episodes = args.episodes

    #     total_score = 0
    #     episode_count = 0
    #     best_average = 0
    #     last_10_scores = deque(maxlen=10)

    #     for episode in range(episodes):
    #         # Reset counters for each episode
    #         agent.last_pipe_passed = 0
    #         agent.last_action_was_jump = 0
            
    #         env.play(player=agent)
            
    #         # Track performance
    #         total_score += env.score
    #         episode_count += 1
    #         last_10_scores.append(env.score)
            
    #         print(f"Episode {episode+1}, Score: {env.score}, Epsilon: {agent.epsilon:.4f}, Recent Avg: {sum(last_10_scores)/len(last_10_scores):.2f}")
            
    #         if episode_count % 100 == 0:
    #             avg_score = total_score / episode_count
    #             print(f"Last 100 episodes average score: {avg_score:.2f}")
                
    #             # Save model if it's performing better
    #             if avg_score > best_average:
    #                 best_average = avg_score
    #                 agent.save_model(path='my_model.ckpt')
    #                 print(f"New best model saved with average score: {best_average:.2f}")
                
    #             total_score = 0
    #             episode_count = 0
            
    #         if episode % 50 == 0:  
    #             MyAgent.update_network_model(agent.network2, agent.network)
    #             print("Target network updated")
                
    #         if episode % 1000 == 0 and episode > 0:
    #             agent.save_model(path=f'checkpoint_model_{episode}.ckpt')
    #             print(f"Checkpoint model saved at episode {episode}")

    # else:  
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=True, level=args.level)
    agent = MyAgent(show_screen=True, load_model_path='my_model.ckpt', mode='eval')

    episodes = 10
    scores = []
    
    for episode in range(episodes):
        env.play(player=agent)
        scores.append(env.score)
        print(f"Evaluation episode {episode+1}, Score: {env.score}")

    print(f'Average score: {np.mean(scores):.2f}')
    print(f'Max score: {np.max(scores)}')
    print(f'Min score: {np.min(scores)}')
