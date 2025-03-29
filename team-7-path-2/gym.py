import gymnasium as gym
from gym import spaces

import tensorflow as tf
from typing import Optional
import numpy as np


class SpaceEnv(gym.Env):
    
    def __init__(self, config: dict = None):
        config = config or {}
        self.view_distance = config.get("view_distance", 5)
        self.action_size = 8
        
        self.width = config.get("width", 1024)
        self.height = config.get("height", 1024)
        self.values = np.zeros((self.width, self.height))
        
        # Define the action space - 8 possible movement directions
        self.action_space = spaces.Discrete(self.action_size)
        
        # Define the observation space
        self.observation_space = spaces.Dict({
            "direction": spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            ),
            "position": spaces.Box(
                low=0, high=max(self.width, self.height), shape=(2,), dtype=np.float32
            ),
            "scores": spaces.Box(
                low=-float('inf'), high=float('inf'), shape=(8 * self.view_distance,), dtype=np.float32
            )
        })
        
        # Define movement vectors for the 8 possible actions
        self.action_vectors = [
            np.array([0, 1]),    # North
            np.array([1, 1]),    # Northeast
            np.array([1, 0]),    # East
            np.array([1, -1]),   # Southeast
            np.array([0, -1]),   # South
            np.array([-1, -1]),  # Southwest
            np.array([-1, 0]),   # West
            np.array([-1, 1]),   # Northwest
        ]
        
        self.default_position = config.get("default_position", np.array([0, 0]))
        self.max_steps = config.get("max_steps", 250)
        self.penalty = config.get("penalty", 0.1)
        self.target_location = config.get("target_location", np.array([self.width-1, self.height-1]))
        
    def step(self, action):
        # Update position based on the selected action
        action_vector = self.action_vectors[action]
        self.position = np.clip(
            self.position + action_vector, 
            np.zeros(2), 
            np.array([self.width-1, self.height-1])
        )
        
        # Check if we've reached the target
        terminated = np.array_equal(self.position.astype(int), self.target_location)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # Calculate reward
        if terminated:
            reward = 100
        else:
            # Get the value at the current position as a reward component
            current_value = self._get_nearby_val(int(self.position[0]), int(self.position[1]))
            reward = current_value - self.penalty
            
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the agent's position
        if options and "pos" in options:
            self.position = options["pos"]
        else:
            self.position = self.default_position.copy()
        
        self.current_step = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def _get_nearby_val(self, x, y):
        # Ensure coordinates are within bounds
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        return self.values[x, y]
    
    def get_nearby_vals(self):
        coord_list = []
        ret_array = np.zeros((9,))
        
        idx = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                pos_x = int(self.position[0]) + i
                pos_y = int(self.position[1]) + j
                coord_list.append((pos_x, pos_y))
                ret_array[idx] = self._get_nearby_val(pos_x, pos_y)
                idx += 1
                
        return coord_list, ret_array

    def _get_obs(self):
        # Get directional information based on view distance
        direction_obs = np.zeros((8 * self.view_distance,), dtype=np.float32)
        # Get nearby values
        _, nearby_values = self.get_nearby_vals()
        
        return {
            "position": self.position.astype(np.float32),
            "direction": direction_obs,
            "scores": nearby_values.astype(np.float32)
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self.position - self.target_location, ord=1
            )
        }

    def render(self, mode='human'):
        # Simple rendering for debugging
        print(f"Agent position: {self.position}")
        print(f"Target position: {self.target_location}")
        print(f"Current step: {self.current_step}/{self.max_steps}")

    def close(self):
        pass

centroids = np.load("centroids.npy")
centroids.shape

ndvi_points = np.load("ndvi_points_2d.npy").flatten().reshape((-1, 1))
ndvi_points.shape

class KerasDQNModel(tf.keras.Model):
    """Custom Keras model for DQN."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(KerasDQNModel, self).__init__()
        
        # Process each observation component separately
        self.position_input = keras.layers.Input(shape=obs_space["position"].shape)
        self.direction_input = keras.layers.Input(shape=obs_space["direction"].shape)
        self.scores_input = keras.layers.Input(shape=obs_space["scores"].shape)
        
        # Process position
        position_layer = keras.layers.Dense(64, activation='relu')(self.position_input)
        
        # Process direction information
        direction_layer = keras.layers.Dense(128, activation='relu')(self.direction_input)
        
        # Process nearby scores
        scores_layer = keras.layers.Dense(64, activation='relu')(self.scores_input)
        
        # Concatenate all inputs
        concat = keras.layers.Concatenate()([position_layer, direction_layer, scores_layer])
        
        # Shared representation
        hidden = keras.layers.Dense(256, activation='relu')(concat)
        hidden = keras.layers.Dense(128, activation='relu')(hidden)
        
        # Action outputs
        self.action_out = keras.layers.Dense(num_outputs)(hidden)
        
        # Value output for Actor-Critic methods
        self.value_out = keras.layers.Dense(1)(hidden)
        
        # Build model
        self.model = keras.Model(
            inputs=[self.position_input, self.direction_input, self.scores_input],
            outputs=[self.action_out, self.value_out]
        )

    def call(self, inputs):
        # Convert observation to tensor
        return self.action_model(inputs)

from gym.envs.registration import register
from ray.tune.registry import register_env

register(
    id='dqn_model/rl',
    entry_point='model.envs:DQN',
    max_episode_steps=300,
)

def env_creator(env_config):
    return SpaceEnv(env_config)

register_env("dqn_model/rl", env_creator)


from ray.rllib.algorithms.dqn.dqn import DQN, DQNConfig

dqn_config = DQNConfig().training(gamma=0.9, lr=0.01).environment("dqn_model/rl")
dqn_trainer = dqn_config.build(env="dqn_model/rl")
algo = dqn_config.build()
algo.train()
algo.stop()


