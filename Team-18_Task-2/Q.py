import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

grid_size = (100, 100)  # Grid size
route_grid = np.zeros(grid_size)  # Black background (0 means no path)
obstacle_grid = np.zeros(grid_size)  # 1 means obstacle
grade_data = np.random.uniform(0, 10, size=grid_size)  # Random grade (steepness)

# Generate a more linear squiggly path
path_x, path_y = 50, 50  # Starting point
route_grid[path_x, path_y] = 1  # Starting point on the path

# Direction choices for more linear movement
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Up, Down, Left, Right (no diagonals)

# Create a path that branches slightly but stays mostly linear
for _ in range(100):
    # Stick to a direction with a high probability
    if random.random() < 0.8:
        direction = random.choice(directions)  # Main direction (80% chance)
    else:
        # Occasionally branch out (20% chance) by picking a different direction
        direction = random.choice(directions)  # Branching direction

    # Apply the direction
    path_x += direction[0]
    path_y += direction[1]
    
    # Keep the path within bounds
    path_x = np.clip(path_x, 0, grid_size[0] - 1)
    path_y = np.clip(path_y, 0, grid_size[1] - 1)
    
    # Mark the path point
    route_grid[path_x, path_y] = 1  # Mark the path point

# Set obstacles as red (for testing)
obstacle_grid[70:80, 70:80] = -1  # Obstacles in red (-1)

# Visualization function to create test data (black background, white line for paths, red obstacles)
def plot_test_data():
    # Create a black background with paths in white and obstacles in red
    test_data = np.zeros(grid_size)  # Black background
    
    # Place existing paths as white lines
    test_data[route_grid == 1] = 1  # Set white path (1 for paths)
    
    # Place obstacles as red boxes
    test_data[obstacle_grid == -1] = 2  # Set red obstacles (2 for obstacles)
    
    # Create a custom colormap with 3 colors: black, white, and red
    cmap = ListedColormap(['black', 'white', 'red'])
    
    # Plot the grid using the custom colormap
    plt.imshow(test_data, cmap=cmap, interpolation='nearest')
    plt.title('Test Data: Black Background with White Path and Red Obstacles')
    plt.axis('off')
    
    # Add a custom legend
    handles = [plt.Line2D([0], [0], color='white', lw=3, label='Existing Path'),
               plt.Line2D([0], [0], color='red', lw=3, label='Obstacle')]
    plt.legend(handles=handles, loc='upper left')

    plt.show()

# Call this function to display the test data
plot_test_data()

# Sample path (after training) to visualize the final route taken
# This can be replaced with the actual trained path from Q-learning
final_path = [(55, 55), (55, 56), (55, 57), (55, 58), (55, 59), (56, 59), (57, 59), (58, 59), (59, 59), (60, 59)]
final_path = np.array(final_path)

actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8 possible moves

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995  # Decay factor for epsilon
epsilon_min = 0.05  # Minimum value for epsilon

# Initialize Q-table with zeros
Q_table = {}

def get_q_value(state, action):
    """Retrieve Q-value for a state-action pair."""
    if (state, action) not in Q_table:
        Q_table[(state, action)] = 0  # Initialize Q-value if it doesn't exist
    return Q_table[(state, action)]

def update_q_value(state, action, reward, next_state):
    """Update Q-value using the Q-learning formula."""
    max_q_next = max([get_q_value(next_state, a) for a in actions])  # max Q(s', a')
    old_q_value = get_q_value(state, action)
    Q_table[(state, action)] = old_q_value + alpha * (reward + gamma * max_q_next - old_q_value)

def get_next_state(state, action):
    """Calculate the next state given the current state and action."""
    y, x = state
    dy, dx = action
    new_state = (y + dy, x + dx)
    
    # Ensure the new state is within bounds
    if 0 <= new_state[0] < grid_size[0] and 0 <= new_state[1] < grid_size[1]:
        return new_state
    else:
        return None  # Return None for out-of-bounds moves

def get_reward(state, action, grade_threshold):
    """Calculate the reward for taking an action at a given state."""
    y, x = state
    new_state = get_next_state(state, action)
    
    if new_state is None:
        return -10  # Dead end, invalid move, penalize slightly
    
    ny, nx = new_state
    if obstacle_grid[ny, nx] == -1:
        return -10  # Penalty for hitting an obstacle

    if route_grid[ny, nx] == 1:
        # Check if steep terrain is encountered
        if grade_data[ny, nx] > grade_threshold:
            return -5  # Penalty for steep terrain
        else:
            return 10  # Positive reward for valid, traversable route

    return -1  # Negative reward for empty space, not part of the route

def train_q_learning(start, goal, grade_threshold, episodes=1000):
    global epsilon
    path = []  # List to store the agent's path
    
    for episode in range(episodes):
        state = start
        total_reward = 0
        steps = 0
        episode_path = []  # Store the path for this episode
        
        while state != goal and steps < 100:
            if random.random() < epsilon:
                action = random.choice(actions)  # Explore
            else:
                action = max(actions, key=lambda a: get_q_value(state, a))  # Exploit

            reward = get_reward(state, action, grade_threshold)
            next_state = get_next_state(state, action)
            
            if next_state is None:
                continue  # Skip invalid moves
            
            update_q_value(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1
            
            episode_path.append(state)  # Store state in the episode path
        
        path.append(episode_path)  # Add the episode's path to the overall path
        
        # Decay epsilon after each episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print("Training complete!")
    
    return path

# Ensure the start is valid (part of the route)
start = (55, 55)  # Start position now within the route grid
if route_grid[start[0], start[1]] != 1:
    print(f"Start position {start} is invalid! Please set it to a valid route point.")
    exit()

# Define goal position
goal = (90, 90)

# Start training for 5000 episodes (you can adjust the number of episodes)
agent_paths = train_q_learning(start, goal, grade_threshold=5.0, episodes=5000)

# Use the path from the last episode as the final path
final_path = agent_paths[-1]

def plot_final_path():
    # Create a black background for the grid
    test_data = np.zeros(grid_size)
    
    # Place the existing paths (white) and obstacles (red) again
    test_data[route_grid == 1] = 1
    test_data[obstacle_grid == -1] = 2
    
    # Plot the test data
    plt.imshow(test_data, cmap='gray', interpolation='nearest')
    
    # Plot the final path (Blue color for the final path)
    final_x, final_y = zip(*final_path)  # Unzip the path into x and y coordinates
    plt.plot(final_y, final_x, color='blue', marker='o', markersize=5, label="Final Path")
    
    plt.title('Final Path After Training')
    plt.axis('off')

    # Add a legend
    handles = [plt.Line2D([0], [0], color='white', lw=3, label='Existing Path'),
               plt.Line2D([0], [0], color='red', lw=3, label='Obstacle'),
               plt.Line2D([0], [0], color='blue', lw=3, label='Final Path')]
    plt.legend(handles=handles, loc='upper left')
    
    plt.show()

# Call this function to display the final path
plot_final_path()