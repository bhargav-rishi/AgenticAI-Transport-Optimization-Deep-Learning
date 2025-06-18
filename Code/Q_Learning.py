import numpy as np
import pandas as pd

# Step 1: Define state and action spaces
origin_zones = list(range(100, 110))   # Simplified example: 10 origin zones
destination_zones = list(range(200, 210))  # 10 destination zones
modes = [
    "dms_mode_Truck", "dms_mode_Rail", "dms_mode_Water",
    "dms_mode_Air", "dms_mode_Multiple_Modes_and_Mail",
    "dms_mode_Pipeline", "dms_mode_Other_and_Unknown"
]

# Step 2: Initialize Q-table
states = [(o, d) for o in origin_zones for d in destination_zones]
Q_table = pd.DataFrame(0, index=pd.MultiIndex.from_tuples(states, names=["origin", "destination"]),
                       columns=modes)

# Step 3: Define learning parameters
alpha = 0.1      # learning rate
gamma = 0.9      # discount factor
epsilon = 0.2    # exploration rate
episodes = 5000

# Simulated reward function (replace with actual model later)
def get_reward(origin, destination, mode):
    # Simulate cheaper cost for Truck and Rail over long distances
    if mode == "dms_mode_Truck":
        return -100 + np.random.randn() * 10
    elif mode == "dms_mode_Air":
        return -300 + np.random.randn() * 20
    elif mode == "dms_mode_Rail":
        return -150 + np.random.randn() * 10
    elif mode == "dms_mode_Water":
        return -170 + np.random.randn() * 5
    else:
        return -200 + np.random.randn() * 15

# Step 4: Q-learning loop
for episode in range(episodes):
    origin = np.random.choice(origin_zones)
    destination = np.random.choice(destination_zones)
    state = (origin, destination)

    # Choose action
    if np.random.rand() < epsilon:
        action = np.random.choice(modes)  # Explore
    else:
        action = Q_table.loc[state].idxmax()  # Exploit

    # Get reward (simulate for now)
    reward = get_reward(origin, destination, action)

    # Update Q-table
    current_q = Q_table.loc[state, action]
    max_future_q = Q_table.loc[state].max()
    new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
    Q_table.loc[state, action] = new_q

# Save Q-table
q_table_path = "/content/sample_data/q_learning_freight_agent.csv"
Q_table.to_csv(q_table_path)
q_table_path
