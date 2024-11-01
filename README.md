# Trading Environment with Reinforcement Learning (DQN)

This repository provides a custom Gym environment for simulating a basic stock trading scenario and training a reinforcement learning (RL) model using the Stable-Baselines3 `DQN` algorithm. 

## Overview

This project includes:
- A custom trading environment that allows the agent to make trading decisions (Hold, Buy, Sell) based on stock market data.
- An RL model based on the DQN (Deep Q-Network) algorithm to train the agent.
- Example code to initialize, train, and save the model.

## Requirements

- Python 3.8+
- Libraries: `gym`, `numpy`, `pandas`, `stable-baselines3`

Install the required libraries:
```bash
pip install gym numpy pandas stable-baselines3
```

## Project Structure

- `TradingEnv`: A custom Gym environment that simulates trading using stock data. Actions include holding, buying, and selling shares. The environment keeps track of the agent's balance and the number of shares held to calculate rewards.
- `data`: Generated stock data with columns `Open`, `High`, `Low`, `Close`, and `Volume`.
- `model`: A DQN model, which can be substituted with other RL algorithms in `stable-baselines3`.

## Code Overview

### 1. Custom Environment (TradingEnv)

The custom environment simulates trading actions and calculates rewards based on the agent's portfolio value. The agent's observation includes the current step’s stock data, account balance, and number of shares held.

### 2. Training the Model

Using `stable-baselines3`'s DQN algorithm:
```python
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)
model.save("dqn_trading_model")
```

### 3. Run and Train

```python
# Initialize the environment
env = DummyVecEnv([lambda: TradingEnv(data)])

# Train the model
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

# Save the model
model.save("dqn_trading_model")
```

## Usage

To load and use the model:
```python
from stable_baselines3 import DQN

# Load the trained model
model = DQN.load("dqn_trading_model")

# Make predictions in the environment
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

## Model Training Results

Below are the results logged during the training of the model using the DQN algorithm:

| Metric                | Value   |
|-----------------------|---------|
| Episodes              | 48      |
| Exploration Rate      | 0.05    |
| FPS                   | 3281    |
| Time Elapsed          | 1 sec   |
| Total Timesteps       | 4752    |

Detailed logs for each interval:
```plaintext
----------------------------------
| rollout/            |          |
|    exploration_rate | 0.248    |
| time/               |          |
|    episodes         | 4        |
|    fps              | 2857     |
|    time_elapsed     | 0        |
|    total_timesteps  | 396      |
----------------------------------
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 8        |
|    fps              | 3419     |
|    time_elapsed     | 0        |
|    total_timesteps  | 792      |
----------------------------------
... (Additional logs for each episode)
----------------------------------
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 48       |
|    fps              | 3281     |
|    time_elapsed     | 1        |
|    total_timesteps  | 4752     |
----------------------------------
```

These results indicate the model's performance in terms of frame rate, exploration rate, and timesteps, reflecting efficient training across 48 episodes.

## Project Motivation

The goal behind building this custom trading environment and reinforcement learning model is to simulate a realistic stock trading scenario and assess how reinforcement learning can enhance trading decisions. By using DQN, the project explores an algorithm's ability to dynamically adjust buy/sell/hold strategies based on historical stock data, with the aim of optimizing portfolio performance over time. In the future I want to apply this code to other platforms and try to model different trading strategies and observe these results.

## Requirements Verification

Check your Stable Baselines3 version and installation path:
```python
import stable_baselines3
print(stable_baselines3.__version__)
print(stable_baselines3.__file__)
```

## License

This project is licensed under the MIT License.

---

This README now includes the training results section, making it comprehensive for others to understand, set up, and run your code on GitHub.
