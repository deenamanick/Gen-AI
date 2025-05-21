Here are some **practical reinforcement learning (RL) exercises**, ranging from beginner to advanced levels. These will help you understand key concepts like environments, agents, rewards, and policies using popular libraries like `OpenAI Gym`, `Stable Baselines3`, and `PyTorch`.

---

## **1. Beginner Exercises**
### **A. OpenAI Gym Basics**
**Problem:** Train an agent to solve the `CartPole-v1` environment (balance a pole on a cart).  
**Tools:** `gym` (OpenAI Gym)  
**Tasks:**  
1. Install and load the environment:
   ```python
   import gym
   env = gym.make("CartPole-v1")
   ```
2. Run a **random agent** (take random actions) and observe rewards.  
3. Implement a **hardcoded policy** (e.g., "move left if pole tilts left").  
4. Plot the rewards over episodes.  

**Key Concepts:**  
- Environment (`env.step()`, `env.reset()`).  
- Observations vs. actions.  
- Reward maximization.  

---

### **B. Q-Learning (Tabular RL)**
**Problem:** Solve the `FrozenLake-v1` environment (navigate icy grid to reach goal).  
**Tools:** `gym`, NumPy  
**Tasks:**  
1. Implement the **Q-table** (states × actions).  
2. Train using **Q-learning**:
   - Update rule:  
     ```python
     Q[state, action] += alpha * (reward + gamma * max(Q[new_state]) - Q[state, action])
     ```
   - Tune hyperparameters (`alpha`, `gamma`, `epsilon`).  
3. Test the trained agent and compare with random actions.  

**Key Concepts:**  
- Exploration vs. exploitation (`epsilon-greedy`).  
- Discount factor (`gamma`).  

---

## **2. Intermediate Exercises**
### **A. Deep Q-Network (DQN)**
**Problem:** Play `LunarLander-v2` (land a spaceship safely).  
**Tools:** `gym`, `PyTorch`/`TensorFlow`, `Stable Baselines3`  
**Tasks:**  
1. Preprocess observations (e.g., normalize pixels in Atari games).  
2. Implement a **DQN**:
   - Use a neural network to approximate Q-values.  
   - Replay buffer for experience sampling.  
   - Target network for stability.  
3. Train and evaluate (compare with random agent).  

**Key Concepts:**  
- Function approximation (replacing Q-table with NN).  
- Experience replay.  

---

### **B. Policy Gradients (REINFORCE)**
**Problem:** Solve `CartPole-v1` using policy gradients.  
**Tools:** `gym`, `PyTorch`  
**Tasks:**  
1. Define a policy network (outputs action probabilities).  
2. Train using **REINFORCE**:
   - Collect trajectories.  
   - Compute discounted rewards.  
   - Update policy via gradient ascent.  
3. Compare with DQN (which learns faster?).  

**Key Concepts:**  
- Policy-based vs. value-based RL.  
- Monte Carlo sampling.  

---

## **3. Advanced Exercises**
### **A. Proximal Policy Optimization (PPO)**
**Problem:** Train a robot to walk (`BipedalWalker-v3`).  
**Tools:** `Stable Baselines3`  
**Tasks:**  
1. Use `PPO` from Stable Baselines3:  
   ```python
   from stable_baselines3 import PPO
   model = PPO("MlpPolicy", env, verbose=1)
   model.learn(total_timesteps=100000)
   ```
2. Tune hyperparameters (`learning_rate`, `n_steps`).  
3. Render trained agent’s performance.  

**Key Concepts:**  
- Actor-critic methods.  
- Trust region optimization.  

---

### **B. Multi-Agent RL (MADDPG)**
**Problem:** Simulate cooperative/competitive agents (e.g., `PettingZoo`).  
**Tools:** `PettingZoo`, `RLlib`  
**Tasks:**  
1. Set up a **multi-agent environment** (e.g., `simple_adversary`).  
2. Train using **MADDPG** (decentralized actors, centralized critics).  
3. Analyze emergent behaviors (do agents cooperate or compete?).  

**Key Concepts:**  
- Centralized training, decentralized execution.  
- Non-stationary environments.  

---

## **4. Bonus: Real-World Applications**
### **A. Self-Driving Car (Simulator)**
**Tools:** `CARLA`, `AirSim`  
**Task:** Train an RL agent to follow lanes using camera inputs.  

### **B. Stock Trading (Custom Environment)**
**Tools:** `gym`, `PyTorch`  
**Task:** Build an RL agent to maximize portfolio returns.  

---

## **Tips for Success**
- **Start simple** (CartPole → LunarLander → Atari).  
- **Monitor training** (TensorBoard, reward plots).  
- **Hyperparameter tuning** matters (e.g., `gamma`, `batch_size`).  
- **Read papers** (DQN, PPO, SAC) for deeper intuition.  

