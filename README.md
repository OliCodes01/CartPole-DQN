# Working DQN for the CartPole-V1 environment
- In this project I implemented a Deep Q-network using PyTorch and trained it to solve the CartPole-v1 environment. 
The agent consistently balances the pole for > 200 steps after training and completes the 'game' everytime.



# Details
- Implemented a DQN using PyTorch. Containing a target and policy / online network.
- Uses replay memory, to train in batches for more stable and better learning
- Epsilon greedy exploration strategy
- Both the target and policy networks are saved after training and can be loaded for evaluation (To-Do)

# Results
- A working DQN to solve the CartPole-v1 environment


# What I learned
- How a DQN is constructed
- Bellman's equation
- Tensors
- Familiarity with PyTorch
- Hyperparameter tuning (Still need to test this a bit more)
- Importance of replay memory (Previous attempt was a single step DQN, which performed very poorly)

# TODO
- Evaluation metrics for training insights (average reward, moving averages & more)
- Documentation
- Hyperparameter tuning and comparison across different runs
- Implement double DQN for even better results
- Prioritized replay memory for smarter sampling
- Applying this to Flappy Bird

