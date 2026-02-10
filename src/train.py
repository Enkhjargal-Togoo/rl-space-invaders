# **** Training with hyperparameters - Space Invaders (RAM) ****
ENV_ID = "ALE/SpaceInvaders-v5"
env = gym.make(ENV_ID, obs_type="ram")
eval_env = gym.make(ENV_ID, obs_type="ram", render_mode="rgb_array")

# Header showing
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

gamma = 0.99
hidden_sizes = (256, 256)
learning_rate = 1e-3
epsilon = 1.0
replay_size = 200_000
minibatch_size = 128
target_update = 4000
max_episodes = 1000
max_steps = 5000
criterion_episodes = 20

agent = AgentDDQN(env,
                  gamma=gamma,
                  hidden_sizes=hidden_sizes,
                  learning_rate=learning_rate,
                  epsilon=epsilon,
                  replay_size=replay_size,
                  minibatch_size=minibatch_size,
                  target_update=target_update)

agent.train(max_episodes, lambda x : min(x) >= 200, criterion_episodes)

# visualise one episode
state, _ = eval_env.reset()
terminated = False
truncated = False
steps = 0
total_reward = 0
frames = []
while not (terminated or truncated or steps > max_steps):
    frames.append(eval_env.render())

    # take action based on policy
    action = agent.policy(preprocess_obs(state))  #Policy also normalised obs

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    state, reward, terminated, truncated, info = eval_env.step(action)
    total_reward += reward
    steps += 1

agent.plot_reward_comparison2()

#Storing episode metrics
print(f'Reward: {total_reward}')

# **** Evaluate average performance over several runs ****
mean_ret, std_ret = agent.evaluate(n_episodes=10)  # uses the greedy, RAW evaluation method
print(f"\nAverage evaluation reward over 10 episodes (raw game score): {mean_ret:.1f} ± {std_ret:.1f}")


# close the environment
eval_env.close()

# create and play video clip using the frames and given fps
clip = ImageSequenceClip(frames, fps=50)
clip.ipython_display(rd_kwargs=dict(logger=None))
