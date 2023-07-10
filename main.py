# messing around with reinforcement learning on openai gym
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

def train_agent(model, timesteps=20000):
    # train agent
    model.learn(total_timesteps=timesteps)

def test_agent(env, model, episodes=10):
    # test agent
    for episode in range(1, episodes+1):
        observation = env.reset()
        terminated = False
        truncated = False
        score = 0
        
        while not (terminated or truncated):
            env.render()
            action, _ = model.predict(observation)
            observation, reward, terminated, info = env.step(action)
            score += reward
            
        print(f'Episode: {episode} Score: {score}')
        
    env.close()

def main():
    # load environment
    environment_name = 'CartPole-v1'
    env = gym.make(environment_name, render_mode='human')

    # print environment info
    print(f'Environment: {environment_name}')
    print(f'Action Space: {env.action_space}')
    print(f'Observation Space: {env.observation_space}\n')

    # define log path
    log_path = os.path.join('Training', 'Logs')

    # create model
    env = DummyVecEnv([lambda: env])
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

    # train agent
    print('Training agent...')
    train_agent(model, timesteps=20000)

    # check if you want to continue training from user input
    cont = True
    while cont:
        if input('Continue training? (y/n): ') == 'y':
            train_agent(model, timesteps=20000)
        else:
            cont = False

    # evaluate agent
    print('Evaluating agent...')
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f'Mean Reward: {mean_reward} Std Reward: {std_reward}')

    # test agent
    print('Testing agent...')
    test_agent(env, model, episodes=10)

    # save model
    print('Saving model...')
    path = os.path.join('Training', 'Saved Models', 'PPO_Model_CartPole')
    model.save(path)

if __name__ == "__main__":
    main()