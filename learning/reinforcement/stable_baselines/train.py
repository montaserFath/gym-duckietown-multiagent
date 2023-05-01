"""
Train an agent using stable_baselines 3
"""
import pyglet
# from PIL import Image
from pyglet.window import key

import os
import numpy as np
from stable_baselines3 import DQN, A2C, PPO, DDPG
# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
# from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator
# from gym_duckietown.wrappers import DiscreteWrapper, ResizeWrapper, NormalizeWrapper, ImgWrapper, RewardCropWrapper
from learning.utils.wrappers import ResizeWrapper, NormalizeWrapper, ImgWrapper, DtRewardWrapper, RewardCropWrapper, DiscreteWrapper, CustomCNN


def main(evaluation: bool = False, training_steps: int = int(1e5), render: bool = False, alg: str = "ppo", cnn: bool = False):
    """main function"""
    env = Simulator(
        frame_rate=15,
        frame_skip=2,
        map_name="4way",
        distortion=False,
        camera_rand=False,
        dynamics_rand=False,
        add_actions_noise=False,
    )

    env = ResizeWrapper(env, crop_top=True)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    # env = DiscreteWrapper(env)
    env = DtRewardWrapper(env)
    # env = RewardCropWrapper(env)
    # env = DummyVecEnv([lambda: env])
    env.reset()
    if evaluation:
        env.render()
        # agent = DQN.load("learning/reinforcement/stable_baselines/4ways_img/models/best_model.zip")
        if alg == "a2c":
            agent = A2C.load("models/resized_img_clip_reward_continuous_action/models/best_model.zip")
        elif alg =="ppo":
            agent = PPO.load("models/cnn_fixed_continuous_actions/models/best_model.zip")
        elif alg == "ddpg":
            agent = DDPG.load("models/cnn_fixed_continuous_actions_trim_reward_ddpg/models/best_model.zip")
        else:
            agent = DQN.load("models/resized_img_clip_reward_continuous_action/models/best_model.zip")
        evaluate(env, agent, render=render)
    else:
        train(env, "cnn_fixed_continuous_actions_trim_reward_ddpg", training_steps, alg=alg, cnn=cnn)


def train(env, model_dir: str, training_steps: int = int(1e5), alg: str = "ppo", cnn: bool = False):
    os.makedirs("models/" + model_dir + "/tensorboard/", exist_ok=True)
    os.makedirs("models/" + model_dir + "/models/", exist_ok=True)
    os.makedirs("models/" + model_dir + "/logs/", exist_ok=True)
    if cnn:
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=64),
        )
    else:
        policy_kwargs = dict(net_arch=[128, 128, 128])
    if alg == "a2c":
        model = A2C("CnnPolicy" if cnn else "MlpPolicy",
                    env,
                    verbose=1,
                    learning_rate=1e-3,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log="models/" + model_dir + "/tensorboard/",
                    )
    elif alg == "ddpg":
        model = DDPG("CnnPolicy" if cnn else "MlpPolicy",
                     env,
                     verbose=1,
                     buffer_size=2000,
                     learning_rate=1e-3,
                     policy_kwargs=policy_kwargs,
                     tensorboard_log="models/" + model_dir + "/tensorboard/",
                     )
    elif alg == "ppo":
        model = PPO("CnnPolicy" if cnn else "MlpPolicy",
                    env,
                    verbose=1,
                    learning_rate=1e-3,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log="models/" + model_dir + "/tensorboard/",
                    )
    else:
        model = DQN("CnnPolicy" if cnn else "MlpPolicy",
                    env,
                    verbose=1,
                    buffer_size=1000,
                    learning_rate=1e-3,
                    batch_size=128,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log="models/" + model_dir + "/tensorboard/",
                    learning_starts=5000,
                    )
    callback = EvalCallback(env,
                            best_model_save_path="models/" + model_dir + "/models/",
                            log_path="models/" + model_dir + "/logs/",
                            eval_freq=int(1e3),
                            )
    model.learn(total_timesteps=training_steps, callback=callback)
    model.save("models/" + model_dir + "/models/final_model")


def evaluate(env, model, eval_ep: int = 10, render: bool = False):
    for ep in range(eval_ep):
        if render:
            env.render()
        state = env.reset()
        done = False
        time_step = 0
        while not done:
            # np.save("models/debugging/state_ep_{}_t_{}".format(ep, time_step), state)
            # action = np.array([env.action_space.sample()])
            action, _ = model.predict(state)
            if render:
                pyglet.clock.schedule_interval(env_step(env, action, render), 1.0 / 1)
            state, reward, done = env_step(env, action)
            time_step += 1
        print("Reset")
    env.close()


def env_step(env, action, render=False):
    if render:
        env.render()
    state, reward, done, _ = env.step(action)
    return state, reward, done


if __name__ == "__main__":
    main(evaluation=True, training_steps=int(1e7), render=True, alg="ddpg", cnn=True)
    # Enter main event loop
    # pyglet.app.run()