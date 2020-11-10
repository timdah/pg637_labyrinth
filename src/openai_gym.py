from src.dqn import train_dqn_gym


def dqn_cart_pole():
    train_dqn_gym(env_name='CartPole-v0',
                  lr=0.001,
                  rb_size=5000,
                  max_frames=10000,
                  start_train_frame=32,
                  epsilon_start=1.0,
                  epsilon_end=0.01,
                  epsilon_decay=7000,
                  batch_size=32,
                  gamma=0.99,
                  target_network_update_freq=1000,
                  log_every=100)


if __name__ == '__main__':
    dqn_cart_pole()
