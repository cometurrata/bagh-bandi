DQN_HYPERPARAMS = {
    'dueling': False,
    'noisy_net': False,
    'double_DQN': False,
    'n_multi_step': 2,
    'buffer_start_size': 10001,
    'buffer_capacity': 15000,
    'epsilon_start': 1.0,
    'epsilon_decay': 10 ** 5,
    'epsilon_final': 0.02,
    'learning_rate': 5e-5,
    'gamma': 0.99,
    'n_iter_update_target': 1000
}


BATCH_SIZE = 32
MAX_N_GAMES = 3000
TEST_FREQUENCY = 10

DEVICE = 'cpu'  # or 'cuda'
SUMMARY_WRITER = True

LOG_DIR = 'data/content/runs'