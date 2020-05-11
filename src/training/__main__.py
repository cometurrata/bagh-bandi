import time
from tensorboardX import SummaryWriter

from src.agents.dqnagent import DQNAgent
from src.infrastructure.settings import *

name = '_'.join([str(k)+'.'+str(v) for k,v in DQN_HYPERPARAMS.items()])
name = 'prv'

def main():
    writer = SummaryWriter(log_dir=LOG_DIR + '/' + name + str(time.time())) if SUMMARY_WRITER else None

    print('Hyperparams:', DQN_HYPERPARAMS)

    # create the agent
    agent = DQNAgent(agent_type, engine, device=DEVICE, summary_writer=writer, hyperparameters=DQN_HYPERPARAMS)

    n_games = 0
    n_iter = 0

    # Play MAX_N_GAMES games
    while n_games < MAX_N_GAMES:
        # act greedly
        action = agent.act_eps_greedy(obs)

        # one step on the environment
        new_obs, reward, done, _ = env.step(action)

        # add the environment feedback to the agent
        agent.add_env_feedback(obs, action, new_obs, reward, done)

        # sample and optimize NB: the agent could wait to have enough memories
        agent.sample_and_optimize(BATCH_SIZE)

        obs = new_obs
        if done:
            n_games += 1

            # print info about the agent and reset the stats
            agent.print_info()
            agent.reset_stats()

            # if n_games % TEST_FREQUENCY == 0:
            #	print('Test mean:', utils.test_game(env, agent, 1))

            obs = env.reset()

    writer.close()


# tensorboard --logdir content/runs --host localhost


if __name__ == '__main__':
    main()