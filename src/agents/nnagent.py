import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.agent import Agent
from src.agents.dqn import DQN


class NNAgent(Agent):
    """
        Agent inspired from https://github.com/andri27-ts/Reinforcement-Learning/tree/master/Week3
    """

    def __init__(self, agent_type, engine, gamma, n_multi_step, device):
        """
        Initiation
        Parameters
        ----------
        observation_space_shape : int
            plop
        action_space_shape : int
            plop
        gamma : int
            plop
        n_multi_step : int
            plop
        device : string
        cpu or cuda to train on GPU
        """
        super(NNAgent, self).__init__(agent_type, engine)
        observation_space_shape = self.engine.board.
        self.target_nn = DQN(observation_space_shape, action_space_shape).to(device)
        self.moving_nn = DQN(observation_space_shape, action_space_shape).to(device)

        self.device = device
        self.gamma = gamma
        self.n_multi_step = n_multi_step

    def set_optimizer(self, learning_rate):
        self.optimizer = optim.Adam(self.moving_nn.parameters(), lr=learning_rate)

    def optimize(self, mini_batch):
        '''
        Optimize the NN
        '''
        # reset the grads
        self.optimizer.zero_grad()
        # caluclate the loss of the mini batch
        loss = self._calulate_loss(mini_batch)
        loss_v = loss.item()

        # do backpropagation
        loss.backward()
        # one step of optimization
        self.optimizer.step()

        return loss_v

    def update_target(self):
        '''
        Copy the moving NN in the target NN
        '''
        self.target_nn.load_state_dict(self.moving_nn.state_dict())
        self.target_nn = self.moving_nn

    def get_max_action(self, obs):
        '''
        Forward pass of the NN to obtain the action of the given observations
        '''
        # convert the observation in tensor
        state_t = torch.tensor(np.array([obs])).to(self.device)
        # forawrd pass
        q_values_t = self.moving_nn(state_t)
        # get the maximum value of the output (i.e. the best action to take)
        _, act_t = torch.max(q_values_t, dim=1)
        return int(act_t.item())

    def _calulate_loss(self, mini_batch):
        '''
        Calculate mini batch's MSE loss.
        It support also the double DQN version
        '''

        states, actions, next_states, rewards, dones = mini_batch

        # convert the data in tensors
        states_t = torch.as_tensor(states, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(dones, dtype=torch.uint8, device=self.device)

        # Value of the action taken previously (recorded in actions_v) in the state_t
        state_action_values = self.moving_nn(states_t).gather(1, actions_t[:, None]).squeeze(-1)
        # NB gather is a differentiable function

        # Next state value in the normal configuration
        next_state_values = self.target_nn(next_states_t).max(1)[0]

        next_state_values = next_state_values.detach()  # No backprop

        # Use the Bellman equation
        expected_state_action_values = rewards_t + (self.gamma ** self.n_multi_step) * next_state_values
        # compute the loss
        return nn.MSELoss()(state_action_values, expected_state_action_values)
