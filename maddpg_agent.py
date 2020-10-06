import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
from ddpg_agent import DDPGAgent
from replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 200  # minibatch size
GAMMA = 0.995  # discount factor
TAU = 1e-3  # for soft update of target parameters
WEIGHT_DECAY = 0  # L2 weight decay
UPDATE_EVERY_N_EPISODE = 4
UPDATE_REPLAY = 3


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, numAgent, random_seed, epsilon = 1, epsilonDecay = 0.995, minEpsilon = 0.00):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.minEpsilon = minEpsilon
        self.numAgent = numAgent
        self.agents = [DDPGAgent(state_size, action_size, random_seed) for i in range(numAgent)]
        self.sharedMemory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def prepareStepInput(self, actions, states, nextStates):
        actions = np.concatenate((actions[0], actions[1]), axis=1)
        states = np.expand_dims(np.hstack([states[0], states[1]]), axis=0)
        nextStates = np.expand_dims(np.hstack([nextStates[0], nextStates[1]]), axis=0)
        return actions, states, nextStates

    def step(self, states, actions, rewards, next_states, dones, currentEpisodeNumber):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        actions, states, next_states = self.prepareStepInput(actions, states, next_states)

        self.sharedMemory.add(states, actions, rewards, next_states, dones)
        # Learn, if enough samples are available in memory
        if len(self.sharedMemory) > BATCH_SIZE and (currentEpisodeNumber % UPDATE_EVERY_N_EPISODE ==0):
            for _ in range(UPDATE_REPLAY):
                for i in range(self.numAgent):
                    experiences = self.sharedMemory.sample()
                    self.learn(experiences, i, GAMMA)
    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = []
        for i, agent in enumerate(self.agents):
            actions.append(agent.act(states[i], self.epsilon, add_noise))

        return actions

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def learn(self, experiences, agentId, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        list_indices = torch.tensor([idx for idx in range(0, self.state_size)]).to(device)
        firstAgentStates = states.index_select(1, list_indices)
        list_indices = torch.tensor([idx for idx in range(self.state_size, self.state_size + self.state_size)]).to(device)
        secondAgentStates = states.index_select(1, list_indices)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        agent = self.agents[agentId]

        actionsNext = torch.cat((agent.actor_target(firstAgentStates), agent.actor_target(secondAgentStates)),dim=1).to(device)



        Q_targets_next = agent.critic_target(next_states, actionsNext).float().cpu().to(device)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)).to(device)
        # Compute critic loss
        Q_expected = agent.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        predictedActions = torch.cat([agent.actor_local(firstAgentStates).to(device), agent.actor_local(secondAgentStates).to(device)], dim=-1).to(device)
        actor_loss = -agent.critic_local(states, predictedActions).mean()
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(agent.critic_local, agent.critic_target, TAU)
        self.soft_update(agent.actor_local, agent.actor_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def saveNetwork(self):
        for i, agent in enumerate(self.agents):
            agentId = str(i + 1)
            torch.save(agent.actor_local.state_dict(), './pretrainedNetworks/actor_local' + agentId + '.pth')
            torch.save(agent.actor_target.state_dict(), './pretrainedNetworks/actor_target' + agentId + '.pth')
            torch.save(agent.critic_local.state_dict(), './pretrainedNetworks/critic_local' + agentId + '.pth')
            torch.save(agent.critic_target.state_dict(), './pretrainedNetworks/critic_target' + agentId + '.pth')

    def loadNetwork(self):
        for i, agent in enumerate(self.agents):
            agentId = str(i + 1)
            agent.actor_local.load_state_dict(torch.load('./pretrainedNetworks/actor_local' + agentId + '.pth'))
            agent.actor_target.load_state_dict(torch.load('./pretrainedNetworks/actor_target' + agentId + '.pth'))
            agent.critic_local.load_state_dict(torch.load('./pretrainedNetworks/critic_local' + agentId + '.pth'))
            agent.critic_target.load_state_dict(torch.load('./pretrainedNetworks/critic_target' + agentId + '.pth'))
