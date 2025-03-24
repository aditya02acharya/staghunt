import random

import numpy as np
import pygame
from gymnasium.spaces import Box, Dict, Discrete, Tuple
from pettingzoo import ParallelEnv

# Reward Constants
STAG_REWARD = 5
HARE_REWARD = 1
STAG_CAPTURE_POWER = 2
CELL_SIZE = 50


class StagHuntEnv(ParallelEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size=7, num_players=2):
        """
        Initialize the Stag Hunt environment.

        Args:
            grid_size (int): Size of the grid
            num_players (int): Number of agents
        """
        if num_players < 2 or num_players > 4:
            raise ValueError("Number of agents must be between 2 and 4 (inclusive)")
        self.grid_size = grid_size
        self.num_players = num_players
        self.agents = [f"agent_{i}" for i in range(num_players)]
        self.possible_agents = self.agents[:]

        # Define the action space for each agent:
        #   movement: Discrete(5) (0: no-op, 1: up, 2: right, 3: down, 4: left)
        #   communication recipient: Discrete(num_players+1) where values 0..num_players-1 select a specific agent,
        #                               and value num_players indicates a broadcast.
        #   communication target: Discrete(3) (0: no communication, 1: target stag, 2: target hare)
        self.action_space = Tuple((Discrete(5), Discrete(self.num_players + 1), Discrete(3)))

        # Define observation space as a dict with two keys:
        # "grid": the grid observation (as before)
        # "comms": a vector of length num_players with values in {0,1,2} (-1: hare, 0: none, 1: stag)
        self.observation_space = Dict({
            "grid": Box(low=0, high=1, shape=(self.grid_size, self.grid_size, self.num_players + 2), dtype=np.int8),
            "comms": Box(low=-1, high=1, shape=(self.num_players,), dtype=np.int8),
        })

        self._reset_internal_state()

    def _reset_internal_state(self):
        """
        Initializes positions and resets communication messages.
        """
        self.step_count = 0
        self.agent_positions = {}
        # Reset communication messages (store tuple: (recipient, target) for each agent)
        self._comm_messages = {agent: (None, 0) for agent in self.agents}

        # Randomly assign unique starting positions
        all_cells = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        random.shuffle(all_cells)
        for agent in self.agents:
            self.agent_positions[agent] = all_cells.pop()
        # Place stag and hare
        self.stag_pos = all_cells.pop()
        self.hare_pos = all_cells.pop()

        # Set capture power for the stag (between 2 and num_players)
        self.stag_capture_power = STAG_CAPTURE_POWER

    def reset(self, seed=None, options=None):
        """Resets the environment and returns observations for all agents."""
        self._reset_internal_state()
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        return observations

    def reset_stag(self):
        """Reset stag to a new location (avoiding positions occupied by agents)."""
        all_cells = [
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if (i, j) not in self.agent_positions.values() or (i, j) != self.hare_pos or (i, j) != self.stag_pos
        ]
        self.stag_pos = random.choice(all_cells)
        return self.stag_pos

    def reset_hare(self):
        """Reset hare to a new location (avoiding positions occupied by agents)."""
        all_cells = [
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if (i, j) not in self.agent_positions.values() or (i, j) != self.hare_pos or (i, j) != self.stag_pos
        ]
        self.hare_pos = random.choice(all_cells)
        return self.hare_pos

    def step(self, actions):
        """
        Parameters:
            actions (dict): Mapping from agent name to an action tuple:
                            (movement_action, comm_recipient, comm_target)
        Returns:
            observations (dict): New observations for each agent.
            rewards (dict): Rewards for each agent.
            dones (dict): Always False (continuous environment).
            infos (dict): Additional info for each agent.
        """
        rewards = {agent: 0.0 for agent in self.agents}

        # First, record communication messages for each agent.
        # Each message is a tuple: (recipient, target)
        # recipient: an integer in [0, num_players] where num_players means broadcast.
        # target: 0 (no communication), 1 (stag), or 2 (hare)
        for agent, action in actions.items():
            # Unpack action tuple
            move_action, comm_recipient, comm_target = action
            self._comm_messages[agent] = (comm_recipient, comm_target - 1)  # Convert to 0-indexed target

            # Update agent positions based on movement action.
            cur_x, cur_y = self.agent_positions[agent]
            new_x, new_y = cur_x, cur_y
            if move_action == 1:  # Up
                new_x = max(cur_x - 1, 0)
            elif move_action == 2:  # Right
                new_y = min(cur_y + 1, self.grid_size - 1)
            elif move_action == 3:  # Down
                new_x = min(cur_x + 1, self.grid_size - 1)
            elif move_action == 4:  # Left
                new_y = max(cur_y - 1, 0)
            # move_action == 0 is no-op.
            self.agent_positions[agent] = (new_x, new_y)

        # Check for stag capture:
        agents_on_stag = [agent for agent, pos in self.agent_positions.items() if pos == self.stag_pos]
        if len(agents_on_stag) == self.stag_capture_power:
            # Only if exactly the required number of agents are present
            for agent in agents_on_stag:
                rewards[agent] += STAG_REWARD
            # Reset stag to a new location (avoiding positions occupied by agents)
            self.reset_stag()
        elif len(agents_on_stag) > self.stag_capture_power:
            # Overcrowding: no rewards are given to any agent on stag.
            for agent in agents_on_stag:
                rewards[agent] = 0
            # Reset stag to a new location (avoiding positions occupied by agents)
            self.reset_stag()

        # Check for hare capture
        agents_on_hare = [agent for agent, pos in self.agent_positions.items() if pos == self.hare_pos]
        if agents_on_hare:
            for agent in agents_on_hare:
                rewards[agent] += HARE_REWARD / len(agents_on_hare)
            # Reset hare to a new location (avoiding positions occupied by agents)
            self.reset_hare()

        self.step_count += 1

        # Construct observations for each agent
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        dones = {agent: False for agent in self.agents}  # Continuous environment: always False
        infos = {agent: {"stag_capture_power": self.stag_capture_power} for agent in self.agents}
        return observations, rewards, dones, infos

    def _get_obs(self, agent):
        """
        Constructs the observation for a given agent.
        Returns a dict with two keys:
          "grid": the (grid_size, grid_size, num_players+2) grid observation.
          "comms": a vector of length num_players containing the communicated targets
                   from each agent (0 if no message or not addressed to this agent).
                   For each other agent j, if that agent's communication message has a target != 0 and
                   (its recipient equals this agent's index or is set to broadcast),
                   then the observed value is that target (1: stag, -1: hare), else 0.
        """
        # Grid observation as before.
        grid_obs = np.zeros((self.grid_size, self.grid_size, self.num_players + 2), dtype=np.int8)
        for i, agent_name in enumerate(self.agents):
            x, y = self.agent_positions[agent_name]
            grid_obs[x, y, i] = 1
        # Mark stag.
        stag_x, stag_y = self.stag_pos
        grid_obs[stag_x, stag_y, self.num_players] = 1

        # Mark hare.
        hare_x, hare_y = self.hare_pos
        grid_obs[hare_x, hare_y, self.num_players + 1] = 1

        # Communication observation:
        # For each agent in self.agents, if that agent's communication is either broadcast (recipient==num_players)
        # or directed specifically to the current agent, then include its communicated target.
        comms_obs = np.zeros(self.num_players, dtype=np.int8)
        # Get current agent's index.
        cur_index = int(agent.split("_")[1])
        for other_agent in self.agents:
            other_index = int(other_agent.split("_")[1])
            # We include messages only from other agents, is it worth having self reflection and why?
            if other_agent == agent:
                continue
            recipient, target = self._comm_messages[other_agent]
            # If target is 0, then there was no communication.
            # Otherwise, if recipient is broadcast (i.e. equals self.num_players) or equals current agent index,
            # then this message is visible.
            if target != 0 and (recipient == self.num_players or recipient == cur_index):
                comms_obs[other_index] = target
            else:
                comms_obs[other_index] = 0
        return {"grid": grid_obs, "comms": comms_obs}

    def render(self, mode="human"):
        """Simple pygame-based rendering of the grid and current communications."""
        if not hasattr(self, "screen"):
            pygame.init()
            width = self.grid_size * CELL_SIZE
            height = self.grid_size * CELL_SIZE + 50  # extra space for info text
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Stag Hunt Environment")
            self.font = pygame.font.SysFont("Arial", 20)
        else:
            width = self.grid_size * CELL_SIZE
            height = self.grid_size * CELL_SIZE + 50  # extra space for info text

        # Fill background
        self.screen.fill((255, 255, 255))

        # Draw grid lines
        for i in range(self.grid_size):
            # Horizontal lines
            pygame.draw.line(self.screen, (200, 200, 200), (0, i * CELL_SIZE), (width, i * CELL_SIZE))
            # Vertical lines
            pygame.draw.line(
                self.screen, (200, 200, 200), (i * CELL_SIZE, 0), (i * CELL_SIZE, self.grid_size * CELL_SIZE)
            )

        # Draw stag as a red rectangle
        stag_x, stag_y = self.stag_pos
        stag_rect = pygame.Rect(stag_y * CELL_SIZE, stag_x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, (150, 75, 0), stag_rect)

        # Draw hare as a green rectangle
        hare_x, hare_y = self.hare_pos
        hare_rect = pygame.Rect(hare_y * CELL_SIZE, hare_x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, (0, 255, 0), hare_rect)

        # Draw agents as circles with unique colors
        agent_colors = [(0, 0, 255), (255, 0, 0), (255, 165, 0), (128, 0, 128), (0, 128, 128)]
        for idx, agent in enumerate(self.agents):
            x, y = self.agent_positions[agent]
            # Calculate center of the cell
            center = (y * CELL_SIZE + CELL_SIZE // 2, x * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(self.screen, agent_colors[idx % len(agent_colors)], center, CELL_SIZE // 3)

            # Draw agent id (using the numeric part of the agent name)
            text = self.font.render(agent.split("_")[1], True, (255, 255, 255))
            text_rect = text.get_rect(center=center)
            self.screen.blit(text, text_rect)

        # Render additional info such as stag capture power below the grid (starting with capture power).
        info_text = f"Stag Capture Power: {self.stag_capture_power}"
        info_surface = self.font.render(info_text, True, (0, 0, 0))
        self.screen.blit(info_surface, (10, self.grid_size * CELL_SIZE + 10))

        # Handle events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        pygame.display.flip()

    def close(self):
        pass


if __name__ == "__main__":
    env = StagHuntEnv(grid_size=7, num_players=2)
    observations = env.reset()
    env.render()

    # Run 10 steps; for demonstration, actions are randomly selected.
    for _ in range(10):
        actions = {}
        for agent in env.agents:
            # Randomly sample movement action.
            move = env.action_space.spaces[0].sample()
            # Randomly sample communication recipient (0..num_players, where num_players means broadcast)
            comm_recipient = env.action_space.spaces[1].sample()
            # Randomly sample communication target (0: no communication, 1: stag, -1: hare)
            comm_target = env.action_space.spaces[2].sample()
            actions[agent] = (move, comm_recipient, comm_target)
        observations, rewards, dones, infos = env.step(actions)
        print("Actions:", actions)
        print("Rewards:", rewards)
        env.render()
        pygame.time.wait(1000)
