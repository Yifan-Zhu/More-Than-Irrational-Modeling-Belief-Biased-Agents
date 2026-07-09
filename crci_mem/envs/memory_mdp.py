from typing import Iterable, Optional, TypeVar
import numpy as np
from crci_mem.envs.core import Actions, Belief
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.world_object import Ball, Key, Wall, WorldObj
from minigrid.core.constants import OBJECT_TO_IDX, TILE_PIXELS
import gymnasium as gym
T = TypeVar("T")

class MemoryDecayMDP(gym.Env):
    """
    2D grid world memory game environment for CRCI with decay; Fully observable version of MemoryDecayExpEnv.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
        }
    
    def __init__(
        self,
        grid_size: int = 9,
        corridor_length: int = 3,
        agent_view_size: int = 3,
        see_through_walls: bool = False,
        max_steps: int = 100,
        decay_rate: float = 0.9,
        render_mode: str = "rgb_array",
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
        **kwargs,
    ):
        
        assert grid_size % 2 == 1
        self.num_tgt = 2
        self.grid_size = grid_size
        self.corridor_length = corridor_length
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls
        self.decay_rate = decay_rate
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size
        self.render_mode = render_mode

        # state: {agent_pos, target}
        # grid state: {0: empty, 1: wall, 2: left_object, 3: right_object}
        self.grid = Grid(grid_size, grid_size)
        self.agent_pos = None
        self.agent_dir = None
        self.target = None

        self.actions = Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.reward_range = (0, 1)
        # obs: full grid state and agent position
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=int),
            "target": spaces.Discrete(3), # target or None
        })

        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_count = 0
        np.random.seed(seed)
        self._gen_grid(self.grid_size, self.corridor_length)
        s = self._get_state()
        return s, {}
    
    def _get_state(self):
        state = {
            "agent_pos": self.agent_pos,
            "target": self.target_dir,
        }
        return state

    def step(self, action):
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False

        reward, terminated = self._move(action)

        if self.step_count >= self.max_steps:
            truncated = True

        s = self._get_state()

        return s, reward, terminated, truncated, {}
    
    def _move(self, action):
        # to be improved
        reward = 0
        terminated = False
        if action == self.actions.left:
            fwd_pos = self.agent_pos + np.array([-1, 0])
            fwd_cell = self.grid.get(*fwd_pos)
            self.agent_dir = 2
            if fwd_cell is None or self.can_overlap(fwd_cell):
                self.agent_pos = fwd_pos
            if tuple(self.agent_pos) == self.success_pos:
                self.agent_pos = fwd_pos
                reward = self._reward()
                terminated = True
            if tuple(self.agent_pos) == self.failure_pos:
                self.agent_pos = fwd_pos
                reward = 0
                terminated = True
        elif action == self.actions.right:
            fwd_pos = self.agent_pos + np.array([1, 0])
            fwd_cell = self.grid.get(*fwd_pos)
            self.agent_dir = 0
            if fwd_cell is None or self.can_overlap(fwd_cell):
                self.agent_pos = fwd_pos
                
            if tuple(self.agent_pos) == self.success_pos:
                self.agent_pos = fwd_pos
                reward = self._reward()
                terminated = True
            if tuple(self.agent_pos) == self.failure_pos:
                self.agent_pos = fwd_pos
                reward = 0
                terminated = True
        elif action == self.actions.up:
            fwd_pos = self.agent_pos + np.array([0, -1])
            fwd_cell = self.grid.get(*fwd_pos)
            self.agent_dir = 3
            if fwd_cell is None or self.can_overlap(fwd_cell):
                self.agent_pos = fwd_pos
                
            if tuple(self.agent_pos) == self.success_pos:
                self.agent_pos = fwd_pos
                reward = self._reward()
                terminated = True
            if tuple(self.agent_pos) == self.failure_pos:
                self.agent_pos = fwd_pos
                reward = 0
                terminated = True
        elif action == self.actions.down:
            fwd_pos = self.agent_pos + np.array([0, 1])
            fwd_cell = self.grid.get(*fwd_pos)
            self.agent_dir = 1
            if fwd_cell is None or self.can_overlap(fwd_cell):
                self.agent_pos = fwd_pos
                
            if tuple(self.agent_pos) == self.success_pos:
                self.agent_pos = fwd_pos
                reward = self._reward()
                terminated = True
            if tuple(self.agent_pos) == self.failure_pos:
                self.agent_pos = fwd_pos
                reward = 0
                terminated = True
        return reward, terminated
    
    def can_overlap(self, cell):
        return cell is None or cell.type in ["ball", "key"]
    
    def _reward(self):
        return 1 - 0.9 * (self.step_count / self.max_steps)
    
    def _gen_grid(self, grid_size, corridor_length):
        self.grid = Grid(grid_size, grid_size)
        self.grid.wall_rect(0, 0, grid_size, grid_size)
        self.start_size = 3
        assert grid_size >= self.start_size + corridor_length + 3
        left_room_wall = grid_size // 2 - 2
        right_room_wall = grid_size // 2 + 2
        for i in range(1, self.start_size + 2):
            self.grid.set(left_room_wall, grid_size - i - 1, Wall())
            self.grid.set(right_room_wall, grid_size - i - 1, Wall())
        self.grid.set(left_room_wall + 1, grid_size - self.start_size - 2, Wall())
        self.grid.set(right_room_wall - 1, grid_size - self.start_size - 2, Wall())
        for i in range(self.start_size + 2, self.start_size + 2 + corridor_length - 1):
            self.grid.set(left_room_wall + 1, grid_size - i - 1, Wall())
            self.grid.set(right_room_wall - 1, grid_size - i - 1, Wall())
        for j in range(0, grid_size):
            if j != grid_size // 2:
                self.grid.set(j, grid_size - 1 - self.start_size - corridor_length, Wall())
            self.grid.set(j, grid_size - 1 - self.start_size - corridor_length - 2, Wall())
        
        self.agent_pos = np.array((grid_size // 2, self._rand_int(2, grid_size - 4)))
        self.agent_dir = 3 # up
        self.target_obj = self._rand_elem([Key, Ball])
        self.target_pos = (grid_size // 2 - 1, grid_size - 3)
        self.grid.set(grid_size // 2 - 1, grid_size - 3, self.target_obj("green"))
        self.other_objs = self._rand_elem([[Ball, Key], [Key, Ball]])
        left_pos = (grid_size // 2 - 1, grid_size - 1 - self.start_size - corridor_length - 1)
        right_pos = (grid_size // 2 + 1, grid_size - 1 - self.start_size - corridor_length - 1)
        self.grid.set(*left_pos, self.other_objs[0]("green"))
        self.grid.set(*right_pos, self.other_objs[1]("green"))
        if self.target_obj == self.other_objs[0]:
            self.target_dir = 1
            self.success_pos = (left_pos[0], left_pos[1])
            self.failure_pos = (right_pos[0], right_pos[1])
        else:
            self.target_dir = 2
            self.success_pos = (right_pos[0], right_pos[1])
            self.failure_pos = (left_pos[0], left_pos[1])

    def render(self, model=None):
        img = self.get_full_render(self.tile_size)
        if self.render_mode == "rgb_array":
            return img
    
    def get_full_render(self, tile_size):

        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir, 
            highlight_mask=None,
        )

        return img
    
    def pprint_grid(self, grid: Grid = None):
        # print the grid and agent and object
        TopX = self.agent_pos[0] - self.agent_view_size // 2
        TopY = self.agent_pos[1] - self.agent_view_size // 2
        if grid is None:
            grid = self.grid
            grid_size = self.grid_size
        else:
            grid_size = grid.width
        if self.agent_pos is None or self.grid is None:
            raise ValueError(
                "The environment hasn't been `reset` therefore the `agent_pos`, `agent_dir` or `grid` are unknown."
            )
        OBJECT_TO_STR = {
            "wall": "W",
            "key": "K",
            "ball": "B",
            "agent": "A",
        }
        output = ""
        for j in range(grid_size):
            for i in range(grid_size):
                if grid != self.grid and (i + TopX, j + TopY) == tuple(self.agent_pos):
                    output += OBJECT_TO_STR["agent"]
                elif grid == self.grid and (i, j) == tuple(self.agent_pos):
                    output += OBJECT_TO_STR["agent"]
                else:
                    cell = grid.get(i, j)
                    if cell is None:
                        output += " "
                    else:
                        output += OBJECT_TO_STR[cell.type]
                output += " "
            output += "\n"
        return output
    
    def _rand_elem(self, iterable: Iterable[T]) -> T:
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]
    
    def _rand_int(self, low: int, high: int) -> int:
        """
        Generate random integer in [low,high[
        """

        return self.np_random.integers(low, high)





