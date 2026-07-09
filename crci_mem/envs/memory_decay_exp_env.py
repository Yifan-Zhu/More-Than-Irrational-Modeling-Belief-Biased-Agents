from typing import Iterable, Optional, TypeVar
import numpy as np
from crci_mem.envs.core import Actions, Belief
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.world_object import Ball, Key, Wall, WorldObj
from minigrid.core.constants import OBJECT_TO_IDX, TILE_PIXELS
import gymnasium as gym
T = TypeVar("T")

class MemoryDecayExpEnv(gym.Env):
    """
    2D grid world memory game environment for CRCI with decay; The initial observation only includes the agent's surrounding grid, exploration on target is needed.
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
        #self.use_belief = use_belief
        self.render_mode = render_mode

        # state: {agent_pos, target}
        # grid state: {0: empty, 1: wall, 2: left_object, 3: right_object}
        self.grid = Grid(grid_size, grid_size)
        self.agent_pos = None
        self.agent_dir = None
        self.target = None
        # view window centered on the agent: agent_view_size//2 cells each side (orientation-agnostic)

        self.actions = Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.reward_range = (0, 1)
        # obs: position of agent, target object
        self.observation_space = spaces.Dict({
            "surrounding": spaces.Box(low=0, high=4, shape=(agent_view_size, agent_view_size, 3), dtype=int),
            "target": spaces.Discrete(3), # target or None
        })
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov

        self.reset()

    def transition_prob(self, x_prev, y_prev, action):
        trans = []
        if action == self.actions.up:
            x_n = x_prev
            y_n = max(0, y_prev-1)
            trans.append((x_n, y_n, 1.0))
        elif action == self.actions.down:
            x_n = x_prev
            y_n = min(self.grid_size-1, y_prev+1)
            trans.append((x_n, y_n, 1.0))
        elif action == self.actions.left:
            x_n = max(0, x_prev-1)
            y_n = y_prev
            trans.append((x_n, y_n, 1.0))
        elif action == self.actions.right:
            x_n = min(self.grid_size-1, x_prev+1)
            y_n = y_prev
            trans.append((x_n, y_n, 1.0))
        else:
            trans.append((x_prev, y_prev, 1.0))
        return trans

    def _rand_elem(self, iterable: Iterable[T]) -> T:
        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]
    
    def _rand_int(self, low: int, high: int) -> int:
        return self.np_random.integers(low, high)
    
    def _get_state(self):
        state = {
            "agent_pos": self.agent_pos,
            "target": self.target_dir,
        }
        return state
    
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_count = 0
        np.random.seed(seed)
        self._gen_grid(self.grid_size, self.corridor_length)
        obs = self._get_obs()
        return obs, {}
    
    def _get_view_exts(self, x=None, y=None):
        if x == None and y == None:
            agent_pos = self.agent_pos
        else:
            agent_pos = np.array((x, y))
        topX = agent_pos[0] - self.agent_view_size // 2
        topY = agent_pos[1] - self.agent_view_size // 2
        botX = topX + self.agent_view_size
        botY = topY + self.agent_view_size
        return topX, topY, botX, botY

    def _check_target_view(self, x=None, y=None):
        topX, topY, botX, botY = self._get_view_exts(x=x, y=y)
        return topX <= self.target_pos[0] < botX and topY <= self.target_pos[1] < botY
    
    def _see_behind(self, cell: WorldObj):
        if cell.type == "wall":
            return False
        return True
    
    def _process_vis(self, view_pos: tuple[int, int], obs_grid: Grid) -> np.ndarray:
        grid_size = obs_grid.width
        mask = np.zeros(shape=(grid_size, grid_size), dtype=bool)
        mask[view_pos[0], view_pos[1]] = True
        ax, ay = view_pos

        for j in reversed(range(0, ay)):
            mask[ax, j] = True
            cell = obs_grid.get(ax, j)
            if cell and self._see_behind(cell):
                continue
            
    def _get_obs(self, s=None):
        # use grid.slice to obtain a view of the agent
        if s is None:
            s = self.agent_pos
        else:
            s = s['agent_pos']
        topX, topY, botX, botY = self._get_view_exts(x=s[0], y=s[1])
        obs_grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)
        
        """
        if not self.see_through_walls:
            vis_mask = obs_grid.process_vis(
                agent_pos=(self.agent_view_size // 2, self.agent_view_size // 2)
            )
        else:
            vis_mask = np.ones(shape=(obs_grid.width, obs_grid.height), dtype=bool)
        """
        obs_grid_masked = obs_grid.encode()
        
        # if the agent can observe the target object, give the target object as observation, otherwise give none
        # check if the target object is in the view
        if self._check_target_view(x=s[0], y=s[1]):
            # check target position in the view
            """
            target_pos_in_view = (self.target_pos[0] - topX, self.target_pos[1] - topY)
            if vis_mask[target_pos_in_view[0], target_pos_in_view[1]]:
                target = OBJECT_TO_IDX[self.target_obj("green").type]
            else: 
                target = None
            """
            #target = OBJECT_TO_IDX[self.target_obj("green").type] # 1: left object, 2: right object
            target = self.target_dir
        else:
            target = 0
        obs = {
            "surrounding": obs_grid_masked,
            "target": target,
        }
        #print("obs_grid")
        #print(self.pprint_grid(obs_grid))
        #print(obs)
        return obs

    def step(self, action):
        """
        if self.use_belief:
            self._update_belief(action, self._get_obs())
        """
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False

        reward, terminated = self._move(action)

        if self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()

        return obs, reward, terminated, truncated, {}

    def _move(self, action):
        # to be improved
        reward = 0
        terminated = False
        if action == self.actions.left:
            fwd_pos = self.agent_pos + np.array([-1, 0])
            fwd_cell = self._cell_at(fwd_pos)
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
            fwd_cell = self._cell_at(fwd_pos)
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
            fwd_cell = self._cell_at(fwd_pos)
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
            fwd_cell = self._cell_at(fwd_pos)
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
    
    def _cell_at(self, pos):
        # Out-of-bounds behaves like the walled border: impassable. minigrid>=3 asserts on
        # grid.get() out of range, whereas the version this env was written against tolerated it;
        # guard so random exploration during training cannot step off the grid and crash.
        x, y = int(pos[0]), int(pos[1])
        if x < 0 or y < 0 or x >= self.grid.width or y >= self.grid.height:
            return Wall()
        return self.grid.get(x, y)

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
        #self.agent_pos = np.array((grid_size // 2, grid_size - 2))
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

    def _update_belief(self, action, obs):
        pass

    def render(self, model=None):
        img = self._get_frame(self.highlight, self.tile_size, self.agent_pov)
        if self.render_mode == "rgb_array":
            return img
        

    def render_grid(self, tile_size: int = None, highlight_mask = None) -> np.ndarray:
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.grid_size, self.grid_size), dtype=bool)

        if tile_size is None:
            tile_size = self.tile_size

        # Compute the total grid size
        width_px = self.grid_size * tile_size
        height_px = self.grid_size * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.grid_size):
            for i in range(0, self.grid_size):
                cell = self.grid.get(i, j)

                assert highlight_mask is not None
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img


    

    def _get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:

            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        

        if agent_pov:
            return self.get_pov_render(tile_size)
        else:
            return self.get_full_render(highlight, tile_size)
        """
        return self.get_full_render(highlight, tile_size)
        
    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        """
        # Compute which cells are visible to the agent
        #_, vis_mask = self.gen_obs_grid()
        vis_mask = np.ones(shape=(self.agent_view_size, self.agent_view_size), dtype=bool)

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        top_left = self.agent_pos - np.array((self.agent_view_size // 2, self.agent_view_size // 2))

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.grid_size, self.grid_size), dtype=bool)

        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left + np.array((vis_i, vis_j))

                if abs_i < 0 or abs_i >= self.grid_size:
                    continue
                if abs_j < 0 or abs_j >= self.grid_size:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir, 
            highlight_mask=highlight_mask if highlight else None,
        )

        return img
    


    def gen_obs_grid(self, x=None, y=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """
        if x == None and y == None:
            topX, topY, botX, botY = self._get_view_exts()
            grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

            return grid, vis_mask
        else:
            if x == None or y == None:
                raise ValueError("x and y must be both None or both int")
            topX, topY, botX, botY = self._get_view_exts(x, y)
            grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)
            grid = grid.encode()
            return grid
        

