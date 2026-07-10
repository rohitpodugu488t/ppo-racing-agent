# Python version:
# Use Python 3.11.9
# (Python 3.11.x is the safest match for this repo.)

# Built-in Python modules used here, so no install is needed:
# argparse, csv, math, pathlib, random

# ==================== ENVIRONMENT SETUP (`environment.py`) ====================
# PowerShell terminal commands:
# py -3.11 -m venv .venv
# .\.venv\Scripts\Activate.ps1
# python -m pip install --upgrade pip
# python -m pip install swig==4.4.1 gymnasium==1.2.3 Box2D==2.3.10 pygame==2.6.1 numpy==2.4.4
#
# Main environment libraries:
# swig==4.4.1
# gymnasium==1.2.3
# Box2D==2.3.10
# pygame==2.6.1
# numpy==2.4.4
#
# Environment dependency packages pulled in by install:
# cloudpickle==3.1.2
# Farama-Notifications==0.0.4

# ==================== TRAINING SETUP (`training.py`) ====================
# PowerShell terminal commands:
# python -m pip install torch==2.11.0 matplotlib==3.10.8 numpy==2.4.4
#
# Full training install including the custom environment:
# python -m pip install swig==4.4.1 gymnasium==1.2.3 Box2D==2.3.10 pygame==2.6.1 numpy==2.4.4 torch==2.11.0 matplotlib==3.10.8
#
# Main training libraries:
# torch==2.11.0
# matplotlib==3.10.8
# numpy==2.4.4
#
# Torch dependency packages pulled in by install:
# filelock==3.28.0
# fsspec==2026.3.0
# jinja2==3.1.6
# MarkupSafe==3.0.3
# networkx==3.6.1
# sympy==1.14.0
# mpmath==1.3.0
# typing_extensions==4.15.0
#
# Matplotlib dependency packages pulled in by install:
# contourpy==1.3.3
# cycler==0.12.1
# fonttools==4.62.1
# kiwisolver==1.5.0
# packaging==26.1
# pillow==12.2.0
# pyparsing==3.3.2
# python-dateutil==2.9.0.post0
# six==1.17.0
#
# Note:
# train_agent.py imports CarRacing from test.py, so training also needs the environment packages above.

# ==================== TESTING SETUP (`testing.py`) ====================
# PowerShell terminal commands:
# No extra separate install is needed if environment + training packages are already installed.
#
# One-command install for testing:
# python -m pip install swig==4.4.1 gymnasium==1.2.3 Box2D==2.3.10 pygame==2.6.1 numpy==2.4.4 torch==2.11.0 matplotlib==3.10.8
#
# Main testing libraries:
# numpy==2.4.4
# torch==2.11.0
# matplotlib==3.10.8
# gymnasium==1.2.3
# Box2D==2.3.10
# pygame==2.6.1
# swig==4.4.1
#
# Note:
# test_agent.py imports helpers from train_agent.py, so matplotlib is needed indirectly too.

import math

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle

try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 60  # Frames per second
ZOOM = 2.3  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env.new_lap = True
        else:
            obj.tiles.remove(tile)


class CarRacing(gym.Env, EzPickle):
    """
    ## Description
    The easiest control task to learn from pixels - a top-down
    racing environment. The generated track is random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```shell
    python gymnasium/envs/box2d/car_racing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

    ## Action Space
    If continuous there are 3 actions :
    - 0: steering, -1 is full left, +1 is full right
    - 1: gas
    - 2: braking

    If discrete there are 5 actions:
    - 0: do nothing
    - 1: steer right
    - 2: steer left
    - 3: gas
    - 4: brake

    ## Observation Space

    A top-down 96x96 RGB image of the car and race track.

    ## Rewards
    The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles
     visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

    ## Starting State
    The car starts at rest in the center of the road.

    ## Episode Termination
    The episode finishes when all the tiles are visited. The car can also go outside the playfield -
     that is, far off the track, in which case it will receive -100 reward and die.

    ## Arguments

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v3>>>>>

    ```

    * `lap_complete_percent=0.95` dictates the percentage of tiles that must be visited by
     the agent before a lap is considered complete.

    * `domain_randomize=False` enables the domain randomized variant of the environment.
     In this scenario, the background and track colours are different on every reset.

    * `continuous=True` specifies if the agent has continuous (true) or discrete (false) actions.
     See action space section for a description of each.

    ## Reset Arguments

    Passing the option `options["randomize"] = True` will change the current colour of the environment on demand.
    Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment.
    `domain_randomize` must be `True` on init for this argument to work.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CarRacing-v3", domain_randomize=True)

    # normal reset, this changes the colour scheme by default
    >>> obs, _ = env.reset()

    # reset with colour scheme change
    >>> randomize_obs, _ = env.reset(options={"randomize": True})

    # reset with no colour scheme change
    >>> non_random_obs, _ = env.reset(options={"randomize": False})

    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = True,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
        )
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self._init_colors()

        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: pygame.Surface | None = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car: Car | None = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.prev_track_idx = 0
        self.prev_steer = 0
        self.verbose = verbose
        self.new_lap = False
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake
        else:
            self.action_space = spaces.Discrete(5)
            # do nothing, right, left, gas, brake

        self.num_rays = 4

        obs_dim = self.num_rays + 8   # path-angle cues + speed + heading + signed offset + corner + slip + prev controls

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.soft_boundary_limit = 1.35 * TRACK_WIDTH
        self.heading_stride = 2
        self.path_lookahead_steps = (2, 5, 9, 14)
        self.corner_lookahead_steps = (4, 8, 12, 18)
        self.corner_turn_norm = 0.5 * math.pi
        self.corner_control_threshold = 0.55
        self.switchback_turn_threshold = 0.18
        self.hairpin_turn_threshold = 0.82
        # Keep reward simple and tightly coupled to finishing the lap.
        self.centerline_penalty_weight = 0.35
        self.heading_penalty_weight = 0.20
        self.progress_reward_weight = 1.20
        self.reverse_progress_penalty_weight = 1.00
        self.tile_reward_weight = 2.50
        self.sharp_corner_threshold = 0.45
        self.corner_target_speed = 0.65
        self.corner_overspeed_penalty_weight = 0.35
        self.corner_slip_tolerance = 0.30
        self.corner_slip_penalty_weight = 0.22
        self.switchback_speed_reduction = 0.02
        self.switchback_overspeed_penalty_weight = 0.75
        self.switchback_slip_penalty_weight = 0.18
        self.hairpin_speed_reduction = 0.08
        self.hairpin_wide_penalty_weight = 0.45
        self.hairpin_wide_offset_threshold = 0.38
        self.hairpin_wide_heading_threshold = 0.05
        self.step_penalty = 0.02
        self.idle_speed_threshold = 0.08
        self.idle_penalty = 0.08
        self.soft_boundary_penalty = 0.90
        self.soft_boundary_penalty_scale = 0.60
        self.hard_offroad_penalty = 5.00
        self.lap_finish_bonus = 20.0
        self.base_max_episode_steps = 1700
        self.steps_per_track_tile = 6.0
        self.max_episode_steps = self.base_max_episode_steps
        self.steer_smoothing = 0.85
        self.gas_smoothing = 0.90
        self.brake_smoothing = 0.75
        self.steer_smoothing_floor = 0.60
        self.brake_smoothing_floor = 0.72
        self.corner_steer_boost = 0.16
        self.switchback_steer_boost = 0.12
        self.corner_brake_assist_weight = 0.0
        self.hairpin_steer_boost = 0.14
        self.hairpin_steer_smoothing_floor = 0.54
        self.hairpin_wide_steer_boost = 0.12
        self.hairpin_wide_steer_gain = 0.18
        self.hairpin_wide_throttle_cut = 0.22

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        assert self.car is not None
        self.car.destroy()

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

    def _reinit_colors(self, randomize):
        assert self.domain_randomize, (
            "domain_randomize must be True to use this function."
        )

        if randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose:
            print(f"Track generation: {i1}..{i2} -> {i2 - i1}-tiles track")
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                    )
                )
        self.track = track
        return True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = 0.0
        self.prev_reward = 0.0
        self.prev_track_idx = 0
        self.prev_steer = 0
        self.prev_gas = 0.0
        self.prev_brake = 0.0
        self.prev_tile_visited_count = 0
        self.episode_steps = 0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        # --------- BUILD TRUE CENTERLINE FROM ROAD ---------
        self.centerline = []
        

        self.centerline = [(t[2], t[3]) for t in self.track]
        self.centerline = self._resample_centerline(self.centerline, spacing=2)
        self.left_edge = []
        self.right_edge = []

        for t in self.track:
            _, beta, x, y = t

            dx = math.cos(beta)
            dy = math.sin(beta)

            left = (x - TRACK_WIDTH * dx, y - TRACK_WIDTH * dy)
            right = (x + TRACK_WIDTH * dx, y + TRACK_WIDTH * dy)

            self.left_edge.append(left)
            self.right_edge.append(right)

        # Optional but IMPORTANT for smooth curves
        self.left_edge = self._resample_centerline(self.left_edge, spacing=3)
        self.right_edge = self._resample_centerline(self.right_edge, spacing=3)

        self.car = Car(self.world, *self.track[0][1:4])
        self.car.hull.angularDamping = 4.0
        self.car.hull.linearDamping = 0.7
        self.max_episode_steps = max(
            self.base_max_episode_steps,
            int(len(self.track) * self.steps_per_track_tile),
        )

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}


    def step(self, action: np.ndarray | int):
        assert self.car is not None
        if action is not None:
            if self.continuous:
                action = action.astype(np.float64)
                car_x, car_y = self.car.hull.position
                closest_i = self._get_closest_centerline_index(car_x, car_y)
                corner_heading_deltas = self._get_heading_deltas(closest_i, self.corner_lookahead_steps)
                preview_heading_deltas = self._get_heading_deltas(closest_i, self.path_lookahead_steps)
                corner_severity = self._get_upcoming_corner_severity(closest_i, corner_heading_deltas)
                switchback_severity = self._get_switchback_severity(preview_heading_deltas)
                hairpin_severity = self._get_hairpin_severity(corner_severity, switchback_severity)
                turn_direction = self._get_turn_direction(preview_heading_deltas)

                next_i = (closest_i + self.heading_stride) % len(self.centerline)
                x1, y1 = self.centerline[closest_i]
                x2, y2 = self.centerline[next_i]
                track_angle = math.atan2(y2 - y1, x2 - x1)
                car_forward_angle = self.car.hull.angle + math.pi / 2.0
                heading_error = self._normalize_angle(track_angle - car_forward_angle)
                dx = car_x - x1
                dy = car_y - y1
                left_normal_x = -math.sin(track_angle)
                left_normal_y = math.cos(track_angle)
                signed_offset = dx * left_normal_x + dy * left_normal_y
                signed_offset_norm = float(np.clip(signed_offset / TRACK_WIDTH, -1.0, 1.0))
                wide_error = max(0.0, -turn_direction * signed_offset_norm)
                hairpin_wide_severity = 0.0
                if hairpin_severity > 0.0:
                    wide_factor = max(
                        0.0,
                        wide_error - self.hairpin_wide_offset_threshold,
                    ) / max(1e-6, 1.0 - self.hairpin_wide_offset_threshold)
                    heading_factor = max(
                        0.0,
                        abs(heading_error) / math.pi - self.hairpin_wide_heading_threshold,
                    ) / max(1e-6, 0.5 - self.hairpin_wide_heading_threshold)
                    hairpin_wide_severity = hairpin_severity * min(1.0, wide_factor) * min(1.0, heading_factor)

                # --- SMOOTH STEERING ---
                self.prev_steer = getattr(self, "prev_steer", 0)

                target = action[0]

                speed = math.hypot(
                        self.car.hull.linearVelocity[0],
                        self.car.hull.linearVelocity[1]
                )
                speed_norm = min(speed / 20.0, 1.0)

                steer_smoothing = (
                    self.steer_smoothing
                    - 0.18 * corner_severity
                    - 0.12 * switchback_severity
                    - 0.08 * hairpin_severity
                    - 0.06 * hairpin_wide_severity
                )
                steer_smoothing_floor = (
                    self.steer_smoothing_floor * (1.0 - max(hairpin_severity, hairpin_wide_severity))
                    + self.hairpin_steer_smoothing_floor * max(hairpin_severity, hairpin_wide_severity)
                )
                steer_smoothing = min(0.92, max(steer_smoothing_floor, steer_smoothing))
                smooth_steer = steer_smoothing * self.prev_steer + (1.0 - steer_smoothing) * target
                if hairpin_wide_severity > 0.0:
                    smooth_steer = float(
                        np.clip(
                            smooth_steer * (1.0 + self.hairpin_wide_steer_gain * hairpin_wide_severity),
                            -1.0,
                            1.0,
                        )
                    )

                steer_scale = max(0.32, 1 - speed * 0.02)
                steer_scale = min(
                    1.08,
                    steer_scale
                    + self.corner_steer_boost * corner_severity
                    + self.switchback_steer_boost * switchback_severity
                    + self.hairpin_steer_boost * hairpin_severity
                    + self.hairpin_wide_steer_boost * hairpin_wide_severity,
                )

                self.car.steer(-smooth_steer * steer_scale)
                self.prev_steer = smooth_steer

                
                # -------- THROTTLE SMOOTHING --------
                self.prev_gas = getattr(self, "prev_gas", 0)
                self.prev_brake = getattr(self, "prev_brake", 0)

                gas_smoothing = min(0.96, self.gas_smoothing + 0.04 * corner_severity + 0.03 * switchback_severity)
                brake_smoothing = self.brake_smoothing - 0.20 * corner_severity - 0.10 * switchback_severity
                brake_smoothing = min(0.92, max(self.brake_smoothing_floor, brake_smoothing))

                gas = gas_smoothing * self.prev_gas + (1.0 - gas_smoothing) * action[1]
                brake = brake_smoothing * self.prev_brake + (1.0 - brake_smoothing) * action[2]

                corner_target_speed = max(
                    0.18,
                    self.corner_target_speed
                    - self.switchback_speed_reduction * switchback_severity
                    - self.hairpin_speed_reduction * hairpin_severity,
                )
                if corner_severity > self.corner_control_threshold:
                    overspeed = max(0.0, speed_norm - corner_target_speed)
                    if overspeed > 0.0:
                        # Prefer lifting throttle over forcing brake in corners.
                        gas = min(
                            gas,
                            max(
                                0.28,
                                1.0
                                - 1.1 * overspeed
                                - 0.20 * hairpin_severity
                                - self.hairpin_wide_throttle_cut * hairpin_wide_severity,
                            ),
                        )

                self.car.gas(gas)
                self.car.brake(brake)

                self.prev_gas = gas
                self.prev_brake = brake
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.car.hull.angularVelocity *= 0.9
        self.car.hull.linearVelocity *= 0.995
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._get_state()

        terminated = False
        truncated = False
        info = {"lap_finished": False}
        if action is not None:  # First step without action, called from reset()
            self.episode_steps += 1
            self.car.fuel_spent = 0.0
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                info["lap_finished"] = False
                info["termination_reason"] = "out_of_bounds"
            elif self.episode_steps >= self.max_episode_steps:
                truncated = True
                info["lap_finished"] = False
                info["termination_reason"] = "time_limit"

        if self.render_mode == "human":
            self.render()
        
        # -------- RL REWARD --------
        car_x, car_y = self.car.hull.position

        closest_track_i = min(
            range(len(self.track)),
            key=lambda i: (self.track[i][2] - car_x)**2 +
                  (self.track[i][3] - car_y)**2
        )

        prev_track_idx = getattr(self, "prev_track_idx", closest_track_i)
        track_progress = closest_track_i - prev_track_idx

        if track_progress < -len(self.track)//2:
            track_progress += len(self.track)
        if track_progress > len(self.track)//2:
            track_progress -= len(self.track)

        self.prev_track_idx = closest_track_i

        closest_i = self._get_closest_centerline_index(car_x, car_y)

        # speed
        vx, vy = self.car.hull.linearVelocity
        speed = math.hypot(vx, vy)
        speed_norm = min(speed / 20.0, 1.0)

        # heading
        next_i = (closest_i + self.heading_stride) % len(self.centerline)
        x1, y1 = self.centerline[closest_i]
        x2, y2 = self.centerline[next_i]

        track_angle = math.atan2(y2 - y1, x2 - x1)
        car_forward_angle = self.car.hull.angle + math.pi / 2.0
        heading_error = self._normalize_angle(track_angle - car_forward_angle)

        # offset
        offset = math.hypot(car_x - x1, car_y - y1)
        offset_ratio = offset / TRACK_WIDTH
        corner_heading_deltas = self._get_heading_deltas(closest_i, self.corner_lookahead_steps)
        preview_heading_deltas = self._get_heading_deltas(closest_i, self.path_lookahead_steps)
        corner_severity = self._get_upcoming_corner_severity(closest_i, corner_heading_deltas)
        switchback_severity = self._get_switchback_severity(preview_heading_deltas)
        hairpin_severity = self._get_hairpin_severity(corner_severity, switchback_severity)
        turn_direction = self._get_turn_direction(preview_heading_deltas)
        dx = car_x - x1
        dy = car_y - y1
        left_normal_x = -math.sin(track_angle)
        left_normal_y = math.cos(track_angle)
        signed_offset = dx * left_normal_x + dy * left_normal_y
        signed_offset_norm = float(np.clip(signed_offset / TRACK_WIDTH, -1.0, 1.0))
        wide_error = max(0.0, -turn_direction * signed_offset_norm)
        hairpin_wide_severity = 0.0
        if hairpin_severity > 0.0:
            wide_factor = max(
                0.0,
                wide_error - self.hairpin_wide_offset_threshold,
            ) / max(1e-6, 1.0 - self.hairpin_wide_offset_threshold)
            heading_factor = max(
                0.0,
                abs(heading_error) / math.pi - self.hairpin_wide_heading_threshold,
            ) / max(1e-6, 0.5 - self.hairpin_wide_heading_threshold)
            hairpin_wide_severity = hairpin_severity * min(1.0, wide_factor) * min(1.0, heading_factor)

        car_angle = self.car.hull.angle
        forward_speed = -vx * math.sin(car_angle) + vy * math.cos(car_angle)
        lateral_speed = vx * math.cos(car_angle) + vy * math.sin(car_angle)
        slip_ratio = abs(lateral_speed) / (abs(forward_speed) + 1.0)
        heading_alignment = max(0.0, math.cos(heading_error))
        center_factor = max(0.0, 1.0 - min(offset_ratio, 1.0))

        # reward
        tile_gain = self.tile_visited_count - getattr(self, "prev_tile_visited_count", 0)
        self.prev_tile_visited_count = self.tile_visited_count

        drive_quality = 0.5 * heading_alignment + 0.5 * center_factor
        forward_progress = min(max(track_progress, 0.0), 1.0)
        reverse_progress = min(max(-track_progress, 0.0), 1.0)

        # Positive reward only comes from moving forward and covering new track.
        reward = -self.step_penalty
        reward += self.progress_reward_weight * forward_progress * (0.30 + 0.70 * drive_quality)
        reward += self.tile_reward_weight * min(tile_gain, 1.0)
        reward -= self.centerline_penalty_weight * min(offset_ratio, self.soft_boundary_limit / TRACK_WIDTH)
        reward -= self.heading_penalty_weight * (abs(heading_error) / math.pi)
        if corner_severity > self.sharp_corner_threshold:
            corner_strength = (corner_severity - self.sharp_corner_threshold) / max(
                1e-6, 1.0 - self.sharp_corner_threshold
            )
            corner_target_speed = max(
                0.18,
                self.corner_target_speed - self.switchback_speed_reduction * switchback_severity,
            )
            if speed_norm > corner_target_speed:
                reward -= (
                    self.corner_overspeed_penalty_weight
                    * (speed_norm - corner_target_speed)
                    * (0.5 + corner_strength + 0.5 * switchback_severity)
                )
            if slip_ratio > self.corner_slip_tolerance:
                reward -= (
                    self.corner_slip_penalty_weight
                    * (slip_ratio - self.corner_slip_tolerance)
                    * (0.5 + corner_strength + switchback_severity)
                )
        if switchback_severity > 0.0 and slip_ratio > self.corner_slip_tolerance * 0.9:
            reward -= (
                self.switchback_slip_penalty_weight
                * (slip_ratio - self.corner_slip_tolerance * 0.9)
                * switchback_severity
            )
        if hairpin_wide_severity > 0.0:
            reward -= (
                self.hairpin_wide_penalty_weight
                * hairpin_wide_severity
                * (0.6 + speed_norm)
            )
        if speed_norm < self.idle_speed_threshold and forward_progress == 0.0 and tile_gain <= 0.0:
            reward -= self.idle_penalty

        info["boundary_state"] = "on_track"
        if TRACK_WIDTH < offset <= self.soft_boundary_limit:
            boundary_excess = offset - TRACK_WIDTH
            boundary_span = max(1e-6, self.soft_boundary_limit - TRACK_WIDTH)
            boundary_ratio = boundary_excess / boundary_span
            reward -= self.soft_boundary_penalty + self.soft_boundary_penalty_scale * boundary_ratio
            info["boundary_state"] = "soft_offroad"
        elif offset > self.soft_boundary_limit:
            reward -= self.hard_offroad_penalty
            terminated = True
            info["lap_finished"] = False
            info["boundary_state"] = "hard_offroad"
            info["termination_reason"] = "off_track_severe"
        if reverse_progress > 0.0:
            reward -= self.reverse_progress_penalty_weight * reverse_progress
        if self.tile_visited_count == len(self.track) or self.new_lap:
            reward += self.lap_finish_bonus
            terminated = True
            info["lap_finished"] = True
            info["termination_reason"] = "lap_finished"

        info["tile_visited_count"] = self.tile_visited_count
        info["track_len"] = len(self.track)
        info["completion_pct"] = self.tile_visited_count / max(1, len(self.track))
        info["episode_steps"] = self.episode_steps
        info["track_progress"] = track_progress
        info["drive_quality"] = drive_quality
        info["corner_severity"] = corner_severity
        info["switchback_severity"] = switchback_severity
        info["hairpin_severity"] = hairpin_severity
        info["hairpin_wide_severity"] = hairpin_wide_severity
        info["slip_ratio"] = slip_ratio

        return self.state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)
        
    def _resample_centerline(self, points, spacing=10):
        import math

        new_points = [points[0]]
        accumulated = 0

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)

            while accumulated + dist >= spacing:
                remain = spacing - accumulated
                ratio = remain / dist

                nx = x1 + dx * ratio
                ny = y1 + dy * ratio

                new_points.append((nx, ny))

                x1, y1 = nx, ny
                dx = x2 - x1
                dy = y2 - y1
                dist = math.hypot(dx, dy)

                accumulated = 0

            accumulated += dist

        return new_points
    
    def _order_centerline(self, points):
        if not points:
            return points

        ordered = [points[0]]
        remaining = points[1:]

        while remaining:
            last = ordered[-1]

            next_idx = min(
                range(len(remaining)),
                key=lambda i: (remaining[i][0] - last[0])**2 +
                            (remaining[i][1] - last[1])**2
            )

            ordered.append(remaining.pop(next_idx))

        return ordered

    def _normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _get_closest_centerline_index(self, car_x, car_y):
        return min(
            range(len(self.centerline)),
            key=lambda i: (self.centerline[i][0] - car_x) ** 2
            + (self.centerline[i][1] - car_y) ** 2,
        )

    def _get_track_heading(self, center_idx, stride=None):
        if stride is None:
            stride = self.heading_stride
        next_idx = (center_idx + stride) % len(self.centerline)
        x1, y1 = self.centerline[center_idx]
        x2, y2 = self.centerline[next_idx]
        return math.atan2(y2 - y1, x2 - x1)

    def _get_heading_deltas(self, closest_i, lookahead_steps):
        base_heading = self._get_track_heading(closest_i)
        heading_deltas = []

        for step in lookahead_steps:
            future_idx = (closest_i + step) % len(self.centerline)
            future_heading = self._get_track_heading(future_idx)
            heading_deltas.append(self._normalize_angle(future_heading - base_heading))

        return heading_deltas

    def _get_switchback_severity(self, heading_deltas):
        switchback_delta = 0.0

        for current_delta, next_delta in zip(heading_deltas, heading_deltas[1:]):
            if (
                current_delta * next_delta < 0.0
                and abs(current_delta) > self.switchback_turn_threshold
                and abs(next_delta) > self.switchback_turn_threshold
            ):
                switchback_delta = max(switchback_delta, min(abs(current_delta), abs(next_delta)))

        return min(switchback_delta / self.corner_turn_norm, 1.0)

    def _get_hairpin_severity(self, corner_severity, switchback_severity):
        hairpin_base = max(0.0, corner_severity - self.hairpin_turn_threshold)
        hairpin_base /= max(1e-6, 1.0 - self.hairpin_turn_threshold)
        return min(max(0.0, hairpin_base * (1.0 - 0.7 * switchback_severity)), 1.0)

    def _get_turn_direction(self, heading_deltas):
        if not heading_deltas:
            return 0.0

        strongest_delta = max(heading_deltas, key=lambda delta: abs(delta))
        if abs(strongest_delta) < 1e-6:
            return 0.0
        return 1.0 if strongest_delta > 0.0 else -1.0

    def _get_upcoming_corner_severity(self, closest_i, heading_deltas=None):
        if heading_deltas is None:
            heading_deltas = self._get_heading_deltas(closest_i, self.corner_lookahead_steps)

        max_heading_delta = max((abs(delta) for delta in heading_deltas), default=0.0)
        return min(max_heading_delta / self.corner_turn_norm, 1.0)


    def _get_state(self):
        import math
        import numpy as np

        car_x, car_y = self.car.hull.position

        # -------- FIND CLOSEST CENTERLINE POINT --------
        closest_i = self._get_closest_centerline_index(car_x, car_y)

        # -------- HEADING --------
        next_i = (closest_i + self.heading_stride) % len(self.centerline)
        x1, y1 = self.centerline[closest_i]
        x2, y2 = self.centerline[next_i]

        track_angle = math.atan2(y2 - y1, x2 - x1)
        car_angle = self.car.hull.angle + math.pi / 2.0

        heading_error = self._normalize_angle(track_angle - car_angle)
        heading_norm = heading_error / math.pi   # [-1,1]

        # -------- OFFSET --------
        dx = car_x - x1
        dy = car_y - y1
        offset = math.hypot(dx, dy)
        left_normal_x = -math.sin(track_angle)
        left_normal_y = math.cos(track_angle)
        signed_offset = dx * left_normal_x + dy * left_normal_y
        signed_offset_norm = float(np.clip(signed_offset / TRACK_WIDTH, -1.0, 1.0))

        turn_deltas = self._get_heading_deltas(closest_i, self.path_lookahead_steps)
        corner_severity = self._get_upcoming_corner_severity(closest_i)

        # -------- SPEED --------
        vx = self.car.hull.linearVelocity[0]
        vy = self.car.hull.linearVelocity[1]
        speed = math.hypot(vx, vy)
        speed_norm = min(speed / 20.0, 1.0)
        slip_ratio = abs(
            vx * math.cos(self.car.hull.angle) + vy * math.sin(self.car.hull.angle)
        ) / (
            abs(-vx * math.sin(self.car.hull.angle) + vy * math.cos(self.car.hull.angle)) + 1.0
        )
        slip_norm = min(slip_ratio, 1.0)

        prev_steer = float(np.clip(getattr(self, "prev_steer", 0.0), -1.0, 1.0))
        prev_gas = float(np.clip(getattr(self, "prev_gas", 0.0), 0.0, 1.0))
        prev_brake = float(np.clip(getattr(self, "prev_brake", 0.0), 0.0, 1.0))

        # Signed curvature cues help with V-turns and back-to-back corners.
        path_cues = [
            float(np.clip(delta / self.corner_turn_norm, -1.0, 1.0))
            for delta in turn_deltas
        ]

        return np.array(
            path_cues
            + [
                speed_norm,
                heading_norm,
                signed_offset_norm,
                corner_severity,
                slip_norm,
                prev_steer,
                prev_gas,
                prev_brake,
            ],
            dtype=np.float32,
        )



    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        # -------- DYNAMIC ZOOM --------
        vx = self.car.hull.linearVelocity[0]
        vy = self.car.hull.linearVelocity[1]
        speed = math.hypot(vx, vy)

        base_zoom = 2.5
        zoom = (base_zoom - 0.008 * speed) * SCALE

        # clamp (important)
        target_zoom = max(1.8 * SCALE, min(2.5 * SCALE, zoom))

        # smooth
        self.prev_zoom = getattr(self,"prev_zoom",target_zoom)
        zoom = 0.9*self.prev_zoom + 0.1*target_zoom
        self.prev_zoom = zoom
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )
        self._render_sensors(zoom,trans,angle)

        self.surf = pygame.transform.flip(self.surf, False, True)

        self._render_clean_controls()

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

        # showing stats
       
    def _render_clean_controls(self):
            if self.car is None:
                return

            base_x = 40
            base_y = WINDOW_H - 60
            font = pygame.font.Font(None, 20)
            # label
            text = font.render("STEER", True, (255, 255, 255))
            self.surf.blit(text, (base_x, base_y - 20))

            text = font.render("THR", True, (255, 255, 255))
            self.surf.blit(text, (base_x + 110, base_y - 60))

            text = font.render("BRK", True, (255, 255, 255))
            self.surf.blit(text, (base_x + 140, base_y - 60))

            # --- Steering (horizontal) ---
            steer = self.car.wheels[0].joint.angle
            steer = max(min(-steer * 2, 1), -1)

            # background bar
            pygame.draw.rect(self.surf, (60, 60, 60), (base_x, base_y, 100, 8))

            center_x = base_x + 50

            if steer >= 0:
                pygame.draw.rect(
                    self.surf,
                    (0, 255, 0),
                    (center_x, base_y, steer * 50, 8),
                )
            else:
                pygame.draw.rect(
                    self.surf,
                (0, 255, 0),
                (center_x + steer * 50, base_y, -steer * 50, 8),
                )
               

            # --- Throttle (vertical) ---
            throttle = self.car.wheels[2].gas
            pygame.draw.rect(self.surf, (60, 60, 60), (base_x + 120, base_y - 40, 8, 40))
            pygame.draw.rect(
                self.surf,
                (0, 200, 255),
                (base_x + 120, base_y - throttle * 40, 8, throttle * 40),
            )

            # --- Brake (vertical) ---
            brake = self.car.wheels[2].brake
            pygame.draw.rect(self.surf, (60, 60, 60), (base_x + 140, base_y - 40, 8, 40))
            pygame.draw.rect(
                self.surf,
                (255, 0, 0),
                (base_x + 140, base_y - brake * 40, 8, brake * 40),
            )

            

            # -------- SPEED --------
            vx = self.car.hull.linearVelocity[0]
            vy = self.car.hull.linearVelocity[1]
            speed = math.hypot(vx, vy)

            # optional scaling for readability
            speed_display = speed * 3.6   # makes it look like km/h (tunable)

            # -------- LAP PROGRESS --------
            progress = (self.tile_visited_count / len(self.track)) * 100

            font = pygame.font.Font(None, 28)

            # SPEED
            speed_text = font.render(f"SPD: {speed_display:.1f}", True, (255,255,255))
            self.surf.blit(speed_text, (base_x, base_y - 100))

            # LAP %
            lap_text = font.render(f"LAP: {progress:.1f}%", True, (255,255,255))
            self.surf.blit(lap_text, (base_x, base_y - 130))

    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, translation, angle
            )

        # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)
        
        # LEFT BORDER
        for i in range(len(self.left_edge)):
            p1 = self._world_to_screen(*self.left_edge[i], zoom, translation, angle)
            p2 = self._world_to_screen(*self.left_edge[(i+1) % len(self.left_edge)], zoom, translation, angle)
            pygame.draw.line(self.surf, (255,255,255), p1, p2, 4)

        # RIGHT BORDER
        for i in range(len(self.right_edge)):
            p1 = self._world_to_screen(*self.right_edge[i], zoom, translation, angle)
            p2 = self._world_to_screen(*self.right_edge[(i+1) % len(self.right_edge)], zoom, translation, angle)
            pygame.draw.line(self.surf, (255,255,255), p1, p2, 4)

        # --------- DRAW CENTERLINE (CORRECT WAY) ---------
        dash_length = 6
        gap_length = 6

        draw = True
        accum = 0

        for i in range(len(self.centerline) - 1):

            x1, y1 = self.centerline[i]
            x2, y2 = self.centerline[i + 1]

            x_start, y_start = x1, y1

            while True:
                dx = x2 - x_start
                dy = y2 - y_start
                seg_len = math.hypot(dx, dy)

                if seg_len == 0:
                    break

                limit = dash_length if draw else gap_length

                if accum + seg_len < limit:
                    if draw:
                        p1 = self._world_to_screen(x_start, y_start, zoom, translation, angle)
                        p2 = self._world_to_screen(x2, y2, zoom, translation, angle)
                        pygame.draw.line(self.surf, (255,255,255), p1, p2, 2)

                    accum += seg_len
                    break

                else:
                    remain = limit - accum
                    ratio = remain / seg_len

                    xm = x_start + dx * ratio
                    ym = y_start + dy * ratio

                    if draw:
                        p1 = self._world_to_screen(x_start, y_start, zoom, translation, angle)
                        pm = self._world_to_screen(xm, ym, zoom, translation, angle)
                        pygame.draw.line(self.surf, (255,255,255), p1, pm, 2)

                    x_start, y_start = xm, ym
                    draw = not draw
                    accum = 0
                

    def _to_screen(self, point, zoom, translation, angle):
        x, y = point

        # translate
        x -= translation[0]
        y -= translation[1]

        # rotate
        px = x * math.cos(angle) - y * math.sin(angle)
        py = x * math.sin(angle) + y * math.cos(angle)

        # scale + center
        px = px * zoom + WINDOW_W / 2
        py = py * zoom + WINDOW_H / 2

        return int(px), int(py)
            

    def _render_sensors(self,zoom,translation,angle):
        import math

        if self.car is None:
            return

        car_x, car_y = self.car.hull.position
        car_angle = self.car.hull.angle

        
        # --------- DYNAMIC TRACK RAYS ---------

        car_x = self.car.hull.position[0]
        car_y = self.car.hull.position[1]

        # find closest track point
        # find closest centerline index (NOT track)
        closest_i = min(
            range(len(self.centerline)),
        key=lambda i: (self.centerline[i][0] - car_x)**2 +
                  (self.centerline[i][1] - car_y)**2
    )

        # get forward direction using next point
        next_i = (closest_i + 3) % len(self.centerline)

        x1, y1 = self.centerline[closest_i]
        x2, y2 = self.centerline[next_i]

        dir_x = x2 - x1
        dir_y = y2 - y1

        norm = math.hypot(dir_x, dir_y)
        dir_x /= norm
        dir_y /= norm

        # convert to angle
        track_angle = math.atan2(dir_y, dir_x)

        prev_i = (closest_i - 3) % len(self.centerline)

        xp, yp = self.centerline[prev_i]

        dir_prev_x = x1 - xp
        dir_prev_y = y1 - yp

        norm = math.hypot(dir_prev_x, dir_prev_y)
        dir_prev_x /= norm
        dir_prev_y /= norm

        angle_prev = math.atan2(dir_prev_y, dir_prev_x)

        # curvature = change in direction
        curvature = abs(track_angle - angle_prev)

        # normalize
        curvature = min(curvature, 1.0)

        # wide on straights, narrow on curves
        max_spread = math.radians(70)
        min_spread = math.radians(25)

        spread = max_spread * (1 - curvature) + min_spread * curvature

        num_rays = 7

        angles = [
            track_angle + spread * (i - num_rays//2) / (num_rays//2)
            for i in range(num_rays)
            ]
        
        max_len = 140

        # --------- TRAJECTORY RAYS (CORRECT) ---------

        car_x = self.car.hull.position[0]
        car_y = self.car.hull.position[1]

        # find closest centerline index
        closest_i = min(
            range(len(self.centerline)),
            key=lambda i: (self.centerline[i][0] - car_x)**2 +
                  (self.centerline[i][1] - car_y)**2
        )

        # ---- dynamic lookahead (THIS creates spread) ----
        speed = math.hypot(
            self.car.hull.linearVelocity[0],
            self.car.hull.linearVelocity[1]
        )

        base = 5
        scale = int(speed * 0.5)

        lookahead = [
            base,
            base + scale//2,
            base + scale,
            base + scale*2,
            # base + scale*3
        ]

        lookahead = [min(k,40) for k in lookahead]

        speed = math.hypot(
        self.car.hull.linearVelocity[0],
    self.car.hull.linearVelocity[1]
    )

        max_len = 10 + speed * 2   # tune this
        max_len = min(max_len, 60)  # cap

        for k in lookahead:
            idx = (closest_i + k) % len(self.centerline)

            tx, ty = self.centerline[idx]

            dx = tx - car_x
            dy = ty - car_y

            dist = math.hypot(dx, dy)

            # clamp length (important)
            if dist > max_len:
                scale = max_len / dist
                tx = car_x + dx * scale
                ty = car_y + dy * scale

            start = self._world_to_screen(car_x, car_y, zoom, translation, angle)
            end = self._world_to_screen(tx, ty, zoom, translation, angle)

            pygame.draw.line(self.surf, (0, 200, 255), start, end, 2)

            # optional: endpoint marker
            pygame.draw.circle(self.surf, (0, 200, 255), (int(end[0]), int(end[1])), 3)

             
    def _world_to_screen(self, x, y, zoom, translation, angle):
        v = pygame.math.Vector2(x, y)

        # SAME as road rendering
        v = v.rotate_rad(angle)
        px = v[0] * zoom + translation[0]
        py = v[1] * zoom + translation[1]

        return px, py
    
    

    def _render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert self.car is not None
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.car.wheels[0].omega,
            vertical_ind(7, 0.01 * self.car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[1].omega,
            vertical_ind(8, 0.01 * self.car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[2].omega,
            vertical_ind(9, 0.01 * self.car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.car.wheels[3].omega,
            vertical_ind(10, 0.01 * self.car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity,
            horiz_ind(30, -0.8 * self.car.hull.angularVelocity),
            (255, 0, 0),
        )

    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = CarRacing(render_mode="human")

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()
