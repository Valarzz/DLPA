import numpy as np
import torch
import random
__numba = True
try:
    from numba import jit
except:
    __numba = False


class Constants:
    # platform constants
    PLATFORM_HEIGHT = 40.0
    WIDTH1 = 250
    WIDTH2 = 275
    WIDTH3 = 50
    GAP1 = 225
    GAP2 = 235
    HEIGHT1 = 0.0
    HEIGHT2 = 0.0
    HEIGHT3 = 0.0
    MAX_HEIGHT = max(1.0, HEIGHT1, HEIGHT2, HEIGHT3)
    MAX_PLATFORM_WIDTH = max(WIDTH1, WIDTH2, WIDTH3)
    TOTAL_WIDTH = WIDTH1 + WIDTH2 + WIDTH3 + GAP1 + GAP2
    MAX_GAP = max(GAP1, GAP2)
    CHECK_SCALE = True

    # enemy constants
    ENEMY_SPEED = 30.0
    ENEMY_NOISE = 0.5
    ENEMY_SIZE = np.array((20.0, 30.0))

    # player constants
    PLAYER_NOISE = ENEMY_NOISE
    PLAYER_SIZE = np.copy(ENEMY_SIZE)

    # action constants
    DT = 0.05
    MAX_DX = 100.0
    MAX_DY = 200.0
    MAX_DX_ON = 70.0
    MAX_DDX = (MAX_DX - MAX_DX_ON) / DT
    MAX_DDY = MAX_DY / DT
    LEAP_DEV = 1.0
    HOP_DEV = 1.0
    VELOCITY_DECAY = 0.99
    GRAVITY = 9.8

    # scaling constants
    SHIFT_VECTOR = np.array([PLAYER_SIZE[0], 0., 0., ENEMY_SPEED,  # basic features
                             0., 0., 0., 0., 0.])  # platform features
    SCALE_VECTOR = np.array([TOTAL_WIDTH + PLAYER_SIZE[0], MAX_DX, TOTAL_WIDTH, 2 * ENEMY_SPEED,  # basic
                             MAX_PLATFORM_WIDTH, MAX_PLATFORM_WIDTH, MAX_GAP, TOTAL_WIDTH, MAX_HEIGHT])  # platform

    # available actions: RUN, HOP, LEAP
    # parameters for actions other than the one selected are ignored
    # action bounds were set from empirical testing using the default constants
    PARAMETERS_MIN = np.array([0, 0, 0])
    PARAMETERS_MAX = np.array([
        30,  # run
        720,  # hop
        430  # leap
    ])


RUN = "run"
HOP = "hop"
LEAP = "leap"
JUMP = "jump"

ACTION_LOOKUP = {
    0: RUN,
    1: HOP,
    2: LEAP,
}


class Platform:
    """ Represents a fixed platform. """

    def __init__(self, xpos, ypos, width):
        self.position = np.array((xpos, ypos))
        self.size = np.array((width, Constants.PLATFORM_HEIGHT))


class Enemy:
    """ Defines the enemy. """

    size = Constants.ENEMY_SIZE
    speed = Constants.ENEMY_SPEED

    def __init__(self, platform, p, v):
        """ Initializes the enemy on the platform. """
        self.dx = -self.speed
        self.platform = platform
        self.position = np.array((p, Constants.PLATFORM_HEIGHT))
        # self.velocity = np.array((v, 0.0))
        self.np_random = np.random  # overwritten by seed()

    def update(self, dt):
        """ Shift the enemy along the platform. """
        right = self.platform.position[0] + self.platform.size[0] - self.size[0]
        if not self.platform.position[0] < self.position[0] < right:
            self.dx *= -1
        self.dx += self.np_random.normal(0.0, Constants.ENEMY_NOISE * dt)
        self.dx = np.clip(self.dx, -self.speed, self.speed)
        self.position[0] += self.dx * dt
        self.position[0] = np.clip(self.position[0], self.platform.position[0], right)


class Player:
    """ Represents the player character. """

    size = Constants.ENEMY_SIZE
    speed = Constants.ENEMY_SPEED

    def __init__(self, p, v):
        """ Initialize the position to the starting platform. """
        # self.position = vector(self.np_random.rand()*0.01, PLATHEIGHT)
        # self.velocity = vector(self.np_random.rand()*0.0001, 0.0)
        # self.position = np.array((0., Constants.PLATFORM_HEIGHT))
        # self.velocity = np.array((0., 0.0))
        self.position = np.array((p, Constants.PLATFORM_HEIGHT))
        self.velocity = np.array((v, 0.0))
        self.np_random = np.random  # overwritten by seed()

    def update(self, dt):
        """ Update the position and velocity. """
        self.position = self.position + self.velocity * dt
        self.position[0] = np.clip(self.position[0], 0.0, Constants.TOTAL_WIDTH)
        self.velocity[0] *= Constants.VELOCITY_DECAY

    def accelerate(self, accel, dt=Constants.DT):
        """ Applies a power to the entity in direction theta. """
        accel = np.clip(accel, (-Constants.MAX_DDX, -Constants.MAX_DDY), (Constants.MAX_DDX, Constants.MAX_DDY))
        self.velocity = self.velocity + accel * dt
        self.velocity[0] -= abs(self.np_random.normal(0.0, Constants.PLAYER_NOISE * dt))
        self.velocity = np.clip(self.velocity, (-Constants.MAX_DX, -Constants.MAX_DY),
                                (Constants.MAX_DX, Constants.MAX_DY))
        self.velocity[0] = max(self.velocity[0], 0.0)

    def ground_bound(self):
        """ Bound dx while on the ground. """
        self.velocity[0] = np.clip(self.velocity[0], 0.0, Constants.MAX_DX_ON)

    def run(self, power, dt):
        """ Run for a given power and time. """
        if dt > 0:
            self.accelerate(np.array((power / dt, 0.0), dtype=object,), dt)

    def jump(self, power):
        """ Jump up for a single step. """
        self.accelerate(np.array((0.0, power / Constants.DT)))

    def jump_to(self, diffx, dy0, dev):
        """ Jump to a specific position. """
        time = 2.0 * dy0 / Constants.GRAVITY + 1.0
        dx0 = diffx / time - self.velocity[0]
        dx0 = np.clip(dx0, -Constants.MAX_DDX, Constants.MAX_DY - dy0)
        if dev > 0:
            noise = -abs(self.np_random.normal(0.0, dev, 2))
        else:
            noise = np.zeros((2,))
        acceleration = np.array((dx0, dy0), dtype=object,) + noise
        self.accelerate(acceleration / Constants.DT)

    def hop_to(self, diffx):
        """ Jump high to a position. """
        self.jump_to(diffx, 35.0, Constants.HOP_DEV)

    def leap_to(self, diffx):
        """ Jump over a gap. """
        self.jump_to(diffx, 25.0, Constants.LEAP_DEV)

    def fall(self):
        """ Apply gravity. """
        self.accelerate(np.array((0.0, -Constants.GRAVITY)))

    def decollide(self, other):
        """ Shift overlapping entities apart. """
        precorner = other.position - self.size
        postcorner = other.position + other.size
        newx, newy = self.position[0], self.position[1]
        if self.position[0] < other.position[0]:
            newx = precorner[0]
        elif self.position[0] > postcorner[0] - self.size[0]:
            newx = postcorner[0]
        if self.position[1] < other.position[1]:
            newy = precorner[1]
        elif self.position[1] > postcorner[1] - self.size[1]:
            newy = postcorner[1]
        if newx == self.position[0]:
            self.velocity[1] = 0.0
            self.position[1] = newy
        elif newy == self.position[1]:
            self.velocity[0] = 0.0
            self.position[0] = newx
        elif abs(self.position[0] - newx) < abs(self.position[1] - newy):
            self.velocity[0] = 0.0
            self.position[0] = newx
        else:
            self.velocity[1] = 0.0
            self.position[1] = newy

    def above_platform(self, platform):
        """ Checks the player is above the platform. """
        return -self.size[0] <= self.position[0] - platform.position[0] <= platform.size[0]

    def on_platform(self, platform):
        """ Checks the player is standing on the platform. """
        ony = self.position[1] - platform.position[1] == platform.size[1]
        return self.above_platform(platform) and ony

    def colliding(self, other):
        """ Check if two entities are overlapping. """
        return _colliding(self.size, self.position, other.size, other.position)


def _colliding(self_size, self_position, other_size, other_position):
    precorner = other_position - self_size
    postcorner = other_position + other_size
    collide = np.all(precorner < self_position)
    collide = collide and np.all(self_position < postcorner)
    return collide

