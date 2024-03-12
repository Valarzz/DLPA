import math
import numpy as np


PLAYER_CONFIG = {
    'POWER_RATE': 0.006,
    'SIZE': 0.9,
    'RAND': 0.1,
    'ACCEL_MAX': 1.0,
    'SPEED_MAX': 1.0,
    'DECAY': 0.4,
    'MASS': 60
}

BALL_CONFIG = {
    'POWER_RATE': 0.027,
    'SIZE': 0.4,
    'RAND': 0.05,
    'ACCEL_MAX': 2.7,
    'SPEED_MAX': 2.7,
    'DECAY': 0.94,
    'MASS': 0.2
}

MINPOWER = -100
MAXPOWER = 100
KICKABLE = PLAYER_CONFIG['SIZE'] + 0.7
CATCHABLE = 2.0
CATCH_PROBABILITY = 1.0
INERTIA_MOMENT = 5.0
PITCH_LENGTH = 40  # 105
PITCH_WIDTH = 30  # 68
CENTRE_CIRCLE_RADIUS = 9.15
PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_LENGTH = 5.5
GOAL_AREA_WIDTH = 18.32
GOAL_WIDTH = 14.02
GOAL_DEPTH = 2.44

SCALE_VECTOR = np.array([PITCH_LENGTH / 2, PITCH_WIDTH, 2.0, 2.0, 2 * np.pi,
                         PITCH_LENGTH / 2, PITCH_WIDTH, 2.0, 2.0, 2 * np.pi,
                         PITCH_LENGTH / 2, PITCH_WIDTH, 6.0, 6.0])
SHIFT_VECTOR = np.array([0.0, PITCH_WIDTH / 2, 1.0, 1.0, np.pi,
                         0.0, PITCH_WIDTH / 2, 1.0, 1.0, np.pi,
                         0.0, PITCH_WIDTH / 2, 3, 3])
LOW_VECTOR = -SHIFT_VECTOR
HIGH_VECTOR = np.array(SCALE_VECTOR-SHIFT_VECTOR)


def bound(value, lower, upper):
    """ Clips off a value which exceeds the lower or upper bounds. """
    if value < lower:
        return lower
    elif value > upper:
        return upper
    else:
        return value


def bound_vector(vect, maximum):
    """ Bounds a vector between a negative and positive maximum range. """
    xval = bound(vect[0], -maximum, maximum)
    yval = bound(vect[1], -maximum, maximum)
    return np.array([xval, yval])


def angle_difference(angle1, angle2):
    """ Computes the real difference between angles. """
    return norm_angle(angle1 - angle2)


def norm_angle(angle):
    """ Normalize the angle between -pi and pi. """
    while angle > np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle


def angle_close(angle1, angle2):
    """ Determines whether an angle1 is close to angle2. """
    return abs(angle_difference(angle1, angle2)) < np.pi/8


def angle_between(pos1, pos2):
    """ Computes the angle between two positions. """
    diff = pos2 - pos1
    # return np.arctan2(diff[1], diff[0])
    return math.atan2(diff[1], diff[0])  # faster than numpy


def angle_position(theta):
    """ Computes the position on a unit circle at angle theta. """
    return vector(np.cos(theta), np.sin(theta))


def vector(xvalue, yvalue):
    """ Returns a 2D numpy vector. """
    # return np.array([float(xvalue), float(yvalue)])
    return np.array([xvalue, yvalue], dtype=np.float64)


def vector_to_tuple(vect):
    """ Converts a numpy array to a tuple. """
    assert len(vect) == 2
    return (vect[0], vect[1])
    # return tuple(map(tuple, vect))
