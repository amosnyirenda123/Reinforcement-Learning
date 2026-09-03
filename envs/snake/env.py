"""RL environment wrapper around game.py.

Converts the raw game into the standard shape an RL agent expects:
  - reset() -> state vector
  - step(action) -> (next_state, reward, done, score)

The "state" is the set of features the neural net actually sees. We don't
give it the whole grid — we give it a compact, hand-picked feature vector:
  - 8 distance sensors (one per compass direction) reporting how much open
    space is between the head and the nearest wall/own-body segment in
    that direction. This is what gives the agent "body awareness" — it
    can sense its own tail wrapping around it, not just the one cell
    directly in front of it.
  - current direction (one-hot)
  - food direction relative to head
  - normalized snake length (== score, scaled to 0..1) so the agent knows
    how big/dangerous its own body currently is and can play more
    cautiously as it grows.
"""
import numpy as np
from game import SnakeGame, Direction, Point

# Action space: relative to the snake's current heading, so the agent can
# never "pick" reverse-into-itself as a move.
STRAIGHT, RIGHT_TURN, LEFT_TURN = 0, 1, 2

_CLOCKWISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

# 8 compass directions around the head, used for the distance sensors
_COMPASS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


class SnakeEnv:
    def __init__(self, grid_size=20):
        self.game = SnakeGame(grid_size=grid_size)

    def reset(self):
        self.game.reset()
        return self._get_state()

    def step(self, action_idx):
        new_direction = self._action_to_direction(action_idx)
        reward, done, score = self.game.step(new_direction)
        return self._get_state(), reward, done, score

    def _action_to_direction(self, action_idx):
        idx = _CLOCKWISE.index(self.game.direction)
        if action_idx == STRAIGHT:
            return _CLOCKWISE[idx]
        elif action_idx == RIGHT_TURN:
            return _CLOCKWISE[(idx + 1) % 4]
        else:  # LEFT_TURN
            return _CLOCKWISE[(idx - 1) % 4]

    def _distance_sensors(self):
        """For each of the 8 compass directions, how many open cells lie
        between the head and the nearest wall or own-body segment,
        normalized to 0..1 (1 = fully open in that direction)."""
        game = self.game
        head = game.snake[0]
        body = set(game.snake[1:])  # exclude the head itself
        sensors = []
        for dx, dy in _COMPASS:
            dist = 0
            x, y = head.x, head.y
            while True:
                x += dx
                y += dy
                dist += 1
                if x < 0 or x >= game.grid_size or y < 0 or y >= game.grid_size:
                    break
                if Point(x, y) in body:
                    break
            sensors.append(dist / game.grid_size)
        return sensors

    def _get_state(self):
        game = self.game
        head = game.snake[0]

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = self._distance_sensors() + [
            # current direction (one-hot)
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # food direction relative to head
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y,
            # normalized snake length == normalized score, so the agent
            # knows how big (and how dangerous) its own body currently is
            len(game.snake) / (game.grid_size * game.grid_size),
        ]
        return np.array(state, dtype=np.float32)


STATE_SIZE = 17
ACTION_SIZE = 3