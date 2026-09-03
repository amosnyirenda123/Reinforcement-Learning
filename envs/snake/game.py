"""Pure Snake game logic. No rendering, no pygame — so this can run headless
at thousands of steps/sec during RL training, and also be driven by the
pygame UI for human play.
"""
import random
from collections import namedtuple
from enum import Enum

Point = namedtuple("Point", ["x", "y"])


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class SnakeGame:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        center = self.grid_size // 2
        self.direction = Direction.RIGHT
        self.snake = [
            Point(center, center),
            Point(center - 1, center),
            Point(center - 2, center),
        ]
        self.score = 0
        self.frame_iteration = 0
        self.won = False
        self._place_food()
        return self._get_state()

    def _place_food(self):
        while True:
            p = Point(
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if p not in self.snake:
                self.food = p
                return

    def is_collision(self, point=None):
        p = point or self.snake[0]
        if p.x < 0 or p.x >= self.grid_size or p.y < 0 or p.y >= self.grid_size:
            return True
        if p in self.snake[1:]:
            return True
        return False

    def step(self, action):
        """action: Direction to move this frame (caller decides — human input
        or agent policy). Returns (reward, game_over, score)."""
        self.frame_iteration += 1
        self.direction = action

        head = self.snake[0]
        dx, dy = self.direction.value
        new_head = Point(head.x + dx, head.y + dy)
        self.snake.insert(0, new_head)

        reward = 0
        game_over = False

        # dying, or taking too long without eating -> end episode
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if new_head == self.food:
            self.score += 1
            reward = 10
            if len(self.snake) == self.grid_size * self.grid_size:
                # snake fills the whole board -> won, no cell left for food
                self.won = True
                game_over = True
                return reward, game_over, self.score
            self._place_food()
        else:
            self.snake.pop()

        return reward, game_over, self.score

    def _get_state(self):
        return {
            "snake": list(self.snake),
            "food": self.food,
            "direction": self.direction,
        }