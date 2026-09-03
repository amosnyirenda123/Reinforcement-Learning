"""Pygame renderer for Snake. Two modes:

  python play.py          -> you play with arrow keys
  python play.py --watch  -> loads model.pth and watches the trained AI play
"""
import sys

import pygame
import torch

from env import SnakeEnv, STRAIGHT, RIGHT_TURN, LEFT_TURN, _CLOCKWISE
from game import Direction, Point

CELL = 25
SPEED_HUMAN = 10
SPEED_AI = 15

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 180, 0)
DARK_GREEN = (0, 100, 0)

_KEY_TO_DIRECTION = {
    pygame.K_UP: Direction.UP,
    pygame.K_DOWN: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT,
}

_OPPOSITE = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


PANEL_HEIGHT = 100
YELLOW = (230, 200, 0)
GRAY = (140, 140, 140)


def draw(screen, env, reasoning=None):
    game = env.game
    screen.fill(BLACK)
    for i, point in enumerate(game.snake):
        color = GREEN if i == 0 else DARK_GREEN
        rect = pygame.Rect(point.x * CELL, point.y * CELL, CELL, CELL)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, BLACK, rect, 1)

    food_rect = pygame.Rect(game.food.x * CELL, game.food.y * CELL, CELL, CELL)
    pygame.draw.rect(screen, RED, food_rect)

    font = pygame.font.SysFont("arial", 20)
    text = font.render(f"Score: {game.score}", True, WHITE)
    screen.blit(text, (5, 5))

    if reasoning is not None:
        # reasoning: list of (direction_name, description, is_chosen),
        # best move first, one per candidate move
        panel_y = game.grid_size * CELL
        panel_rect = pygame.Rect(0, panel_y, game.grid_size * CELL, PANEL_HEIGHT)
        pygame.draw.rect(screen, (20, 20, 20), panel_rect)

        small_font = pygame.font.SysFont("consolas", 18)
        header = small_font.render("What the AI is thinking:", True, GRAY)
        screen.blit(header, (8, panel_y + 6))

        for i, (direction_name, description, chosen) in enumerate(reasoning):
            color = YELLOW if chosen else WHITE
            prefix = "-> " if chosen else "   "
            line = small_font.render(f"{prefix}{direction_name}: {description}", True, color)
            screen.blit(line, (8, panel_y + 28 + i * 22))

    pygame.display.flip()


def play_human():
    env = SnakeEnv()
    pygame.init()
    screen = pygame.display.set_mode((env.game.grid_size * CELL, env.game.grid_size * CELL))
    pygame.display.set_caption("Snake")
    clock = pygame.time.Clock()

    env.reset()
    current_direction = env.game.direction

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key in _KEY_TO_DIRECTION:
                new_dir = _KEY_TO_DIRECTION[event.key]
                if new_dir != _OPPOSITE[current_direction]:
                    current_direction = new_dir

        # translate the absolute human direction into env's relative action
        idx_now = _CLOCKWISE.index(env.game.direction)
        idx_new = _CLOCKWISE.index(current_direction)
        diff = (idx_new - idx_now) % 4
        action = {0: STRAIGHT, 1: RIGHT_TURN, 3: LEFT_TURN}.get(diff, STRAIGHT)

        _, _, done, score = env.step(action)
        draw(screen, env)
        clock.tick(SPEED_HUMAN)

        if done:
            if env.game.won:
                print(f"YOU WON! Filled the whole board. Score: {score}")
            else:
                print(f"Game over! Score: {score}")
            env.reset()
            current_direction = env.game.direction


def _describe_move(game, resulting_dir):
    """Plain-English reason for a candidate move, grounded in the actual
    board: would it crash immediately, and does it get closer to food?"""
    head = game.snake[0]
    dx, dy = resulting_dir.value
    new_head = Point(head.x + dx, head.y + dy)
    if game.is_collision(new_head):
        return "would crash immediately"
    cur_dist = abs(head.x - game.food.x) + abs(head.y - game.food.y)
    new_dist = abs(new_head.x - game.food.x) + abs(new_head.y - game.food.y)
    return "moves toward food" if new_dist < cur_dist else "moves away from food"


def watch_ai():
    from agent import Agent

    agent = Agent()
    checkpoint = torch.load("model.pth")
    agent.model.load_state_dict(checkpoint["model_state"])
    agent.model.eval()

    env = SnakeEnv()
    pygame.init()
    screen = pygame.display.set_mode((env.game.grid_size * CELL, env.game.grid_size * CELL + PANEL_HEIGHT))
    pygame.display.set_caption("Snake AI")
    clock = pygame.time.Clock()

    state = env.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        state_t = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = agent.model(state_t)
        action = torch.argmax(q_values).item()

        # build the reasoning readout before stepping (step mutates
        # env.game.direction, which _action_to_direction depends on).
        # Ranked best-to-worst by the network's own Q-values, described
        # in plain English rather than raw numbers.
        ranked_actions = sorted((STRAIGHT, RIGHT_TURN, LEFT_TURN), key=lambda a: -q_values[a].item())
        reasoning = []
        for a in ranked_actions:
            resulting_dir = env._action_to_direction(a)
            description = _describe_move(env.game, resulting_dir)
            reasoning.append((resulting_dir.name, description, a == action))

        state, _, done, score = env.step(action)
        draw(screen, env, reasoning=reasoning)
        clock.tick(SPEED_AI)

        if done:
            if env.game.won:
                print(f"AI WON! Filled the whole board. Score: {score}")
            else:
                print(f"AI game over! Score: {score}")
            state = env.reset()


if __name__ == "__main__":
    if "--watch" in sys.argv:
        watch_ai()
    else:
        play_human()