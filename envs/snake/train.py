"""Training loop: play games, let the agent learn after every move and
after every full game, track progress, save the best model.

Run: python train.py
Stop any time with Ctrl+C — model.pth is saved whenever a new high score
is reached, AND every CHECKPOINT_INTERVAL games regardless, so a kill/
resume never loses more than a few dozen games of progress even during a
long plateau. Resumes from model.pth automatically if it already exists
(weights AND exploration progress, so a resumed run doesn't re-explore
randomly from scratch). Training also stops on its own the moment the
snake fills the entire board (a win).
"""
import os

import torch

from agent import Agent
from env import SnakeEnv

MODEL_PATH = "model.pth"
CHECKPOINT_INTERVAL = 50  # save progress at least this often, even with no new record
TARGET_UPDATE_INTERVAL = 25  # sync the target network this often (in games)


def _save_checkpoint(agent, record):
    torch.save(
        {"model_state": agent.model.state_dict(), "n_games": agent.n_games, "record": record},
        MODEL_PATH,
    )


def train():
    agent = Agent()
    env = SnakeEnv()
    record = 0
    total_score = 0

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH)
        agent.model.load_state_dict(checkpoint["model_state"])
        agent.n_games = checkpoint["n_games"]
        record = checkpoint["record"]
        agent.update_target()  # sync target net to the loaded weights, not the initial random ones
        print(f"Resumed from {MODEL_PATH}: n_games={agent.n_games}, record={record}")

    session_games = 0  # this run only, so a resumed run's mean isn't
                        # diluted by scores from before the restart

    state = env.reset()
    while True:
        old_state = state

        action = agent.get_action(old_state)
        state, reward, done, score = env.step(action)

        # learn immediately from this one move
        agent.train_short_memory(old_state, action, reward, state, done)
        agent.remember(old_state, action, reward, state, done)

        if done:
            # game over: replay a batch from memory for a deeper update
            agent.n_games += 1
            session_games += 1
            agent.train_long_memory()

            if agent.n_games % TARGET_UPDATE_INTERVAL == 0:
                agent.update_target()

            if score > record:
                record = score
                _save_checkpoint(agent, record)
            elif agent.n_games % CHECKPOINT_INTERVAL == 0:
                _save_checkpoint(agent, record)

            total_score += score
            mean_score = total_score / session_games
            print(
                f"Game {agent.n_games:5d}  Score {score:3d}  "
                f"Record {record:3d}  Mean {mean_score:.2f}"
            )

            if env.game.won:
                _save_checkpoint(agent, record)
                print(f"WON! Filled the whole board on game {agent.n_games}. Stopping training.")
                return

            state = env.reset()


if __name__ == "__main__":
    train()