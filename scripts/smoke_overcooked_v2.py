"""Run one seeded random-action episode; this checks setup, not policy quality."""

import argparse
import json

import jax

from jaxmarl import make


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--steps", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view-size", type=positive_int, default=None,
                        help="View radius; omit for full-grid observations")
    args = parser.parse_args()

    env = make("overcooked_v2", layout=args.layout, max_steps=args.steps,
               agent_view_size=args.view_size)
    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    shapes = {agent: list(obs[agent].shape) for agent in env.agents}
    returns = {agent: 0.0 for agent in env.agents}
    step_fn = jax.jit(env.step)
    terminated = False

    for step in range(args.steps):
        keys = jax.random.split(key, env.num_agents + 2)
        key, step_key = keys[:2]
        actions = {
            agent: env.action_space(agent).sample(keys[i + 2])
            for i, agent in enumerate(env.agents)
        }
        obs, state, rewards, dones, _ = step_fn(step_key, state, actions)
        for agent in env.agents:
            returns[agent] += float(rewards[agent])
        terminated = bool(dones["__all__"])
        if terminated:
            break

    if not terminated:
        raise RuntimeError("Episode did not terminate at the configured horizon")
    print(json.dumps({
        "status": "passed", "devices": [str(d) for d in jax.devices()],
        "layout": args.layout, "seed": args.seed, "view_size": args.view_size,
        "steps": step + 1, "observation_shapes": shapes, "returns": returns,
    }, indent=2))


if __name__ == "__main__":
    main()
