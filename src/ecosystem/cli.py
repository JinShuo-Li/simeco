"""Command line interface for interactive and batch simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reporting import print_summary, summary, write_history
from .simulation import Simulation
from .snapshot import load_snapshot, save_snapshot


def _simulation(args: argparse.Namespace) -> Simulation:
    requested_mode = getattr(args, "controller_mode", None)
    if getattr(args, "load", None):
        simulation = load_snapshot(args.load)
        if requested_mode is not None:
            simulation.learning = requested_mode != "instinct_only"
            simulation.memory = requested_mode == "instinct+learning+memory"
        return simulation
    mode = requested_mode or "instinct+learning+memory"
    return Simulation(
        seed=args.seed,
        learning=mode != "instinct_only",
        memory=mode == "instinct+learning+memory",
    )


def run_batch(args: argparse.Namespace) -> int:
    simulation = _simulation(args)
    simulation.run(args.steps)
    if args.metrics:
        write_history(simulation, args.metrics)
    if args.snapshot:
        save_snapshot(simulation, args.snapshot)
    print_summary(simulation)
    return 0


def run_compare(args: argparse.Namespace) -> int:
    rows = []
    for seed in range(args.seed, args.seed + args.replicates):
        for learning, memory in ((False, False), (True, False), (True, True)):
            simulation = Simulation(seed=seed, learning=learning, memory=memory)
            simulation.run(args.steps)
            rows.append(summary(simulation))
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def run_inspect(args: argparse.Namespace) -> int:
    simulation = load_snapshot(args.snapshot)
    if args.organism is None:
        print_summary(simulation)
        return 0
    animal = simulation.organisms.get(args.organism)
    if animal is None:
        raise SystemExit(f"organism {args.organism} is not alive in this snapshot")
    data = animal.to_dict()
    policy = data.pop("adaptive_policy")
    data["adaptive_policy_summary"] = {
        "shape": [policy["inputs"], policy["hidden"], policy["memory_size"], policy["outputs"]],
        "updates": policy["updates"],
        "reward_total": policy["reward_total"],
        "baseline": policy["baseline"],
        "memory": policy["memory"],
        "previous_actions": policy["previous_actions"],
        "previous_outcomes": policy["previous_outcomes"],
        "parameter_count": (
            policy["hidden"] * policy["inputs"]
            + policy["hidden"]
            + policy["outputs"] * policy["hidden"]
            + policy["outputs"]
            + sum(len(row) for name in ("wz", "uz", "wr", "ur", "wh", "uh") for row in policy[name])
            + 3 * policy["memory_size"]
            + policy["outputs"] * policy["memory_size"]
        ),
        "tbptt_updates": policy["tbptt_updates"],
        "buffered_transitions": len(policy["trajectory"]),
    }
    if args.weights:
        data["adaptive_policy"] = policy
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


def run_interactive(args: argparse.Namespace) -> int:
    from .tui import run_tui

    simulation = _simulation(args)
    simulation = run_tui(simulation, args.snapshot, max_steps=args.max_steps)
    if args.save_on_exit:
        save_snapshot(simulation, args.snapshot)
    print_summary(simulation)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="ecosystem", description="A learning predator-prey ecosystem")
    subparsers = root.add_subparsers(dest="command")

    batch = subparsers.add_parser("run", help="run quickly without a TUI")
    batch.add_argument("--steps", type=int, default=1000)
    batch.add_argument("--seed", type=int, default=3)
    batch.add_argument("--load", metavar="SNAPSHOT")
    batch.add_argument("--snapshot", metavar="PATH")
    batch.add_argument("--metrics", metavar="CSV")
    learning = batch.add_mutually_exclusive_group()
    learning.add_argument(
        "--learning", action="store_const", const="instinct+learning+memory",
        dest="controller_mode", default=None,
        help="use instinct plus recurrent lifetime learning (default)",
    )
    learning.add_argument(
        "--feedforward-learning", action="store_const", const="instinct+learning",
        dest="controller_mode", help="use the V3-style adaptive policy without memory",
    )
    learning.add_argument(
        "--instinct-only", "--no-learning", action="store_const", const="instinct_only",
        dest="controller_mode", help="use innate behavior without adaptive influence or updates",
    )
    batch.set_defaults(func=run_batch)

    compare = subparsers.add_parser(
        "compare", help="compare instinct-only, feed-forward, and recurrent modes"
    )
    compare.add_argument("--steps", type=int, default=1500)
    compare.add_argument("--seed", type=int, default=3)
    compare.add_argument("--replicates", type=int, default=3)
    compare.add_argument("--output", metavar="JSON")
    compare.set_defaults(func=run_compare)

    inspect = subparsers.add_parser("inspect", help="inspect a saved ecosystem or organism")
    inspect.add_argument("snapshot")
    inspect.add_argument("--organism", type=int)
    inspect.add_argument("--weights", action="store_true", help="include all MLP parameters")
    inspect.set_defaults(func=run_inspect)

    tui = subparsers.add_parser("tui", help="observe and control the ecosystem in a terminal")
    tui.add_argument("--seed", type=int, default=3)
    tui.add_argument("--load", metavar="SNAPSHOT")
    tui.add_argument("--snapshot", default="snapshots/latest.eco.gz")
    tui.add_argument("--max-steps", type=int, help=argparse.SUPPRESS)
    tui.add_argument("--save-on-exit", action="store_true")
    learning = tui.add_mutually_exclusive_group()
    learning.add_argument(
        "--learning", action="store_const", const="instinct+learning+memory",
        dest="controller_mode", default=None,
        help="use instinct plus recurrent lifetime learning (default)",
    )
    learning.add_argument(
        "--feedforward-learning", action="store_const", const="instinct+learning",
        dest="controller_mode", help="use the V3-style adaptive policy without memory",
    )
    learning.add_argument(
        "--instinct-only", "--no-learning", action="store_const", const="instinct_only",
        dest="controller_mode", help="use innate behavior without adaptive influence or updates",
    )
    tui.set_defaults(func=run_interactive)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not hasattr(args, "func"):
        parser().print_help()
        return 0
    return args.func(args)
