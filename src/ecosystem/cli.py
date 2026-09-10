"""Command line interface for interactive and batch simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reporting import print_summary, summary, write_history
from .benchmark import run_memory_benchmark
from .social_benchmark import run_social_benchmark
from .communication_benchmark import run_communication_benchmark
from .simulation import Simulation
from .synchronous import SynchronousSimulation
from .snapshot import load_snapshot, save_snapshot
from .v6_validation import run_v6_validation
from .acceleration_benchmark import benchmark_suite


def _simulation(args: argparse.Namespace) -> Simulation:
    requested_mode = getattr(args, "controller_mode", None)
    ablation = getattr(args, "social_ablation", "none")
    communication_ablation = getattr(args, "communication_ablation", "none")
    communication = not getattr(args, "no_communication", False)
    if getattr(args, "load", None):
        simulation = load_snapshot(args.load)
        if requested_mode is not None:
            simulation.learning = requested_mode != "instinct_only"
            simulation.memory = requested_mode in (
                "instinct+learning+memory", "instinct+learning+memory+social"
            )
            simulation.social_memory = requested_mode == "instinct+learning+memory+social"
        simulation.social_identity_shuffle = ablation == "identity-shuffled"
        simulation.social_embeddings = ablation != "embeddings-disabled"
        simulation.reverse_entity_order = ablation == "order-reversed"
        simulation.communication = communication
        simulation.transmission_enabled = communication_ablation != "sender-disabled"
        simulation.inbox_enabled = communication_ablation != "inbox-disabled"
        simulation.token_permutation = (
            [0, 2, 3, 4, 5, 6, 7, 8, 1]
            if communication_ablation == "tokens-permuted" else None
        )
        simulation.randomize_received_tokens = communication_ablation == "tokens-randomized"
        simulation.randomize_signal_strengths = communication_ablation == "strengths-randomized"
        simulation.communication_sender_identity_shuffle = (
            communication_ablation == "sender-identity-shuffled"
        )
        return simulation
    mode = requested_mode or "instinct+learning+memory+social"
    simulation_type = (
        Simulation
        if getattr(args, "backend", "synchronous") == "legacy" or mode == "instinct_only"
        else SynchronousSimulation
    )
    extra = {}
    if simulation_type is SynchronousSimulation:
        device = getattr(args, "device", "auto")
        if device == "auto":
            try:
                import torch
                device = "xpu" if torch.xpu.is_available() else "cpu"
            except (ImportError, AttributeError):
                device = "cpu"
        extra["device"] = device
    return simulation_type(
        seed=args.seed,
        learning=mode != "instinct_only",
        memory=mode in ("instinct+learning+memory", "instinct+learning+memory+social"),
        social_memory=mode == "instinct+learning+memory+social",
        social_identity_shuffle=ablation == "identity-shuffled",
        social_embeddings=ablation != "embeddings-disabled",
        reverse_entity_order=ablation == "order-reversed",
        communication=communication,
        transmission_enabled=communication_ablation != "sender-disabled",
        inbox_enabled=communication_ablation != "inbox-disabled",
        token_permutation=(
            [0, 2, 3, 4, 5, 6, 7, 8, 1]
            if communication_ablation == "tokens-permuted" else None
        ),
        randomize_received_tokens=communication_ablation == "tokens-randomized",
        randomize_signal_strengths=communication_ablation == "strengths-randomized",
        communication_sender_identity_shuffle=(
            communication_ablation == "sender-identity-shuffled"
        ),
        **extra,
    )


def run_batch(args: argparse.Namespace) -> int:
    simulation = _simulation(args)
    reset_at=getattr(args,"reset_social_at",None)
    if reset_at is not None and 0<reset_at<args.steps:
        simulation.run(reset_at)
        for animal in simulation.organisms.values():
            animal.adaptive_policy.reset_social_memory()
        simulation.run(args.steps-reset_at)
    else:
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
        for learning, memory, social in (
            (False, False, False), (True, False, False),
            (True, True, False), (True, True, True),
        ):
            simulation = Simulation(
                seed=seed, learning=learning, memory=memory, social_memory=social
            )
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
            sum(len(row) for row in policy["entity_w"])
            + len(policy["entity_b"])
            + policy["hidden"] * policy["inputs"]
            + policy["hidden"]
            + policy["outputs"] * policy["hidden"]
            + policy["outputs"]
            + sum(len(row) for name in ("wz", "uz", "wr", "ur", "wh", "uh") for row in policy[name])
            + 3 * policy["memory_size"]
            + policy["outputs"] * policy["memory_size"]
        ),
        "tbptt_updates": policy["tbptt_updates"],
        "buffered_transitions": len(policy["trajectory"]),
        "social_memory_entries": len(policy["social_memory"]),
        "entity_encoder_shape": [len(policy["entity_w"][0]), len(policy["entity_w"])],
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


def run_benchmark(args: argparse.Namespace) -> int:
    result = run_memory_benchmark(args.seed, args.episodes, args.trials)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_social_benchmark_command(args: argparse.Namespace) -> int:
    result = run_social_benchmark(args.seed, args.episodes)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_communication_benchmark_command(args: argparse.Namespace) -> int:
    rows = [
        run_communication_benchmark(seed, args.episodes, args.trials)
        for seed in range(args.seed, args.seed + args.replicates)
    ]
    result = {"runs": rows}
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_v6_validation_command(args: argparse.Namespace) -> int:
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "xpu" if torch.xpu.is_available() else "cpu"
        except (ImportError, AttributeError):
            device = "cpu"
    result = run_v6_validation(
        steps=args.steps,
        seeds=range(args.seed, args.seed + args.replicates),
        output=args.output,
        device=device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_acceleration_benchmark_command(args: argparse.Namespace) -> int:
    result = benchmark_suite(args.steps, args.seed, not args.no_xpu)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
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
    batch.add_argument(
        "--backend", choices=("synchronous", "legacy"), default="synchronous",
        help="tick execution model (default: synchronous batched policies)",
    )
    batch.add_argument(
        "--device", choices=("auto", "cpu", "xpu"), default="auto",
        help="resident policy tensor device for the synchronous backend",
    )
    batch.add_argument(
        "--social-ablation",
        choices=("none","identity-shuffled","embeddings-disabled","order-reversed"),
        default="none",
    )
    batch.add_argument("--reset-social-at", type=int, metavar="STEP")
    batch.add_argument(
        "--no-communication", action="store_true",
        help="run the V5.1 social learner without communication",
    )
    batch.add_argument(
        "--communication-ablation",
        choices=(
            "none", "sender-disabled", "inbox-disabled", "tokens-permuted",
            "tokens-randomized", "strengths-randomized", "sender-identity-shuffled",
        ),
        default="none",
    )
    learning = batch.add_mutually_exclusive_group()
    learning.add_argument(
        "--learning", "--social-learning", action="store_const",
        const="instinct+learning+memory+social",
        dest="controller_mode", default=None,
        help="use temporal plus social-memory learning (default)",
    )
    learning.add_argument(
        "--temporal-learning", action="store_const", const="instinct+learning+memory",
        dest="controller_mode", help="use recurrent learning without individual memory",
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
        "compare", help="compare instinct, feed-forward, temporal, and social modes"
    )
    compare.add_argument("--steps", type=int, default=1500)
    compare.add_argument("--seed", type=int, default=3)
    compare.add_argument("--replicates", type=int, default=3)
    compare.add_argument("--output", metavar="JSON")
    compare.set_defaults(func=run_compare)

    benchmark = subparsers.add_parser("benchmark", help="train the delayed-cue memory task")
    benchmark.add_argument("--seed", type=int, default=41)
    benchmark.add_argument("--episodes", type=int, default=6000)
    benchmark.add_argument("--trials", type=int, default=400)
    benchmark.add_argument("--output", metavar="JSON")
    benchmark.set_defaults(func=run_benchmark)

    social_benchmark = subparsers.add_parser(
        "social-benchmark", help="train and ablate individual-specific associations"
    )
    social_benchmark.add_argument("--seed", type=int, default=53)
    social_benchmark.add_argument("--episodes", type=int, default=5000)
    social_benchmark.add_argument("--output", metavar="JSON")
    social_benchmark.set_defaults(func=run_social_benchmark_command)

    communication_benchmark = subparsers.add_parser(
        "communication-benchmark", help="train and causally ablate hidden-cue signaling"
    )
    communication_benchmark.add_argument("--seed", type=int, default=3)
    communication_benchmark.add_argument("--replicates", type=int, default=5)
    communication_benchmark.add_argument("--episodes", type=int, default=4000)
    communication_benchmark.add_argument("--trials", type=int, default=400)
    communication_benchmark.add_argument("--output", metavar="JSON")
    communication_benchmark.set_defaults(func=run_communication_benchmark_command)

    validation = subparsers.add_parser(
        "v6-validate", help="run the short-run paired ecosystem communication study"
    )
    validation.add_argument("--steps", type=int, default=2000)
    validation.add_argument("--seed", type=int, default=3)
    validation.add_argument("--replicates", type=int, default=5)
    validation.add_argument("--output", default="experiments/v6_short_run_validation.json")
    validation.add_argument("--device", choices=("auto", "cpu", "xpu"), default="auto")
    validation.set_defaults(func=run_v6_validation_command)

    acceleration = subparsers.add_parser(
        "acceleration-benchmark", help="compare legacy CPU and synchronous CPU/XPU ticks"
    )
    acceleration.add_argument("--steps", type=int, default=100)
    acceleration.add_argument("--seed", type=int, default=3)
    acceleration.add_argument("--no-xpu", action="store_true")
    acceleration.add_argument("--output", metavar="JSON")
    acceleration.set_defaults(func=run_acceleration_benchmark_command)

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
    tui.add_argument("--backend", choices=("synchronous", "legacy"), default="synchronous")
    tui.add_argument("--device", choices=("auto", "cpu", "xpu"), default="auto")
    learning = tui.add_mutually_exclusive_group()
    learning.add_argument(
        "--learning", "--social-learning", action="store_const",
        const="instinct+learning+memory+social",
        dest="controller_mode", default=None,
        help="use temporal plus social-memory learning (default)",
    )
    learning.add_argument(
        "--temporal-learning", action="store_const", const="instinct+learning+memory",
        dest="controller_mode", help="use recurrent learning without individual memory",
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
