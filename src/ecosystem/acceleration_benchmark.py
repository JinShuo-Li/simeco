"""End-to-end benchmark for legacy and synchronous policy execution."""

from __future__ import annotations

import time
import cProfile
import pstats

from .reporting import lightweight_summary
from .simulation import Simulation
from .synchronous import SynchronousSimulation


def benchmark_backend(backend: str, device: str, steps: int, seed: int) -> dict:
    cls = Simulation if backend == "legacy" else SynchronousSimulation
    kwargs = {} if backend == "legacy" else {"device": device}
    simulation = cls(seed=seed, learning=True, memory=True, social_memory=True, **kwargs)
    # Warm-up is reported separately and not mixed with steady-state tick rate.
    warmup = min(8, max(1, steps // 10))
    warmup_started = time.perf_counter()
    simulation.run(warmup)
    warmup_seconds = time.perf_counter() - warmup_started
    if hasattr(simulation, "timings"):
        for name in simulation.timings:
            if name != "initialization":
                simulation.timings[name] = 0.0
    started = time.perf_counter()
    simulation.run(steps)
    run_seconds = time.perf_counter() - started
    report_started = time.perf_counter()
    report = lightweight_summary(simulation)
    report_seconds = time.perf_counter() - report_started
    if hasattr(simulation, "record_reporting_time"):
        simulation.record_reporting_time(report_seconds)
    timing = getattr(simulation, "timings", {})
    if backend == "legacy":
        profiler = cProfile.Profile()
        profiled = Simulation(seed=seed, learning=True, memory=True, social_memory=True)
        profiled.run(warmup)
        profiler.enable()
        profiled.run(steps)
        profiler.disable()
        stats = pstats.Stats(profiler)
        categories = {name: 0.0 for name in (
            "observation", "policy_forward", "learning", "reporting"
        )}
        for (_, _, function), values in stats.stats.items():
            cumulative = values[3]
            if function == "observe":
                categories["observation"] += cumulative
            elif function in ("advance", "choose", "choose_communication"):
                categories["policy_forward"] += cumulative
            elif function == "learn":
                categories["learning"] += cumulative
            elif function == "_record_history":
                categories["reporting"] += cumulative
        accounted = sum(categories.values())
        categories["environment"] = max(0.0, stats.total_tt - accounted)
        scale = run_seconds / max(1e-12, stats.total_tt)
        timing = {name: value * scale for name, value in categories.items()}
        timing["reporting"] += report_seconds
    return {
        "backend": backend, "device": device, "seed": seed, "steps": steps,
        "population_end": len(simulation.organisms),
        "warmup_seconds": round(warmup_seconds, 6),
        "run_seconds": round(run_seconds, 6),
        "ticks_per_second": round(steps / run_seconds, 4),
        "time_breakdown_seconds": {
            name: round(value, 6) for name, value in timing.items()
            if name != "initialization"
        } if timing else {"total_simulation": round(run_seconds, 6), "reporting": round(report_seconds, 6)},
        "lightweight_report": report,
    }


def benchmark_suite(steps: int = 100, seed: int = 3, include_xpu: bool = True) -> dict:
    legacy = benchmark_backend("legacy", "cpu", steps, seed)
    synchronous_cpu = benchmark_backend("synchronous", "cpu", steps, seed)
    runs = {"legacy_cpu": legacy, "synchronous_cpu": synchronous_cpu}
    if include_xpu:
        runs["synchronous_xpu"] = benchmark_backend("synchronous", "xpu", steps, seed)
    reference = legacy["run_seconds"]
    return {
        "runs": runs,
        "speedup_vs_legacy": {
            name: round(reference / result["run_seconds"], 4)
            for name, result in runs.items() if name != "legacy_cpu"
        },
    }
