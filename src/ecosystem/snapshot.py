"""Versioned, portable simulation snapshots."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path

from .config import WorldConfig
from .model import Metrics, Organism
from .simulation import Simulation

SNAPSHOT_FORMAT = "living-ecosystem"
SNAPSHOT_VERSION = 6


def save_snapshot(simulation: Simulation, path: str | Path) -> Path:
    """Atomically save all continuation and analysis state as compressed JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": SNAPSHOT_FORMAT,
        "version": SNAPSHOT_VERSION,
        "step": simulation.step_count,
        "seed": simulation.seed,
        "learning": simulation.learning,
        "controller_mode": simulation.controller_mode,
        "memory": simulation.memory,
        "social_memory": simulation.social_memory,
        "next_id": simulation.next_id,
        "config": simulation.config.to_dict(),
        "resources": simulation.resources,
        "organisms": [animal.to_dict() for animal in simulation.organisms.values()],
        "metrics": simulation.metrics.to_dict(),
        "last_events": simulation.last_events,
        "rng_state": simulation.rng_state(),
    }
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                compressed.write(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_snapshot(path: str | Path) -> Simulation:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format") != SNAPSHOT_FORMAT:
        raise ValueError("not a Living Ecosystem snapshot")
    if payload.get("version") != SNAPSHOT_VERSION:
        raise ValueError(f"unsupported snapshot version: {payload.get('version')}")

    simulation = Simulation.__new__(Simulation)
    simulation.config = WorldConfig.from_dict(payload["config"])
    simulation.seed = payload["seed"]
    simulation.learning = payload["learning"]
    simulation.memory = payload["memory"]
    simulation.social_memory = payload["social_memory"]
    simulation.step_count = payload["step"]
    simulation.next_id = payload["next_id"]
    simulation.resources = payload["resources"]
    animals = [Organism.from_dict(item) for item in payload["organisms"]]
    simulation.organisms = {animal.id: animal for animal in animals}
    simulation.metrics = Metrics.from_dict(payload["metrics"])
    simulation.last_events = payload["last_events"]
    simulation.rng = __import__("random").Random()
    simulation.set_rng_state(payload["rng_state"])
    return simulation
