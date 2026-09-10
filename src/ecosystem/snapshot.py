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
SNAPSHOT_VERSION = 8


def save_snapshot(simulation: Simulation, path: str | Path) -> Path:
    """Atomically save all continuation and analysis state as compressed JSON."""
    if hasattr(simulation, "policy_store"):
        # Serialized snapshots are explicit TBPTT boundaries. This applies all
        # buffered experience before materializing the resident tensor store.
        simulation.policy_store.learn_trajectory()
        simulation.synchronize_policy_state()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": SNAPSHOT_FORMAT,
        "version": SNAPSHOT_VERSION,
        "backend": getattr(simulation, "backend", "legacy"),
        "step": simulation.step_count,
        "seed": simulation.seed,
        "learning": simulation.learning,
        "controller_mode": simulation.controller_mode,
        "memory": simulation.memory,
        "social_memory": simulation.social_memory,
        "social_identity_shuffle": simulation.social_identity_shuffle,
        "social_embeddings": simulation.social_embeddings,
        "reverse_entity_order": simulation.reverse_entity_order,
        "communication": simulation.communication,
        "transmission_enabled": simulation.transmission_enabled,
        "inbox_enabled": simulation.inbox_enabled,
        "token_permutation": simulation.token_permutation,
        "randomize_received_tokens": simulation.randomize_received_tokens,
        "randomize_signal_strengths": simulation.randomize_signal_strengths,
        "communication_sender_identity_shuffle": simulation.communication_sender_identity_shuffle,
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


def load_snapshot(path: str | Path, device: str = "cpu") -> Simulation:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format") != SNAPSHOT_FORMAT:
        raise ValueError("not a Living Ecosystem snapshot")
    if payload.get("version") != SNAPSHOT_VERSION:
        raise ValueError(f"unsupported snapshot version: {payload.get('version')}")

    if payload.get("backend", "legacy") == "synchronous":
        from .synchronous import SynchronousSimulation
        simulation = SynchronousSimulation.__new__(SynchronousSimulation)
    else:
        simulation = Simulation.__new__(Simulation)
    simulation.config = WorldConfig.from_dict(payload["config"])
    simulation.seed = payload["seed"]
    simulation.learning = payload["learning"]
    simulation.memory = payload["memory"]
    simulation.social_memory = payload["social_memory"]
    simulation.social_identity_shuffle = payload["social_identity_shuffle"]
    simulation.social_embeddings = payload["social_embeddings"]
    simulation.reverse_entity_order = payload["reverse_entity_order"]
    simulation.communication = payload["communication"]
    simulation.transmission_enabled = payload["transmission_enabled"]
    simulation.inbox_enabled = payload["inbox_enabled"]
    simulation.token_permutation = payload["token_permutation"]
    simulation.randomize_received_tokens = payload["randomize_received_tokens"]
    simulation.randomize_signal_strengths = payload["randomize_signal_strengths"]
    simulation.communication_sender_identity_shuffle = payload[
        "communication_sender_identity_shuffle"
    ]
    simulation.step_count = payload["step"]
    simulation.next_id = payload["next_id"]
    simulation.resources = payload["resources"]
    animals = [Organism.from_dict(item) for item in payload["organisms"]]
    simulation.organisms = {animal.id: animal for animal in animals}
    simulation.metrics = Metrics.from_dict(payload["metrics"])
    simulation.last_events = payload["last_events"]
    simulation.rng = __import__("random").Random()
    simulation.set_rng_state(payload["rng_state"])
    if payload.get("backend", "legacy") == "synchronous":
        from .batched_policy import BatchedPolicyStore
        simulation.device = device
        simulation.policy_store = BatchedPolicyStore(
            simulation.organisms.values(), device=device, capacity=max(128, len(animals) * 2),
            learning_rates={
                "herbivore": simulation.config.herbivore.learning_rate,
                "predator": simulation.config.predator.learning_rate,
            },
        )
        simulation.timings = {
            "observation": 0.0, "policy_forward": 0.0, "learning": 0.0,
            "environment": 0.0, "reporting": 0.0, "initialization": 0.0,
        }
        simulation._pending_transition = None
    return simulation
