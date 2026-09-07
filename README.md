# simeco V2 — Instinct / Learning Separation

A spatial predator–prey simulation designed to be watched over SSH. Plants grow
across a toroidal landscape, herbivores graze and reproduce, and predators hunt.
Every animal owns an `InstinctController`, an individual 16–10–5
`AdaptivePolicy`, and an `ActionArbiter`. Instinct supplies survival behavior from
birth; the learned network contributes a bounded refinement and is copied with
mutation into offspring.

There are no target populations, emergency births, carrying-capacity controllers,
or extinction prevention rules. Population balance comes from plant regrowth,
energy intake and costs, crowding, capture probability, reproduction, and aging.

## Setup

The requested environment is described in `environment.yml` and the project has
no runtime dependencies outside Python's standard library.

```bash
conda env create -f environment.yml  # creates the eco environment
conda activate eco
python -m pip install -e .
```

This workspace has already been installed into the local `eco` environment.

## Watch the ecosystem

```bash
conda activate eco
ecosystem tui
```

The map uses `h` for herbivores, `P` for predators, `*` for a crowded cell, and
increasingly dense punctuation for plant biomass. The side panel shows population,
plant biomass, births, deaths, hunt success, learning updates, mean reward,
generation, and population sparklines.

Controls:

| Key | Action |
| --- | --- |
| `space` | pause or resume |
| `+` / `-` | change simulation speed |
| `s` | save the configured snapshot (default `snapshots/latest.eco.gz`) |
| `l` | replace the live run with that snapshot |
| `i` or `n` | cycle through living organisms and show biological/learning state |
| `q` or `Esc` | quit |

Start the TUI from an existing state or save automatically on exit:

```bash
ecosystem tui --load snapshots/latest.eco.gz
ecosystem tui --snapshot snapshots/night-run.eco.gz --save-on-exit
ecosystem tui --instinct-only
```

## Fast runs and experiments

Headless batch runs are much faster than the TUI and emit a JSON summary:

```bash
ecosystem run --steps 10000 --seed 3 \
  --metrics runs/seed3.csv --snapshot snapshots/seed3-10000.eco.gz

ecosystem run --load snapshots/seed3-10000.eco.gz --steps 2000 \
  --snapshot snapshots/seed3-12000.eco.gz

ecosystem run --steps 10000 --seed 3 --instinct-only
```

Run paired instinct-only and instinct+learning trials with the same seeds:

```bash
ecosystem compare --steps 5000 --seed 3 --replicates 3 \
  --output runs/comparison.json
```

The summary includes direct behavioral probes. `prey_flee_response` is the mean
probability assigned by the complete controller to the correct opposite movement
under four canonical predator cues. `predator_pursuit_response` is the probability
of pursuing directional prey. Food per herbivore action and hunts per 1,000
predator actions measure realized efficiency.

Inspect a snapshot or one animal. `--weights` exposes the complete MLP for later
analysis:

```bash
ecosystem inspect snapshots/seed3-10000.eco.gz
ecosystem inspect snapshots/seed3-10000.eco.gz --organism 42
ecosystem inspect snapshots/seed3-10000.eco.gz --organism 42 --weights
```

## Simulation rules

The world is a wrapping 48×22 grid. Every cell contains continuous plant biomass.
Plants follow local logistic growth and weak diffusion from neighboring cells.
Herbivores automatically graze after moving; predators can capture a co-located
herbivore with a configurable probability. Failed captures reward the escaping
prey and penalize the predator.

Each tick, animals pay an idle or movement cost plus a local crowding cost. Nearby
predators also pay interference costs when competing for the same local prey. Food
becomes energy. Mature animals above a reproduction energy threshold may reproduce,
paying energy to create the offspring. Eligible adults accumulate reproductive
readiness instead of winning a per-tick birth lottery; readiness is consumed at
birth. Animals die from starvation, predation, or old age. All rates and life
histories are in `src/ecosystem/config.py` and are embedded into snapshots.

The five actions are stay, north, east, south, and west. Feeding and reproduction
are consequences of biological state rather than separate policy actions.

Each 16-value observation contains:

1. a bias value, normalized energy, normalized age, and plant biomass underfoot;
2. plant signals north/east/south/west, discounted over the species' vision range;
3. herbivore signals in those four directions;
4. predator signals in those four directions.

Organisms cannot see global counts, future state, or target populations.

## Controller architecture

Each organism owns three separate controller objects:

1. `InstinctController` produces fixed species-specific action preferences. A
   herbivore rests to eat, seeks richer plant cells, and flees opposite a predator.
   A predator follows prey signals and conserves energy when no prey is visible.
   Energy, maturity, and reproductive readiness provide innate reproduction.
2. `AdaptivePolicy` produces learned residual preferences. It is an independent
   tanh MLP for every animal; no weights, optimizer state, or replay data are shared.
3. `ActionArbiter` adds weighted instinct and adaptive preferences, applies a
   temperature and 3.5% exploration, then samples one of stay/north/east/south/west.

The arbiter weights instinct at `1.0` and the adaptive residual at `0.12`. Adaptive
preferences are bounded to `[-1, 1]`, so learning can change ambiguous feeding,
search, and energy decisions but cannot erase a strong innate escape or pursuit
response. In instinct-only mode the adaptive weight is exactly zero.

## Learning and evolution

`AdaptivePolicy` is a dependency-free tanh network. A new founder gets independently
randomized weights. After arbitration, a small immediate policy-gradient update
reinforces or suppresses the chosen residual relative to that individual's moving
reward baseline. The gradient includes the arbiter mixing weight and residual
bound. Negative surprises use a 1.8× learning rate.

Rewards are outcome-oriented: energy cost, food energy, successful reproduction,
surviving a tick, successful or failed capture, escape from a failed attack, and
starvation. Directional flee and pursuit rewards from V1 are gone; that knowledge
now lives entirely in `InstinctController`. There is no replay buffer, shared
optimizer, or species-wide policy.

An offspring starts with a deep copy of its parent's current weights and biases.
Each parameter independently mutates with the configured probability and Gaussian
noise. Its reward baseline is partly inherited, while update counters and action
memory start fresh. Thus lifetime learning can become inherited behavior, then
selection and mutation can modify it across generations.

`--instinct-only` (with `--no-learning` retained as an alias) gives the adaptive
network zero action influence and performs no updates. Individual networks are
still inherited and mutated, allowing the same snapshot shape in both modes.

## Snapshots

V2 snapshots are versioned gzip-compressed JSON. A snapshot records the controller
mode, tick, seed, configuration, plant grid, biological state and lineage, full
adaptive parameters and learning baseline, instinct state, arbiter weights and last
preferences, reproductive readiness, metrics/history, ID allocator, pending action
memory, and Python RNG state. Saving uses an atomic replace. V1 snapshots migrate
to the new controller objects on load. A deterministic continuation test verifies
that the original and restored simulations remain identical after further steps.

Because the payload is JSON inside gzip, analysis tools can inspect it without
importing this project:

```bash
gzip -dc snapshots/seed3-10000.eco.gz | python -m json.tool | less
```

## Observed dynamics

The critical instinct-only gate was run for 10,000 steps on seeds 3–5. All three
trophic levels survived every run. Final herbivore/predator counts were `55/8`,
`62/13`, and `52/6`; all continued to show resource and predator/prey
oscillations. No learning updates occurred.

The committed `experiments/v2_learning_comparison.json` contains paired 5,000-step
runs for the same seeds. Both modes retained all three trophic levels at that
horizon. Adaptive learning increased mean plant food obtained per herbivore action
from `1.069` to `1.481` (about 38.5%) and herbivore reward per step from `41.23` to
`51.74` (about 25.5%). Hunts per 1,000 predator actions rose from `5.32` to `5.67`
(about 6.6%). The innate flee and pursuit probes stayed near `0.958` and `0.857` in
both modes, demonstrating that learning refined outcomes without rediscovering or
replacing core survival responses. Predator population and reward effects were
mixed across seeds because increased hunting can also increase competition.

## Tests

```bash
conda activate eco
python -m unittest discover -s tests -v
```

Tests cover innate flee/pursuit, zero adaptive influence and updates in instinct-only
mode, independent and mutated offspring parameters, reinforcement direction,
observation shape, ecological events, deterministic V2 snapshot continuation, and
TUI sparkline rendering. The TUI can be smoke-tested on a headless host with a
pseudo-terminal:

```bash
script -q -c 'TERM=xterm ecosystem tui --seed 3 --max-steps 20' /tmp/eco-tui.log
```

`--max-steps` is intentionally hidden from normal help; it exists for automated
terminal verification.
