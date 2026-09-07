# Living Ecosystem

A spatial predator–prey simulation designed to be watched over SSH. Plants grow
across a toroidal landscape, herbivores graze and reproduce, and predators hunt.
Every animal owns a separate 16–10–5 MLP, learns during its life, and gives an
independently copied and mutated version of that network to its offspring.

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
ecosystem tui --no-learning
```

## Fast runs and experiments

Headless batch runs are much faster than the TUI and emit a JSON summary:

```bash
ecosystem run --steps 10000 --seed 3 \
  --metrics runs/seed3.csv --snapshot snapshots/seed3-10000.eco.gz

ecosystem run --load snapshots/seed3-10000.eco.gz --steps 2000 \
  --snapshot snapshots/seed3-12000.eco.gz

ecosystem run --steps 3000 --seed 3 --no-learning
```

Run paired learning/no-learning trials with the same seeds:

```bash
ecosystem compare --steps 3000 --seed 3 --replicates 3 \
  --output runs/comparison.json
```

The summary includes direct behavioral probes. `prey_flee_response` is the mean
probability assigned to the correct opposite movement under four canonical
directional predator cues. `predator_pursuit_response` is the corresponding mean
probability of moving toward prey. These make policy changes visible even when
population counts differ between runs.

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

Each tick, animals pay an idle or movement cost plus a local crowding cost. Food
becomes energy. Mature animals above a reproduction energy threshold may reproduce,
paying energy to create the offspring. Animals die from starvation, predation, or
old age. All rates and life histories are in `src/ecosystem/config.py` and are
embedded into snapshots.

The five actions are stay, north, east, south, and west. Feeding and reproduction
are consequences of biological state rather than separate policy actions.

Each 16-value observation contains:

1. a bias value, normalized energy, normalized age, and plant biomass underfoot;
2. plant signals north/east/south/west, discounted over the species' vision range;
3. herbivore signals in those four directions;
4. predator signals in those four directions.

Organisms cannot see global counts, future state, or target populations.

## Learning and evolution

`TinyMLP` is a dependency-free tanh network with a softmax action head. A new
founder gets independently randomized weights. Action selection has 4% uniform
exploration. After an action, a small immediate policy-gradient update reinforces
or suppresses it relative to that individual's moving reward baseline. Negative
surprises use a 1.8× learning rate, making danger and failed hunts matter quickly.

Rewards are local and survival-related: movement energy, food energy, reproduction,
predator proximity and escape direction, prey pursuit direction, successful or
failed capture, and starvation. This is intentionally simple enough to inspect;
there is no replay buffer, shared optimizer, or species-wide policy.

An offspring starts with a deep copy of its parent's current weights and biases.
Each parameter independently mutates with the configured probability and Gaussian
noise. Its reward baseline is partly inherited, while update counters and action
memory start fresh. Thus lifetime learning can become inherited behavior, then
selection and mutation can modify it across generations.

`--no-learning` freezes within-life updates but retains individual networks,
mutation, inheritance, and ecological selection. It is therefore a comparison of
lifetime plasticity rather than a comparison against motionless agents.

## Snapshots

Snapshots are versioned gzip-compressed JSON. A snapshot records the tick, seed,
learning flag, complete parameter configuration, plant grid, biological fields and
lineage for every living animal, every animal's full MLP and learning baseline,
metrics/history, ID allocator, pending action memory, and Python RNG state. Saving
uses an atomic replace. A deterministic continuation test verifies that the
original and restored simulations remain identical after further steps.

Because the payload is JSON inside gzip, analysis tools can inspect it without
importing this project:

```bash
gzip -dc snapshots/seed3-10000.eco.gz | python -m json.tool | less
```

## Observed dynamics

The tuned default seed is `3`. In a learning-enabled 10,000-step validation run,
all trophic levels persisted: the sampled `(herbivore, predator, plant/cell)` state
moved through `(39,5,6.5)` at step 1,000, `(43,12,5.0)` at 2,000,
`(24,3,3.4)` at 6,000, `(32,10,6.9)` at 8,000, and `(56,6,5.2)` at 10,000.
The run reached generation 40. This shows delayed grazing/resource waves alongside
predator cycles rather than convergence to a fixed count.

The committed `experiments/learning_comparison.json` contains paired 3,000-step
runs for seeds 3–5. Across those trials, learned predator pursuit response averaged
about 0.38 versus 0.12 without learning. Learned prey escape response averaged
about 0.28 versus 0.15 (about 0.22 if the extinct trial is omitted). Learned
predators also spent far less time idle. Individual outcomes remain stochastic:
one no-learning seed suffered complete collapse and some predator lineages became
small. Use several seeds when drawing conclusions.

## Tests

```bash
conda activate eco
python -m unittest discover -s tests -v
```

Tests cover independent and mutated offspring parameters, reinforcement direction,
observation shape, ecological events, deterministic snapshot continuation, and TUI
sparkline rendering. The TUI can be smoke-tested on a headless host with a pseudo
terminal:

```bash
script -q -c 'TERM=xterm ecosystem tui --seed 3 --max-steps 20' /tmp/eco-tui.log
```

`--max-steps` is intentionally hidden from normal help; it exists for automated
terminal verification.
