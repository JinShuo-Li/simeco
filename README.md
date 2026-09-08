# simeco V3 — Embodied Perception & Action

A spatial predator–prey ecosystem built for a headless Linux terminal. Plants
grow over a toroidal grid, herbivores graze, predators hunt, and both animal
populations reproduce, age, compete, and die. Population targets, emergency
births, and extinction-prevention rules are absent.

Every animal has an orientation and owns three controller objects:

- a species-specific `InstinctController` with fixed survival rules;
- its own 33–10–12 `AdaptivePolicy` MLP and learning state;
- an `ActionArbiter` that mixes instinct and learned residual preferences.

Offspring receive a mutated deep copy of the parent's learned policy. There is no
shared species network.

## Setup

`environment.yml` creates the requested Conda environment. Runtime code uses
only the Python standard library.

```bash
conda env create -f environment.yml
conda activate eco
python -m pip install -e .
```

## TUI

```bash
conda activate eco
ecosystem tui
ecosystem tui --instinct-only
ecosystem tui --load snapshots/latest.eco.gz
```

The map shows herbivores as `h`, predators as `P`, crowded cells as `*`, and
plant biomass with increasingly dense punctuation. The panel reports populations,
resources, births, starvation, hunt success, learning updates, generation, and
population sparklines. Inspecting an animal shows its heading and last four-head
action.

| Key | Action |
| --- | --- |
| `space` | pause or resume |
| `+` / `-` | change speed |
| `s` | save the configured snapshot |
| `l` | restore that snapshot |
| `i` or `n` | inspect the next living animal |
| `q` or `Esc` | quit |

## Headless runs

```bash
ecosystem run --steps 10000 --seed 3 --instinct-only
ecosystem run --steps 10000 --seed 3 --learning \
  --metrics runs/seed3.csv --snapshot snapshots/seed3.eco.gz
ecosystem run --load snapshots/seed3.eco.gz --steps 2000 \
  --snapshot snapshots/seed3-resumed.eco.gz
ecosystem compare --steps 5000 --seed 3 --replicates 3 \
  --output runs/comparison.json
```

`--no-learning` is an alias for `--instinct-only`. Inspect saved state and
optionally emit every individual MLP parameter:

```bash
ecosystem inspect snapshots/seed3.eco.gz
ecosystem inspect snapshots/seed3.eco.gz --organism 42
ecosystem inspect snapshots/seed3.eco.gz --organism 42 --weights
```

## Perception

The compact observation has 33 normalized values:

1. six self-state values: bias, energy fraction, age fraction, hunger, resource
   underfoot, and reproductive readiness;
2. a body-relative 3×3 plant patch with forward/back and left/right axes;
3. 3×3 egocentric herbivore-density sectors;
4. 3×3 egocentric predator-density sectors.

Animal signals cover only the species' configured Manhattan vision radius. Their
strength falls with distance and accumulates when several animals occupy a sector.
The plant patch covers adjacent cells. Rotating changes the egocentric encoding;
controllers never receive absolute compass direction, global counts, identity, or
history.

## Multi-head actions

The controller samples four heads independently on every tick:

| Head | Choices |
| --- | --- |
| locomotion | hold, forward, turn left, turn right |
| effort | low, cruise, sprint |
| interaction | none, feed, attack |
| reproduction | defer, intend |

Turning changes heading. A cruise or sprint turn also advances in the new
direction; a low-effort turn rotates in place. Sprint covers two cells and costs
more energy. Feeding, attacking, and reproduction happen only when the
corresponding intent is selected and the environment's biological conditions are
met. Herbivores cannot attack, predators cannot eat plants, attacks require
co-location, and reproduction still requires maturity, energy, and accumulated
readiness.

## Instinct, learning, and arbitration

Herbivore instinct approaches local plants when hungry, feeds on occupied plant
cells, turns away and sprints from predators, conserves effort when safe and
satiated, and expresses reproduction intent when mature and energetic. Predator
instinct turns toward prey, attacks co-located prey, uses cruise effort during
pursuit, searches when hungry, conserves effort with no prey, and gates
reproduction by physiology.

The adaptive MLP receives the same observation and emits one bounded residual for
each of the 12 action logits. The arbiter adds instinct at weight `1.0` and the
adaptive residual at `0.12`, applies temperature `0.85` and 3.5% exploration,
then samples each head separately. Strong survival instincts therefore work from
birth while learning can refine effort, feeding, attacks, movement, and
reproductive timing.

Learning is a cheap immediate policy-gradient update against the individual's
moving reward baseline. Negative surprises use a 1.8× rate. Outcomes include
energy spent, food gained, capture success or failure, escape, survival,
reproduction, and starvation. Directional pursuit and flight are innate and have
no reward-shaping terms. In instinct-only mode the MLP is neither evaluated nor
updated and its arbiter weight is exactly zero.

## Ecological rules

Continuous plants regrow logistically and diffuse between adjacent cells. Every
action spends energy according to movement and effort; crowding and predator
competition add costs. Food restores energy. Eligible adults accumulate
reproductive readiness, spend energy at birth, and create an offspring nearby.
Animals die through starvation, predation, or old age. All ecological parameters
live in `src/ecosystem/config.py` and are embedded in snapshots.

The tuned V3 default uses low predator metabolism and slow predator reproduction.
This lets a lineage wait through prey troughs without a population stabilizer,
while finite prey reproduction and capture probability prevent unchecked predator
growth.

## Snapshots

V3 snapshots are versioned gzip-compressed JSON. They contain the step, seed,
mode, full configuration, plant grid, every organism's physiology, orientation,
lineage, individual MLP parameters and learning memory, instinct state, arbiter
weights and last per-head decisions, reproductive readiness, metrics/history, ID
allocator, and Python RNG state. Saves use an atomic replacement. V1/V2 snapshots
are rejected because their observation and action dimensions cannot drive a V3
controller safely.

```bash
gzip -dc snapshots/seed3.eco.gz | python -m json.tool | less
```

The deterministic continuation test advances an original and restored simulation
and compares resources, organisms, controller state, metrics, and actions exactly.

## Validation and observed behavior

The critical instinct-only gate ran seeds 3–5 for 12,000 steps. All three trophic
levels survived, including after the 9,000-step predator founder lifespan. Final
herbivore/predator populations were `56/3`, `66/7`, and `98/4`; predator
lineages reached generations 3, 2, and 2, while herbivore lineages reached
generation 33.
Populations rose and fell without target-size logic.

The behavioral probes expose action probabilities under controlled observations:
hungry herbivores approach food (`0.6712`), prey turn from danger (`0.9772`)
and sprint (`0.9205`), predators pursue visible prey (`0.8664`), attack
co-located prey (`0.9764`), and choose low effort while satiated with no prey
(`0.7821`). Hunger raises food approach probability by `0.3762`, feeding by
`0.0480`, and predator search by `0.2169`. Run summaries also report food and
hunt energy efficiency, sprint fractions, unnecessary sprinting, reproductive
intent efficiency, starvation, survival, and rewards.

Representative paired learning results are stored in
`experiments/v3_embodied_comparison.json`. Across three paired 5,000-step runs,
learning reduced herbivore sprinting by 5.3%, unnecessary sprinting by 6.8%, and
starvation deaths from three to two. Predator reward improved by 2.9% and hunts per
1,000 actions improved by 0.4%. Herbivore food per action fell 3.5% and food-energy
efficiency fell 2.1%, so V3 learning shows measurable energy restraint and modest
predator refinement rather than a universal fitness gain. The paired seeds and
mixed result are retained in full instead of selecting favorable runs.

## Tests

```bash
conda activate eco
python -m unittest discover -s tests -v
```

Tests cover rotated perception, distant egocentric sectors, physiology, all action
heads, instinct responses, zero adaptive influence in instinct-only mode,
individual and inherited policies, learning direction, ecological events,
deterministic V3 snapshot continuation, and TUI rendering.

Headless TUI smoke test:

```bash
script -q -c 'TERM=xterm ecosystem tui --seed 3 --max-steps 20' /tmp/eco-tui.log
```

`--max-steps` is hidden from normal help and exists for automated terminal
verification.
