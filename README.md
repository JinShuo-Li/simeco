# simeco V5 — Social Perception & Individual Memory

A spatial predator–prey ecosystem built for a headless Linux terminal. Plants
grow over a toroidal grid, herbivores graze, predators hunt, and both animal
populations reproduce, age, compete, and die. Population targets, emergency
births, and extinction-prevention rules are absent.

V5 preserves V4.1's embodiment and temporal learning and adds local recognition
of individuals. Every animal has an
orientation and owns three controller objects:

- a species-specific `InstinctController` with fixed survival rules;
- its own adaptive 33-value ecological input, eight individual slots, 16-value
  encoder, 12-value recurrent state, 12 action logits, and private social table;
- an `ActionArbiter` that mixes instinct and learned residual preferences.

Offspring receive a mutated deep copy of the parent's learned parameters, while
runtime temporal and social memories start fresh. There is no shared species
network, social memory, communication, or group label.

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
action, recurrent-memory activity, and most recent outcome.

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
ecosystem run --steps 5000 --seed 3 --feedforward-learning
ecosystem run --steps 5000 --seed 3 --temporal-learning
ecosystem run --steps 5000 --seed 3 --social-learning
ecosystem run --load snapshots/seed3.eco.gz --steps 2000 \
  --snapshot snapshots/seed3-resumed.eco.gz
ecosystem compare --steps 5000 --seed 3 --replicates 3 \
  --output runs/comparison.json
ecosystem benchmark --episodes 6000 --seed 41 \
  --output runs/delayed-cue.json
ecosystem social-benchmark --episodes 5000 --seed 53 \
  --output runs/social-cue.json
```

`--no-learning` is an alias for `--instinct-only`. Inspect saved state and
optionally emit every individual MLP parameter:

```bash
ecosystem inspect snapshots/seed3.eco.gz
ecosystem inspect snapshots/seed3.eco.gz --organism 42
ecosystem inspect snapshots/seed3.eco.gz --organism 42 --weights
```

## Perception

Instinct continues to receive the unchanged compact observation of 33 normalized
values:

1. six self-state values: bias, energy fraction, age fraction, hunger, resource
   underfoot, and reproductive readiness;
2. a body-relative 3×3 plant patch with forward/back and left/right axes;
3. 3×3 egocentric herbivore-density sectors;
4. 3×3 egocentric predator-density sectors.

Animal signals cover only the species' configured Manhattan vision radius. Their
strength falls with distance and accumulates when several animals occupy a sector.
The plant patch covers adjacent cells. Rotating changes the egocentric encoding;
the ecological encoder never supplies absolute compass direction, global counts,
identity, or history.

In social mode the adaptive controller also sees the nearest eight individuals
inside the observer's existing vision radius. Each 14-value physical slot contains
egocentric forward/right offset, distance, relative velocity, relative heading
as sine/cosine, same/different-species bits, body condition rounded to quarters,
and the four currently visible action choices. A four-value retrieved social
embedding completes each 18-value slot. Exact energy, reward, hidden state, and
numeric ID never enter policy input. Slot order is distance-first; IDs are used
only inside the engine to retrieve private memory.

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

## Instinct, temporal learning, social memory, and arbitration

Herbivore instinct approaches local plants when hungry, feeds on occupied plant
cells, turns away and sprints from predators, conserves effort when safe and
satiated, and expresses reproduction intent when mature and energetic. Predator
instinct turns toward prey, attacks co-located prey, uses cruise effort during
pursuit, searches when hungry, conserves effort with no prey, and gates
reproduction by physiology.

The adaptive policy encodes the ecological and optional social input through 16 tanh units. A
12-value GRU-like state uses separate update, reset, and candidate gates. Its
input is the encoding plus a 12-value one-hot representation of the previous
four-head action and the previous four head-specific outcomes. The current
encoding and new memory emit one bounded residual for each of the 12 action
logits. Memory-to-action weights start at exactly zero, so an untrained animal has
no random temporal influence even though its gates can begin forming state.

Recurrent learning buffers eight transitions per organism and applies truncated
backpropagation through time. Each action head receives its own discounted return
with `gamma=0.95`; later food, capture, escape, reproduction, injury, or energy
outcomes can therefore update earlier states and choices. Gradients are clipped,
the short trajectory objective is normalized by its length, and negative
advantages retain the 1.8× response. There is no replay buffer, PPO, framework,
planning, prediction head, communication, or shared memory. Instinct remains entirely
memoryless. Feedforward mode keeps immediate, per-head V3-style updates.

Each animal's private table maps an engine-side organism ID to a four-value
learned embedding, encounter count, last-seen tick, and bounded outcome trace.
New entries start with a zero embedding. TBPTT updates an embedding through the
same policy gradient as the neural parameters and associates discounted later
outcomes with individuals present earlier in the trajectory. This association
contains no friend, enemy, leader, partner, or cooperation label and adds no
directional reward.

The table holds at most 64 individuals. Entries unseen for 600 ticks expire; when
capacity is exceeded, the least recently seen and least encountered entries leave
first. Offspring inherit mutated neural parameters but receive an empty social
table, empty trajectory, zero recurrent state, and no recent action/outcome
context.

The arbiter adds instinct at weight `1.0` and the adaptive residual at `0.12`,
applies temperature `0.85` and 3.5% exploration, then samples each head
separately. Strong survival instincts therefore work from birth while recurrent
learning can refine effort, feeding, attacks, movement, and reproductive timing.

Learning uses a policy-gradient update against the individual's moving per-head
reward baselines. Outcomes include
energy spent, food gained, capture success or failure, escape, survival,
reproduction, and starvation. Directional pursuit and flight are innate and have
no reward-shaping terms. `--instinct-only` neither evaluates nor updates the
adaptive layer. `--feedforward-learning` provides the V3-style baseline,
`--temporal-learning` enables V4.1 recurrence without individual memory, and
`--social-learning` enables the complete V5 controller.

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

V5 snapshots are versioned gzip-compressed JSON. They contain the step, seed,
mode, full configuration, plant grid, every organism's physiology, orientation,
lineage, encoder and all GRU gate parameters, current 12-value memory, the
partially filled trajectory with its recurrent activations, previous action and
per-head outcomes, learning baselines, instinct state, arbiter weights and last
decisions, reproductive readiness, metrics/history, ID allocator, and Python RNG
state, plus every social entry, embedding, encounter count, last-seen tick,
outcome trace, and last visible IDs used by metrics. Saves use an atomic
replacement. Earlier snapshot versions are rejected.

```bash
gzip -dc snapshots/seed3.eco.gz | python -m json.tool | less
```

The deterministic continuation test advances an original and restored simulation
and compares resources, organisms, controller state, metrics, and actions exactly.

## V5 validation and observed behavior

The critical instinct-only gate ran seeds 3–5 for 12,000 steps. It reproduced the
V3 population trajectory at every 2,000-step checkpoint, showing that private
recurrent initialization and mutation do not consume ecological RNG. All trophic
levels survived beyond the 9,000-step predator founder lifespan. Final
herbivore/predator populations were `56/3`, `66/7`, and `98/4`; predator
lineages reached generations 3, 2, and 2, while herbivore lineages reached
generation 33. Populations rose and fell without target-size logic.

The behavioral probes expose action probabilities under controlled observations:
hungry herbivores approach food (`0.6712`), prey turn from danger (`0.9772`)
and sprint (`0.9205`), predators pursue visible prey (`0.8664`), attack
co-located prey (`0.9764`), and choose low effort while satiated with no prey
(`0.7821`). Hunger raises food approach probability by `0.3762`, feeding by
`0.0480`, and predator search by `0.2169`. Run summaries also report food and
hunt energy efficiency, sprint fractions, unnecessary sprinting, reproductive
intent efficiency, starvation, survival, and rewards.

The retained delayed-cue benchmark presents LEFT or RIGHT, replaces it with
identical blank observations for delays 1, 2, 4, and 8, and rewards only the final
choice. Seed 41 scored 50% at every delay before training and 100% at every delay
after 6,000 balanced episodes. This is the direct percentage-scale evidence that
identical current input produces different learned actions from prior context.
The command reports all delay accuracies and uses a nine-transition truncation so
the cue plus eight blanks remain in one graph.

Ecological counterfactuals isolate previous perception, action, outcome, and full
natural history while holding the current blank perception identical. Across
seeds 3–5 after 5,000 steps, full vanished-food context changed action probability
by `0.00164`, `0.00236`, `0.00294`, and `0.00334` at delays 1, 2, 4,
and 8. Post-danger effects were `0.00163`, `0.00245`, `0.00330`, and
`0.00390`; lost-prey pursuit effects were smaller at `0.00010` through
`0.00022`. For food at delay 2, perception-only, action-only, outcome-only,
and full effects were `0.00029`, `0.00199`, `0.00124`, and `0.00236`.
These natural effects persist across the measured horizon but remain below one
percentage point under the conservative unchanged arbiter.

The social benchmark repeatedly pairs one internal identity with a rewarded
forward choice and another with a rewarded turn while every visible physical
value remains identical. Seed 53 moved from 50% to 100% accuracy after 5,000
episodes; forward probability became 88.1% versus 11.9%. Swapping the identity
lookup produced 0% accuracy. Resetting the social table or disabling it produced
50%. This causally isolates learned identity memory.

The four-mode ecological comparison ran every planned seed 3–5 for 2,000 steps.
Mean final populations were `157.7/4.3` instinct, `155.3/4.0` feedforward,
`156.0/4.3` temporal, and `158.3/4.3` social. No social or temporal run
starved. Relative to temporal learning, social mode raised hunt success from
21.7% to 22.3%, lowered herbivore sprinting 1.7%, and improved predator reward
4.7%; food and hunt energy efficiencies fell 2.1% and 10.5%.

About 79.4% of social-mode encounters repeated from the preceding tick, 23.0% of
opportunities with a conspecific ahead resulted in forward movement, and predator
co-location was highly persistent in every mode. Social and temporal following
differed by only 0.14 percentage points, so these runs do not support claims of
learned herding, partner preference, or pack cooperation. Natural good/bad
identity histories separated embeddings by `0.0822`, but changed action
probabilities by only `0.00008`; strong identity-driven behavior appeared in
the controlled task, not robustly in the ecology.

The complete protocol and aggregate results are in
`experiments/v5_social_validation.json`. Every consecutive seed is retained.

## Tests

```bash
conda activate eco
python -m unittest discover -s tests -v
```

Tests cover rotated perception, distant egocentric sectors, physiology, all action
heads, instinct responses, independent recurrent parameters and state, temporal
context dependence under identical current input, fresh offspring memory,
head-specific learning, feedforward mode, ecological events, deterministic V5
snapshot continuation, social-table eviction, identity-specific action,
social-memory ablations, and TUI rendering.

Headless TUI smoke test:

```bash
script -q -c 'TERM=xterm ecosystem tui --seed 3 --max-steps 20' /tmp/eco-tui.log
```

`--max-steps` is hidden from normal help and exists for automated terminal
verification.
