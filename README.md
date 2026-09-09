# simeco V6 — Emergent Communication

A dependency-free spatial predator–prey ecosystem for studying whether costly,
meaning-free signals become useful to individually learning animals. V6 keeps
V5.1 embodiment, temporal learning, ecology, instincts, shared entity encoding,
private identity memory, and the four physical action heads.

## Communication design

The adaptive policy has two additional categorical heads:

| Head | Choices |
| --- | --- |
| signal token | `0` silence; `1..8` arbitrary symbols |
| signal strength | low, medium, high |

The symbols have no built-in meaning. Instinct still returns exactly the 12
preferences for locomotion, effort, interaction, and reproduction; it neither
emits nor interprets signals. With the adaptive layer disabled, animals are
silent.

Non-silent signals cost `0.006`, `0.020`, or `0.060` energy and travel Manhattan
distances `2`, `4`, or `7`. Silence is free and is not transmitted. A receiver
keeps at most six messages for three ticks. Each message exposes only a token
one-hot vector, strength, approximate egocentric direction and distance, age,
and the sender's retrieved private four-value social embedding. Numeric sender
IDs are engine-only dictionary keys and never enter a policy feature.

Physical individuals and messages pass through the same learned 33-to-8 entity
encoder (29 sensor values plus the private embedding). Elementwise mean/max
pooling gives a permutation-insensitive 16-value social aggregate. This joins
the unchanged 33-value ecological observation before the 16-value encoder and
12-value recurrent state. The recurrent policy outputs the original four
physical heads plus the two communication heads.

Communication receives only the ecological return already used by V5.1. There
is no reward for sending, receiving, matching symbols, information, coordination,
or helping, and no explicit trust score or high-level social action. Analysis
computes token/context, token/receiver-action, and token/future-outcome mutual
information without feeding those measurements back into learning.

## Setup and commands

```bash
conda env create -f environment.yml
conda activate eco
python -m pip install -e .

# V6 and the paired V5.1 control
ecosystem run --steps 2000 --seed 3 --social-learning
ecosystem run --steps 2000 --seed 3 --social-learning --no-communication

# Controlled hidden-cue benchmark, seeds 3–7
ecosystem communication-benchmark --seed 3 --replicates 5 \
  --episodes 4000 --output experiments/v6_communication_benchmark.json

# Paired short-run study and snapshot-matched interventions, seeds 3–7
ecosystem v6-validate --steps 2000 --seed 3 --replicates 5 \
  --output experiments/v6_short_run_validation.json

# Individual intervention from a snapshot
ecosystem run --load snapshots/v6.eco.gz --steps 1000 \
  --communication-ablation tokens-randomized
```

Communication interventions are `sender-disabled`, `inbox-disabled`,
`tokens-permuted`, `tokens-randomized`, `strengths-randomized`, and
`sender-identity-shuffled`. Existing `--social-ablation embeddings-disabled`
removes retrieved private embeddings. The validation command forks each V6 run
at step 1,000, preserving identical ecosystem, neural, message, trajectory, and
RNG state before applying each intervention.

The requested ecosystem validation in this iteration is exactly 2,000 steps for
every consecutive seed 3–7. These runs are short-run evidence only; they cannot
establish stable conventions, long-term communication, or evolutionary
equilibrium.

## Ecology and learning boundary

Plants regrow and spread on a toroidal grid. Herbivores graze; predators pursue
and attack prey; both species spend energy, reproduce, age, and die. There are
no target populations or extinction prevention rules. Instinct provides only
species survival behavior. Every organism owns its adaptive parameters,
recurrent state, trajectory, and bounded private social table.

Offspring inherit mutated ecological, recurrent, entity, physical-action, and
communication parameters. They start with no inbox, emitted action, recurrent
state, TBPTT trajectory, action/outcome context, or parent social memories.

## Controlled benchmark

The sender sees a hidden binary cue. The receiver sees the sender's later symbol
but never the cue, then selects left or right. Both policies are independent;
the only return is correct task completion. Training uses arbitrary symbols and
a small rollout batch to reduce delayed-return variance. Evaluation reports
pre-training and learned accuracy plus blocked, fixed-permutation, independently
randomized, strength-randomized, sender-identity, and embedding interventions.

This benchmark proves capacity to acquire a protocol, not spontaneous use in the
ecosystem. Natural communication is claimed only if ecosystem interventions
causally change outcomes or receiver behavior.

## Metrics and snapshots

History and summaries include signal/silence rates, token and strength
distributions, communication energy, sender/receiver species, per-sender token
counts, conditional token distributions for local predator, food, prey, hunger,
and recent attack contexts, receiver actions after messages, future returns, and
mutual information. Physical social measures retain spacing, repeated dyads,
following, predator co-location, hunt and escape success, food efficiency,
survival, reproduction, starvation, and total energy expenditure.

V6 snapshot format 8 stores communication parameters through the policy,
current communication actions, bounded inboxes, all social and recurrent state,
partially filled TBPTT trajectories, metrics/history, configuration, organisms,
resources, allocator, and Python RNG state. Continuation is tested value-for-value
against an uninterrupted run.

## V5.1 scientific baseline

V5.1 found useful physical social perception, but natural identity embeddings
were effectively unused: identity counterfactual action effects were below
`0.000005`, mean embedding separation was `0.00035`, and disabling embeddings
reproduced the normal seed-3 trajectory. V6 does not force communication to
matter or tune ecology toward a desired symbol. A valid outcome is controlled
protocol learning alongside insufficient ecological incentive for spontaneous
communication.

## Tests

```bash
python -m unittest discover -s tests -v
script -q -c 'TERM=xterm ecosystem tui --seed 3 --max-steps 20' /tmp/eco-tui.log
```

Tests cover silence, increasing range/cost, inbox expiry, feature privacy,
arbitrary permutation, controlled learning, blocked channels, instinct isolation,
offspring reset, entity-order invariance, and deterministic V6 continuation, in
addition to the V5.1 perception, ecology, controller, temporal, and identity
coverage.
