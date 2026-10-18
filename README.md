# Motion Lab

**Learning how machines move, predict, and act.**

Motion Lab is a documentation-first robotics learning project. It traces a practical path from feedback control in simulation to systems that learn dynamics, predict future states, and plan across multiple time scales.

The project is intentionally incremental:

```text
Feedback control
    -> recorded state transitions
    -> analytical forward dynamics
    -> learned dynamics
    -> model predictive control
    -> visual and latent prediction
    -> temporal and hierarchical planning
```

## Status

> **In progress — documentation foundation only.**

The repository currently defines the research direction, engineering standards, and the specification for the first experiment. Simulator source code, recorded trajectories, trained models, and verified benchmark results have **not** been published yet. Any future result must be reproducible from committed code and artifacts before it is reported as complete.

## Why this project exists

A robot does not become intelligent merely by producing motion. It must connect three capabilities:

1. **Move:** convert a desired state into an action through control.
2. **Predict:** estimate how the world will change after an action.
3. **Act:** choose actions whose predicted outcomes serve a goal.

Motion Lab develops those capabilities in that order. Each experiment should produce a concrete artifact needed by the next one: a controller produces trajectories, trajectories support a dynamics model, a dynamics model supports planning, and increasingly abstract predictions support longer-horizon decisions.

## Learning path

| Stage | Central question | Intended artifact | Status |
| --- | --- | --- | --- |
| 1. Feedback control | How do P and PD controllers change motion? | MuJoCo experiment and controller comparison | In progress |
| 2. State-transition data | What experience must be recorded for prediction? | Versioned `(state, action, next_state)` trajectories | Planned |
| 3. Forward dynamics | Can known physics predict the next state? | Analytical or simulator-backed prediction baseline | Planned |
| 4. Learned dynamics | Can a model learn transitions from experience? | Trained one-step and multi-step predictors | Planned |
| 5. Model predictive control | Can predictions select better actions? | Receding-horizon controller | Planned |
| 6. Visual and latent prediction | Can useful futures be predicted without reconstructing every pixel? | JEPA-inspired representation and predictor experiments | Planned |
| 7. Temporal hierarchy | Can planning operate at several time scales? | Short-horizon skills and higher-level goal planner | Planned |

See the [project roadmap](docs/roadmap.md) for stage gates and dependencies.

## Experiment 01: P vs PD feedback control

The first experiment studies a simple but important distinction:

- A **proportional (P) controller** reacts to position error.
- A **proportional-derivative (PD) controller** reacts to position error and velocity.

For a target position \(q^*\), position \(q_t\), and velocity \(\dot q_t\):

```math
e_t = q^* - q_t
```

```math
u_t^{P} = K_p e_t
```

```math
u_t^{PD} = K_p e_t - K_d \dot q_t
```

The planned MuJoCo experiment will compare the controllers under the same model, initial conditions, targets, simulation step, actuator limits, and evaluation procedure. Planned metrics include rise time, settling time, overshoot, steady-state error, control effort, and failure rate across a deterministic scenario set.

The complete experiment specification is in [experiments/01-feedback-control/README.md](experiments/01-feedback-control/README.md).

## Repository map

```text
motion-lab/
├── README.md
├── docs/
│   ├── roadmap.md
│   ├── research-foundations.md
│   ├── engineering-standards.md
│   └── references.md
└── experiments/
    └── 01-feedback-control/
        └── README.md
```

As implementation is added, experiments should remain self-contained and link to shared packages rather than duplicate core logic.

## Documentation

- [Roadmap](docs/roadmap.md) — experiment sequence, dependencies, and completion gates.
- [Research foundations](docs/research-foundations.md) — control, dynamics, planning, representation learning, and temporal abstraction.
- [Engineering standards](docs/engineering-standards.md) — reproducibility, data contracts, tests, evaluation, and reporting rules.
- [References](docs/references.md) — primary papers, textbooks, and official technical documentation.
- [Experiment 01](experiments/01-feedback-control/README.md) — planned setup and P-versus-PD evaluation protocol.

## Project principles

- **Establish a baseline before adding learning.** A learned model should be compared with a simple, understood alternative.
- **Change one important variable at a time.** Comparisons are meaningful only when the environment and evaluation conditions are controlled.
- **Treat data as an interface.** States, actions, timing, units, terminal conditions, and metadata must be explicit.
- **Separate prediction from control.** Model accuracy and task performance answer different questions and should be measured independently.
- **Report distributions, not a best run.** Seeds, scenario coverage, failure cases, and variability belong beside aggregate metrics.
- **Distinguish fact from intent.** Planned experiments are not implemented experiments, and generated numbers are not verified results.

## Contributing

Contributions should preserve the staged learning path and follow the [engineering standards](docs/engineering-standards.md). A change that reports experimental results should include the exact configuration, seed policy, raw or derivable metrics, and instructions sufficient to reproduce the result.

## License

No license has been selected yet. Until one is added, the repository's contents remain subject to the default protections of copyright law.
