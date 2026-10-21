# Motion Lab

Reproducible robotics control and dynamics experiments in MuJoCo.

Motion Lab currently implements a deterministic single-joint control system
with P and PD controllers, independent physics and control rates, actuator
limits, trajectory telemetry, automated tests, and synchronized visual replay.
The same experiment structure is intended to support trajectory collection,
dynamics models, and model-based control without replacing the foundations at
each stage.

![Synchronized side-by-side comparison of P and PD feedback controllers](assets/p_vs_pd_side_by_side.gif)

The orange arm uses proportional control. The blue arm adds velocity feedback.
Both runs use the same initial state, target, torque limit, model, and clock.
The GIF shows one deterministic scenario, not an aggregate evaluation.

## Quickstart

Motion Lab requires Python 3.12 and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/Hostileoracle0606/motion-lab.git
cd motion-lab
uv sync
uv run pytest -q
```

Run either controller without opening a viewer:

```bash
uv run python experiments/01-feedback-control/scripts/run_episode.py --controller p
uv run python experiments/01-feedback-control/scripts/run_episode.py --controller pd
```

On macOS, open the synchronized P-versus-PD comparison:

```bash
uv run mjpython experiments/01-feedback-control/scripts/run_visual_demo.py
```

Export the same comparison directly to a 1280 x 720 MP4:

```bash
uv run python experiments/01-feedback-control/scripts/record_visual_demo.py
```

The exporter writes
`experiments/01-feedback-control/results/p_vs_pd_side_by_side.mp4`.

## Current capabilities

| Area | Implementation |
| --- | --- |
| Plant | One actuated revolute joint modeled in MuJoCo XML |
| Control | Independently testable P and PD control laws with torque saturation |
| Execution | Deterministic observe-control-step episodes driven by TOML configuration |
| Timing | 500 Hz physics integration and 100 Hz control in the default configuration |
| Telemetry | Position, velocity, target, raw torque, applied torque, and saturation per control decision |
| Tuning | Declared P/PD gain sweep with a critical-damping estimate |
| Visualization | Single-controller replay and synchronized side-by-side comparison |
| Recording | Headless MP4 export with experiment overlays |
| Verification | Unit and integration coverage for configuration, controllers, model contracts, simulation, playback, recording, and tuning |

## System architecture

The controller is deliberately independent of MuJoCo. It receives measured
state and a target, then returns a bounded torque command. The simulation owns
the model state and advances physics. Playback consumes completed episode
records and never recomputes the control action.

```text
TOML configuration
       |
       v
MuJoCo state ---> controller ---> bounded torque
       ^                              |
       |                              v
       +------ physics substeps <-----+
                       |
                       v
              trajectory samples
                 /           \
                v             v
          headless output   visual replay / MP4
```

The default experiment advances physics every 2 ms and makes a new control
decision every 10 ms. The torque is therefore held constant for five physics
steps before the controller observes the next state. Keeping these clocks
separate allows the simulator to resolve motion more frequently than the
software updates its command.

## Experiment 01: Feedback control

The first system compares proportional and proportional-derivative feedback on
the same one-link plant.

For target position \(q^*\), measured position \(q_t\), and measured velocity
\(\dot q_t\):

```math
u_t^{P} = K_p(q^* - q_t)
```

```math
u_t^{PD} = K_p(q^* - q_t) - K_d\dot q_t
```

The proportional term commands torque from position error. The derivative term
opposes motion relative to a stationary target. Both requested torques are
clamped to the same actuator limit before they reach the MuJoCo model.

The experiment specification, model parameters, tuning assumptions, and
evaluation protocol are documented in
[Experiment 01](experiments/01-feedback-control/README.md).

### Configuration

The default scenario is defined in
[`configs/demo.toml`](experiments/01-feedback-control/configs/demo.toml). It
contains the initial and target positions, episode duration, physics and
control intervals, torque limit, and controller gains.

Changing one of these values produces a new declared scenario without changing
the controller or simulation source. Useful comparisons include:

- increasing or decreasing `controllers.p.kp`;
- varying `controllers.pd.kd` while keeping proportional gain fixed;
- reducing `episode.torque_limit_nm` to expose saturation;
- changing `episode.control_interval_s` while preserving an integer number of
  physics steps per control decision.

Inspect the declared gain sweep with:

```bash
uv run python experiments/01-feedback-control/scripts/sweep_gains.py
```

## Repository structure

```text
motion-lab/
├── assets/                         README media
├── docs/
│   ├── engineering-standards.md    evidence and evaluation requirements
│   ├── references.md               primary technical references
│   ├── research-foundations.md     control, dynamics, and planning context
│   └── roadmap.md                  staged system development
├── experiments/
│   └── 01-feedback-control/
│       ├── configs/                versioned scenarios and gain sweeps
│       ├── models/                 MuJoCo plants and comparison scene
│       ├── scripts/                executable experiment entry points
│       ├── src/feedback_control/   control, simulation, replay, and recording
│       └── tests/                  unit and integration checks
├── pyproject.toml                  package and test configuration
└── uv.lock                         locked dependency graph
```

The implementation is split by responsibility:

- `controllers.py` contains pure control laws and actuator limiting.
- `simulation.py` owns MuJoCo execution and trajectory sampling.
- `config.py` loads and validates the experiment contract.
- `playback.py` replays completed trajectories in the native viewer.
- `recording.py` renders the comparison to video.
- `tuning.py` defines gain-sweep diagnostics and damping estimates.

## Experiments

| Stage | System | Technical focus | Status |
| --- | --- | --- | --- |
| 01. Feedback control | One-link MuJoCo plant | P/PD control, saturation, multirate simulation, replay | Implemented; formal scenario evaluation in progress |
| 02. State-transition data | Versioned control trajectories | Schema, validation, replay, and dataset splits | Planned |
| 03. Forward dynamics | Analytical or simulator-backed predictor | One-step prediction and open-loop rollout error | Planned |
| 04. Learned dynamics | Model trained on recorded transitions | Held-out prediction and multi-step drift | Planned |
| 05. Model predictive control | Receding-horizon controller | Planning under action and compute constraints | Planned |
| 06. Visual and latent prediction | Observation-space prediction | Task-relevant representations for control | Planned |
| 07. Temporal hierarchy | Multi-rate goals and skills | Planning across control horizons | Planned |

See the [roadmap](docs/roadmap.md) for dependencies and completion gates.

## Engineering principles

- **Keep control laws simulator-independent.** Controller equations should be
  testable without launching MuJoCo.
- **Make timing explicit.** Physics integration, control decisions, sampling,
  and playback are separate concerns.
- **Put experiment-defining values in versioned configuration.** Results should
  identify the configuration that produced them.
- **Compare systems under shared constraints.** Initial state, target, model,
  actuator limits, duration, and metric definitions must remain controlled.
- **Preserve raw observations before interpretation.** Recorded state and action
  data should support later metrics, replay, and dynamics modeling.
- **Separate implementation from evidence.** A working demonstration is not a
  multi-scenario result, and planned work is not implemented work.

## Scope and limitations

- The current system is simulation-only and has not been validated on physical
  hardware.
- The included GIF represents one deterministic scenario.
- Persisted trajectory datasets and formal multi-scenario P/PD results have not
  been published.
- The current plant has one degree of freedom, no contacts, and gravity disabled
  to isolate the initial damping comparison.
- Timing output from an ordinary run is not a real-time performance benchmark.

## Documentation

- [Experiment 01](experiments/01-feedback-control/README.md) — setup, model,
  controller definitions, protocol, and commands.
- [Engineering standards](docs/engineering-standards.md) — evidence levels,
  testing, data contracts, evaluation, and reporting requirements.
- [Roadmap](docs/roadmap.md) — experiment dependencies and completion gates.
- [Technical foundations](docs/research-foundations.md) — control, dynamics,
  planning, representation learning, and temporal abstraction.
- [References](docs/references.md) — textbooks, papers, and official technical
  documentation.

## Contributing

Changes should preserve explicit interfaces between control, simulation,
recording, and evaluation. New experiments should include a versioned
configuration, executable entry point, relevant automated checks, and enough
information to reproduce any reported result. See the
[engineering standards](docs/engineering-standards.md) before publishing
measurements or comparisons.

## License

No license has been selected yet. Until one is added, the repository's contents
remain subject to the default protections of copyright law.
