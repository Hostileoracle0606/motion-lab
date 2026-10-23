# Experiment 01 — P vs PD Feedback Control

## Status

> **In progress — environment, model, controllers, closed-loop runner, trajectory schema and validator, pure controller metrics, declared gain sweep, versioned evaluation scenario suite, and real-time visual replay implemented; formal evaluated results not yet published.**

This directory contains the experiment specification, pinned Python environment,
initial one-link MuJoCo model, independently tested P and PD controller
equations, a deterministic closed-loop episode runner, versioned trajectory
persistence and validation, pure trajectory metrics, a theory-guided gain sweep,
and a versioned multi-scenario evaluation configuration. Evaluated result
artifacts have not been published. The declared protocol remains an input to
future evaluation, not a measurement.

## Research question

How does adding derivative feedback change the motion of a simple MuJoCo-controlled joint when the environment, target, initial state, action limits, and evaluation protocol are held constant?

## Environment setup

The experiment uses Python 3.12 and dependencies pinned in the repository-level
`pyproject.toml` and `uv.lock` files. From the repository root, create or update
the local environment with:

```bash
uv sync
```

Activate it when running commands directly:

```bash
source .venv/bin/activate
```

Alternatively, prefix a command with `uv run` to execute it inside the managed
environment without activating it first.

Run one deterministic episode with either controller:

```bash
uv run python experiments/01-feedback-control/scripts/run_episode.py --controller p
uv run python experiments/01-feedback-control/scripts/run_episode.py --controller pd
```

Inspect the declared P and PD gain sweep:

```bash
uv run python experiments/01-feedback-control/scripts/sweep_gains.py
```

Run the screen-recordable visual comparison on macOS:

```bash
uv run mjpython experiments/01-feedback-control/scripts/run_visual_demo.py
```

Export the same synchronized comparison directly to a 1280 x 720 MP4:

```bash
uv run python experiments/01-feedback-control/scripts/record_visual_demo.py
```

The generated video is written to
`experiments/01-feedback-control/results/p_vs_pd_side_by_side.mp4`. Direct
export records only the MuJoCo scene and experiment overlays, never the desktop.

The command computes both tested episodes before opening a clean MuJoCo window,
then replays P in orange and PD in blue side by side on the same clock. The red
guides mark their shared target. Viewer overlays identify each controller and
show time, position, error, velocity, applied torque, and actuator saturation.
By default, the shared starting frame is held for two seconds and the final
comparison for 1.5 seconds, which provides time to begin and end a screen
recording.

Useful presentation options include:

```bash
# Rehearse the comparison twice.
uv run mjpython experiments/01-feedback-control/scripts/run_visual_demo.py --repeat 2

# Inspect only one controller.
uv run mjpython experiments/01-feedback-control/scripts/run_visual_demo.py --controller pd
```

The viewer is a presentation layer, not a second simulation path. It copies
recorded states into an independent MuJoCo model and never advances physics or
recomputes controller actions. This keeps the displayed motion tied to the same
`EpisodeResult` records exercised by the headless tests.

The first experiment is deliberately narrow. Its purpose is to establish the repository's basic loop:

```text
define a system
  -> apply a controller
    -> record a trajectory
      -> compute reproducible metrics
        -> inspect success and failure cases
```

## Hypotheses

These are hypotheses to test, not reported findings:

1. With comparable proportional gain, a well-tuned PD controller will reduce oscillation and overshoot relative to a P controller.
2. Increasing derivative gain beyond a useful range may slow response or amplify discretization and measurement noise.
3. Better settling behavior may require different control effort, so no controller should be judged on settling time alone.

## System under study

The implementation uses the smallest MuJoCo model that makes the comparison
meaningful: a single actuated rotational joint with one degree of freedom.

The model specification must declare:

- generalized position and velocity conventions;
- joint range;
- mass, inertia, damping, and friction;
- actuator type, gear, and control range;
- gravity setting;
- solver and integration settings;
- simulation time step;
- controller update interval.

Starting simple makes controller behavior interpretable before moving to coupled joints, contacts, or image observations.

### Initial one-link model

The initial plant is defined in `models/one_link.xml`. It uses the following
declared values:

| Property | Value |
| --- | --- |
| Generalized position | Hinge angle in radians; positive rotation moves the arm toward positive x |
| Generalized velocity | Hinge angular velocity in radians per second |
| Joint axis and range | y axis, from -pi/2 to +pi/2 radians |
| Link | 0.8 m capsule with 1.0 kg mass |
| Joint dynamics | damping 0.05, friction loss 0, armature 0.01 |
| Actuator | Direct motor with gear 1 and control range [-3, 3] |
| Gravity | Disabled (`0 0 0`) for the initial damping comparison |
| Integration | RK4 with a 0.002 s simulation step |
| Solver | Newton |
| Demonstration poses | -pi/3 rad start keyframe and +pi/3 rad visible target |

The controller interval is not defined by the model. It is declared by the
experiment configuration and validated against the simulation time step before
an episode runs.

## Controllers

Let:

- \(q^*\) be the target position;
- \(q_t\) be position at control step \(t\);
- \(\dot q^*\) be target velocity, initially zero;
- \(\dot q_t\) be measured velocity;
- \(K_p\geq0\) be proportional gain;
- \(K_d\geq0\) be derivative gain;
- \(u_t\) be the commanded actuator input.

The position error is:

```math
e_t=q^*-q_t
```

### Proportional controller

```math
u_t^{P}=K_p e_t
```

### Proportional-derivative controller

```math
u_t^{PD}=K_p e_t+K_d(\dot q^*-\dot q_t)
```

For a stationary target:

```math
u_t^{PD}=K_p e_t-K_d\dot q_t
```

The command sent to MuJoCo must respect the same actuator limits for both controllers:

```math
u_t^{applied}=\operatorname{clip}(u_t,u_{min},u_{max})
```

The recorded trajectory should contain both the unclipped controller output and applied action so saturation is visible.

### Controller implementation

The pure controller functions live in `src/feedback_control/controllers.py`.
They are deliberately independent from MuJoCo: each function receives ordinary
numeric state and target values and returns both the requested torque and the
bounded torque sent to the actuator. This separation allows the equations and
clipping behavior to be tested without running the simulator.

The `ControllerOutput` record contains:

- `raw_torque`: the control law's requested torque before actuator limits;
- `applied_torque`: the torque after symmetric clipping;
- `saturated`: whether clipping changed the requested torque.

### Initial gain-selection procedure

The headline P and PD runs use the same proportional gain, `Kp = 2.0`, so the
presence of derivative feedback is the central changed variable. For the local
linear, unsaturated one-joint model, the approximate derivative gain for
critical damping is:

```math
K_d^{crit} = 2\sqrt{I K_p} - b
```

where `I` is the effective joint inertia reported by MuJoCo and `b` is the
model's passive joint damping. At the declared starting state this gives an
estimate near `1.3136`; the nearest value in the versioned sweep is `Kd = 1.3`.

This calculation assumes continuous time, locally constant inertia, zero
gravity, and no actuator saturation. The sweep therefore also reports final
error, final speed, overshoot, peak speed, saturated steps, and target crossings
for inspection. These are tuning diagnostics, not yet the experiment's formal
multi-scenario metrics.

## Versioned evaluation baseline

`configs/evaluation-v1.toml` is the initial formal evaluation input. Scenarios
live once at the suite level, so a future evaluator must give P and PD the same
start-target pairs rather than maintaining controller-specific copies.

| Element | Declared value |
| --- | --- |
| Schema | Version 1 |
| Plant | `models/one_link.xml` |
| Physics step | 0.002 s |
| Control interval | 0.010 s |
| Episode duration | 4.0 s |
| Target and initial velocity | 0 rad/s |
| Seeds | `[0]` |
| P gains | `Kp = 2.0` |
| PD gains | `Kp = 2.0`, `Kd = 1.3` |
| Applied-torque limit | 3.0 N m, symmetric |
| Rise thresholds | 10% to 90% of the commanded change |
| Settling band | 0.05 rad and 0.10 rad/s |
| Steady-state window | Final 0.50 s |
| Noise | Zero position, velocity, and action noise |

The four fixed scenarios are:

| Scenario | Start (rad) | Target (rad) | Direction |
| --- | ---: | ---: | --- |
| `negative-to-positive-wide` | -1.047198 | 1.047198 | Positive |
| `positive-to-negative-wide` | 1.047198 | -1.047198 | Negative |
| `negative-to-positive-narrow` | -0.523599 | 0.523599 | Positive |
| `positive-to-negative-narrow` | 0.523599 | -0.523599 | Negative |

This is sufficient for the first noiseless baseline because it exercises both
command directions at two magnitudes while keeping the plant, gains, timing,
limits, target velocity, and disturbances fixed. The symmetry makes a
direction-dependent implementation mistake visible without adding a large
scenario grid. It is not evidence about noisy sensing, delays, perturbed plant
parameters, other gains, or other robot models.

The public `load_evaluation_config` seam rejects unsupported schema versions,
missing or duplicate seeds, invalid timing, model-contract mismatches,
non-finite or out-of-range states, duplicate or degenerate scenarios,
one-direction-only suites, and inconsistent metric thresholds. Failure rules
declare non-finite state, joint-limit violation, and failure to settle by the
timeout as episode failures. Those failures must remain in aggregate reporting.

If noise, delay, or parameter perturbations are added, the noiseless nominal
experiment remains the first baseline and each perturbation becomes a separate
scenario group.

## Fair-comparison rules

P and PD evaluations must share:

- the same MuJoCo model and simulator settings;
- the same initial-state and target pairs;
- the same episode duration and sampling cadence;
- the same actuator limits;
- the same metric implementation;
- the same failure and exclusion rules;
- a documented, comparable gain-search budget.

Controller gains need not be numerically identical if each controller is tuned, but tuning procedures and selection criteria must be reported.

## Episode procedure

For each controller configuration and scenario:

1. Reset to the declared initial state.
2. Set the declared target and confirm it is within model limits.
3. At each control step, observe \(q_t\) and \(\dot q_t\).
4. Compute the controller output.
5. Clip it to the shared actuator range.
6. Advance the simulator for the declared number of substeps.
7. Record the transition and controller diagnostics.
8. Stop at the time limit or a declared failure condition.
9. Compute episode metrics from the saved record.

Metrics should be computed after collection from the recorded trajectory, rather than only as mutable state inside the control loop.

### Implemented closed-loop runner

The initial runner uses a 0.002 s MuJoCo simulation step and a 0.010 s control
interval, so each selected action is held for five physics steps. A four-second
episode therefore contains 400 controller observations and 2,000 physics steps.
The control loop records samples in memory. `trajectory.py` losslessly converts
those samples and the final simulator state into durable transitions.

## Transition record

`src/feedback_control/trajectory.py` defines trajectory schema version 1. One
UTF-8 JSON file contains one complete episode. `save_trajectory` creates and
validates a record; `load_trajectory` reconstructs the typed record and runs the
same validation again. Generated files belong under `results/trajectories/`,
which is ignored by Git unless a small artifact is deliberately selected as
published evidence.

Episode-level metadata has this contract:

| Field | Type / shape | Meaning |
| --- | --- | --- |
| `schema_version` | integer scalar | Exact on-disk contract version; currently `1` |
| `episode_id` | non-empty string | Stable identity chosen by the collection caller |
| `seed` | non-negative integer | Declared collection seed, including `0` for the deterministic runner |
| `environment_id` | string | `motion-lab/experiment-01/one-link` |
| `environment_version` | 64-character SHA-256 | Digest of the MuJoCo XML bytes |
| `mujoco_version` | string | Installed MuJoCo package version |
| `config_digest` | 64-character SHA-256 | Digest of the canonical embedded configuration |
| `configuration` | object | Model identity, timing, state/target, limit, and both controllers' gains |
| `controller` | object | Selected controller name and its `kp`/`kd` gains |
| `transitions` | ordered array | One object for every control interval |

Each transition has this contract:

| Field | Type / shape | Unit and meaning |
| --- | --- | --- |
| `step_index` | integer scalar | Zero-based, contiguous control-step index |
| `time_s` | finite number | Seconds from reset; exactly `step_index * control_interval_s` |
| `state` | two-scalar object | `position_rad` in rad and `velocity_rad_s` in rad/s at time \(t\) |
| `target` | two-scalar object | Target position in rad and target velocity in rad/s |
| `raw_torque_nm` | finite number | Controller request before clipping, in N m |
| `applied_torque_nm` | finite number | Action held for the interval after clipping, in N m |
| `saturated` | Boolean | Whether clipping changed the controller request |
| `next_state` | two-scalar object | Position and velocity after the configured simulator advance |
| `terminated` | Boolean | A declared task/failure terminal condition occurred |
| `truncated` | Boolean | An external limit ended the episode |
| `termination_reason` | string or `null` | Boundary reason; `null` before the final transition |

The temporal contract is `(state_t, applied_torque_t) -> next_state_t`:
`state` is observed before the action is applied, and `next_state` is observed
after the configured simulator substeps. Consequently,
`next_state[t] == state[t + 1]`. The final transition uses the runner's final
state instead of inventing an extra control interval. The current runner always
reaches its fixed time limit, so only the last transition has
`truncated=true` and `termination_reason="time_limit"`.

Validation rejects incompatible or incomplete schemas; non-finite numbers;
invalid shapes, types, identities, timing, and terminal boundaries;
discontinuous state chains; configuration/controller inconsistencies; and
applied actions outside or inconsistent with the configured torque limit.

Programmatic use is deliberately separate from simulation:

```python
from pathlib import Path

from feedback_control.trajectory import load_trajectory, save_trajectory

trajectory_path = Path(
    "experiments/01-feedback-control/results/trajectories/p-seed-0.json"
)
saved = save_trajectory(
    trajectory_path,
    episode_id="p-seed-0",
    seed=0,
    result=result,
    config=config,
)
loaded = load_trajectory(trajectory_path)
assert loaded == saved
```

Here `result` is an existing `EpisodeResult` and `config` is the exact
`ExperimentConfig` used to produce it. `save_trajectory` returns the validated
record; the caller retains the output path separately.

## Evaluation metrics

Let \(q_0\) be the initial position, \(q^*\) the target, and \(A=q^*-q_0\) the commanded change. Degenerate cases where \(|A|\) is near zero should be excluded by scenario construction or handled separately.

### Rise time

The elapsed time for the response to move from a declared lower fraction to a declared upper fraction of \(A\), commonly 10% to 90%. The chosen thresholds must be stored in configuration.

The implemented `rise_time` metric normalizes progress by the signed command,
so increasing and decreasing targets use the same thresholds. Positions are
uniform samples beginning at time zero; threshold times use the first sampled
crossings without interpolation. The result is in seconds. A missing lower or
upper crossing is returned as an explicit timing outcome with no finite time.

### Maximum overshoot

For a step in either direction, measure progress along the signed command direction:

```math
r_t = \frac{(q_t-q^*)\operatorname{sign}(A)}{|A|}
```

Then normalized overshoot is:

```math
M_p = \max(0, \max_t r_t)
```

Report the value as a fraction or percentage and state which.

The implemented `normalized_maximum_overshoot` metric reports a fraction of
the commanded displacement. It is direction-aware and rejects a zero-amplitude
command rather than dividing by an undefined scale.

### Settling time

The earliest time \(t_s\) after which absolute error remains inside a configured band for the rest of the evaluation window:

```math
|q_t-q^*| \leq \epsilon_q \quad \text{for all } t\geq t_s
```

Optionally require a velocity tolerance as well:

```math
|\dot q_t|\leq\epsilon_v
```

Episodes that never meet the criterion should be reported as not settled, not assigned a misleading finite time.

The implemented `settling_time` metric uses uniformly spaced samples beginning
at time zero and returns the earliest sampled time whose remaining position
samples all satisfy the configured tolerance. Velocity samples and a velocity
tolerance may be supplied together to require both bands. A trajectory whose
final observed samples do not satisfy the declared bands returns the explicit
`not_settled` outcome and no finite time.

### Steady-state error

Use a declared tail window \(\mathcal{T}_{tail}\):

```math
e_{ss}=\frac{1}{|\mathcal{T}_{tail}|}
\sum_{t\in\mathcal{T}_{tail}} |q^*-q_t|
```

The window duration must be reported.

The implemented `steady_state_absolute_error` metric reports radians and uses
the final action-bearing samples whose duration equals the declared tail
window. The window must be a positive integer multiple of the control interval
and cannot exceed the sampled episode duration.

### Integrated absolute error

```math
IAE = \sum_t |q^*-q_t|\Delta t
```

This gives a single measure of error accumulated over the episode.

The implemented `integrated_absolute_error` metric consumes one pre-action
position sample for every action-bearing control interval. The terminal state
is not counted as an additional interval. It validates that the target,
positions, and control interval are finite, requires at least one sample, and
reports the result in radian-seconds.

The episode-level `evaluate_episode` boundary applies that pure metric to the
pre-action samples in one `EpisodeResult`. It requires a non-empty,
static-target episode and verifies that the declared control interval matches
every recorded interval, including the final sample-to-terminal-state advance.
The terminal state checks the final interval duration but does not contribute a
second position-error sample.

### Control effort

Use at least one declared action-based measure, such as:

```math
E_u = \sum_t (u_t^{applied})^2\Delta t
```

This is a control-effort proxy, not necessarily physical energy. Physical energy requires an appropriate torque-and-velocity calculation and matching actuator semantics.

The implemented `squared_control_effort` metric consumes one applied torque
for every full control interval and reports
\((\mathrm{N\,m})^2\,\mathrm{s}\). It deliberately uses applied rather than
requested torque and remains a proxy, not a claim about mechanical or
electrical energy.

### Saturation and failure

Report:

- fraction of control steps at an actuator limit;
- episodes that fail to settle before timeout;
- joint-limit or safety violations;
- numerical failures or non-finite values.

Failures remain in aggregate reporting unless a documented pre-existing exclusion rule applies.

The implemented `saturation_fraction` metric reports the fraction of applied
torques exactly at either configured symmetric limit. It rejects samples
outside that limit as invalid data. All metric functions reject empty or
non-finite numeric inputs instead of producing aggregate values from corrupted
trajectories.

## Analysis and plots

Planned outputs for each comparison:

- position and target versus time;
- velocity versus time;
- controller output and applied action versus time;
- error versus time;
- metric summary across the full scenario set;
- gain-sweep heatmaps or response maps where useful;
- a table of failures and saturation rates.

Plots should include units, controller configuration, scenario identity, and a link or filename for the machine-readable source data.

## Reproducibility checklist

Experiment 01 cannot be marked complete until:

- [ ] MuJoCo model and controller code are committed.
- [ ] Runtime and dependencies are pinned.
- [x] All experiment-defining values are in versioned configuration.
- [x] Initial conditions, targets, and seeds are declared.
- [x] Trajectory schema and validator are implemented.
- [x] Unit tests cover controller equations, clipping, and metrics.
- [ ] A deterministic smoke run passes from a clean checkout.
- [ ] The full evaluation is generated by a documented command.
- [ ] Raw or losslessly derived metric records are retained.
- [ ] Reported tables and plots can be regenerated.
- [ ] Results include variability and all declared failures.

## Expected repository additions

Implementation is expected to add a structure similar to:

```text
experiments/01-feedback-control/
├── README.md
├── configs/
├── models/
├── scripts/
├── src/
├── tests/
└── results/
```

This is a proposed organization, not a statement that these files already exist.

## Link to the next stage

The most important output is not a winning controller. It is a trustworthy trajectory pipeline. Once transitions can be recorded, validated, split, and replayed, the same `(state, action, next_state)` records can support the forward-dynamics and learned-dynamics stages in the [roadmap](../../docs/roadmap.md).
