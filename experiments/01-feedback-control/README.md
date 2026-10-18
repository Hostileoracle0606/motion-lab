# Experiment 01 — P vs PD Feedback Control

## Status

> **In progress — protocol documented; implementation and results not yet published.**

This directory currently contains the experiment specification only. There is no committed MuJoCo environment, controller source, trajectory dataset, or verified benchmark result yet. The planned values below are starting points, not measurements.

## Research question

How does adding derivative feedback change the motion of a simple MuJoCo-controlled joint when the environment, target, initial state, action limits, and evaluation protocol are held constant?

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

The initial implementation should use the smallest MuJoCo model that makes the comparison meaningful: a single actuated rotational joint or similarly simple one-degree-of-freedom plant.

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

## Planned setup

Exact values must be committed in configuration before execution. The initial protocol should include:

| Element | Planned choice |
| --- | --- |
| Plant | One actuated rotational joint |
| Simulator | MuJoCo, version pinned with the implementation |
| Target velocity | Zero |
| Initial states | Fixed, versioned set spanning both target directions |
| Targets | Fixed, versioned set within joint limits |
| Episode length | Fixed duration sufficient to observe settling or timeout |
| Control rate | Fixed and reported separately from simulator step |
| P sweep | Small declared grid of \(K_p\) values |
| PD sweep | Declared grid of \((K_p,K_d)\) values |
| Seeds | Declared list, even if the base environment is deterministic |
| Action limit | Identical for P and PD |

If noise, delay, or parameter perturbations are added, the noiseless nominal experiment remains the first baseline and each perturbation becomes a separate scenario group.

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

## Transition record

This experiment begins the data path toward learned dynamics. Each control step should eventually record:

```text
episode_id
step_index
time_s
state                 # at least q_t and qdot_t
target_state
controller_output     # before clipping
action                # applied after clipping
next_state
position_error
velocity_error
action_saturated
terminated
truncated
termination_reason
controller_type
controller_gains
environment_version
config_digest
seed
```

The observation/action order must be explicit: `state` is observed before `action` is applied, and `next_state` is observed after the configured simulator advance.

## Evaluation metrics

Let \(q_0\) be the initial position, \(q^*\) the target, and \(A=q^*-q_0\) the commanded change. Degenerate cases where \(|A|\) is near zero should be excluded by scenario construction or handled separately.

### Rise time

The elapsed time for the response to move from a declared lower fraction to a declared upper fraction of \(A\), commonly 10% to 90%. The chosen thresholds must be stored in configuration.

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

### Steady-state error

Use a declared tail window \(\mathcal{T}_{tail}\):

```math
e_{ss}=\frac{1}{|\mathcal{T}_{tail}|}
\sum_{t\in\mathcal{T}_{tail}} |q^*-q_t|
```

The window duration must be reported.

### Integrated absolute error

```math
IAE = \sum_t |q^*-q_t|\Delta t
```

This gives a single measure of error accumulated over the episode.

### Control effort

Use at least one declared action-based measure, such as:

```math
E_u = \sum_t (u_t^{applied})^2\Delta t
```

This is a control-effort proxy, not necessarily physical energy. Physical energy requires an appropriate torque-and-velocity calculation and matching actuator semantics.

### Saturation and failure

Report:

- fraction of control steps at an actuator limit;
- episodes that fail to settle before timeout;
- joint-limit or safety violations;
- numerical failures or non-finite values.

Failures remain in aggregate reporting unless a documented pre-existing exclusion rule applies.

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
- [ ] All experiment-defining values are in versioned configuration.
- [ ] Initial conditions, targets, and seeds are declared.
- [ ] Trajectory schema and validator are implemented.
- [ ] Unit tests cover controller equations, clipping, and metrics.
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
