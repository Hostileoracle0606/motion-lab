# Research Foundations

This document introduces the ideas that connect Motion Lab's experiments. It is a conceptual guide, not evidence that any described system has been implemented in this repository.

## 1. Feedback control

### Open loop and closed loop

An open-loop system selects an action without measuring the result. A closed-loop system repeatedly observes the system, compares it with a target, and adjusts the action.

Let the desired joint position be \(q^*\), the measured position be \(q_t\), and the error be:

```math
e_t = q^* - q_t
```

A proportional controller applies:

```math
u_t = K_p e_t
```

Increasing \(K_p\) often produces a faster response, but it can also increase overshoot, oscillation, actuator saturation, or sensitivity to delay. The outcome depends on the plant, time step, and limits; gain alone does not guarantee a particular behavior.

A proportional-derivative controller adds a velocity-dependent term:

```math
u_t = K_p(q^* - q_t) + K_d(\dot{q}^* - \dot{q}_t)
```

For a stationary target, \(\dot{q}^*=0\), giving:

```math
u_t = K_p(q^* - q_t) - K_d\dot{q}_t
```

The derivative term can add damping, but finite differences and noisy velocity estimates can make it unstable or erratic if they are not handled carefully.

### What controller metrics mean

- **Rise time:** time to first enter a declared fraction of the requested change.
- **Overshoot:** amount the response exceeds the target, usually normalized by the commanded change.
- **Settling time:** first time after which error stays within a declared tolerance band.
- **Steady-state error:** residual error over a declared tail window.
- **Control effort:** a declared summary such as \(\sum_t u_t^2\Delta t\) or \(\sum_t |u_t|\Delta t\).
- **Failure rate:** fraction of episodes that violate a declared condition, such as a time limit or safety bound.

Every metric needs an operational definition. “Stable,” “smooth,” and “fast” are not measurements by themselves.

## 2. States, actions, observations, and time

These terms must remain distinct:

- **State \(s_t\):** information sufficient, under the model assumptions, to predict how the system evolves. A simulator state might include positions and velocities.
- **Action \(a_t\):** command supplied to the system, such as torque or a normalized actuator request.
- **Observation \(o_t\):** information available to the controller or learner. It may be the full simulator state, a noisy subset, or an image.
- **Transition:** one time-indexed record connecting current information, an action, and the resulting information.

A common Markov transition is:

```math
s_{t+1} \sim p(s_{t+1}\mid s_t, a_t)
```

A deterministic approximation writes:

```math
s_{t+1} = f(s_t, a_t)
```

The sampling interval \(\Delta t\) is part of this function. A model trained at one interval is not automatically valid at another.

## 3. Trajectories as a data product

An episode is a time-ordered trajectory:

```math
\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)
```

At minimum, a useful transition record should make the following reconstructable:

- episode and step identity;
- current state, action, and next state;
- timestamps or the exact fixed time step;
- termination and truncation flags;
- coordinate frames and physical units;
- environment, model, and controller versions;
- initial condition, target, and random seed.

Splitting adjacent rows randomly across training and test data leaks nearly identical temporal context. Motion Lab will prefer episode-level or scenario-level partitions.

## 4. Forward dynamics

A forward-dynamics model predicts an effect from a current condition and an action:

```math
\hat{s}_{t+1} = f_\theta(s_t, a_t)
```

The parameter \(\theta\) may represent known physical parameters, a numerical simulator, learned weights, or a combination.

### One-step and multi-step prediction

One-step evaluation supplies the true state at every step. Multi-step open-loop evaluation feeds predictions back into the model:

```math
\hat{s}_{t+k+1} = f_\theta(\hat{s}_{t+k}, a_{t+k})
```

Small one-step errors can compound. Therefore, strong one-step accuracy does not by itself establish reliable planning over long horizons.

### Useful baselines

Before a neural model, consider:

- no-change prediction, \(\hat{s}_{t+1}=s_t\);
- constant-velocity prediction;
- linear regression or local linear dynamics;
- analytical integration where equations are known;
- simulator-backed prediction.

Baselines reveal whether model complexity produces meaningful improvement.

## 5. Learned dynamics

A basic learned model can minimize a transition loss:

```math
\mathcal{L}_{1} = \frac{1}{N}\sum_{i=1}^{N}
\left\|f_\theta(s_i,a_i)-s'_i\right\|_W^2
```

The weighting matrix \(W\) matters because state components can have different units and scales. Angular coordinates may require wrapped differences, and quaternion errors require geometry-aware treatment.

Common extensions include:

- predicting the state change \(\Delta s_t\) rather than the absolute next state;
- conditioning on a history when observations are partially observable;
- predicting a probability distribution for stochastic or ambiguous outcomes;
- optimizing multi-step losses to expose compounding error during training;
- combining known physics with a learned residual.

The model should be evaluated both where the data are dense and where the controller visits unfamiliar states.

## 6. Model predictive control

Model predictive control (MPC) repeatedly solves a finite-horizon planning problem. At time \(t\):

1. Observe the current state.
2. Propose candidate action sequences.
3. Predict each sequence's consequences with the dynamics model.
4. Score predicted trajectories with a cost function.
5. Execute only the first selected action.
6. Observe again and replan.

For horizon \(H\):

```math
J = \sum_{k=0}^{H-1} c(\hat{s}_{t+k},a_{t+k})
+ c_T(\hat{s}_{t+H})
```

Replanning can correct some model error, but it does not make model quality irrelevant. A planner can exploit inaccuracies by selecting actions that look good only inside the model. Closed-loop evaluation in the actual simulator is therefore essential.

## 7. From state prediction to visual prediction

When observations are images, predicting every future pixel may spend capacity on detail that does not affect action: texture, lighting noise, or inherently uncertain background motion.

An encoder can map an observation into a representation:

```math
z_t = E(o_t)
```

A predictor then estimates a future representation, possibly conditioned on actions and temporal context:

```math
\hat{z}_{t+1} = P(z_t,a_t)
```

The hope is that \(z\) preserves predictable, task-relevant structure. That hope must be tested through downstream control, probing, transfer, and collapse diagnostics; a low latent loss alone is insufficient.

## 8. JEPA-inspired objectives

Joint-Embedding Predictive Architecture (JEPA) work motivates prediction in representation space rather than requiring direct reconstruction of the input. Motion Lab uses **JEPA-inspired** as a research direction, not as a claim of equivalence to I-JEPA, V-JEPA, or any other published architecture.

A generic objective has a context encoder, target encoder, and predictor:

```math
z_c = E_c(x_c), \qquad z_t = E_t(x_t), \qquad \hat{z}_t=P(z_c, m)
```

```math
\mathcal{L}_{JEPA-like} = d(\hat{z}_t, \operatorname{sg}(z_t))
```

Here \(m\) can describe a masked region, future time, action sequence, or other prediction condition. A practical design must still answer:

- What prevents representation collapse?
- How is the target encoder updated?
- What spatial or temporal information is hidden from the context?
- How are actions represented?
- Which downstream task tests whether the representation is useful?

## 9. Temporal abstraction and hierarchical planning

Long tasks are difficult to plan one simulator step at a time. Hierarchical systems introduce abstractions that last for multiple steps, such as a skill, option, subgoal, or latent action.

A two-level system can be viewed as:

```math
g_k \sim \pi_{high}(g_k \mid z_k), \qquad
a_t \sim \pi_{low}(a_t \mid s_t,g_k)
```

The high-level policy selects a goal \(g_k\) at a slower rate; the low-level policy acts until the goal is reached, times out, or terminates. This creates new research problems: defining reusable skills, predicting their outcomes, handling variable duration, and recovering when a subgoal fails.

## 10. Connecting the stages

The progression is cumulative:

```text
P/PD control
  produces trajectories
    that define dynamics data
      that train and test predictors
        that enable short-horizon planning
          that can move into learned representations
            that support longer temporal abstractions
```

Skipping an interface usually hides a problem rather than removing it. The project's central discipline is to make each interface explicit and test it before adding the next layer.
