# Contributing to Motion Lab

Contributions should preserve clear boundaries between experiments and keep
reported behavior reproducible from committed code, configuration, and
artifacts.

Before publishing measurements or comparisons, read the
[engineering standards](docs/engineering-standards.md). They define the
repository's evidence levels, configuration requirements, testing expectations,
and reporting rules.

## Experiment contract

Each experiment should define:

- the system and assumptions under evaluation;
- versioned configuration and dependency requirements;
- executable entry points;
- state, action, timing, and unit conventions;
- automated unit and integration checks;
- evaluation criteria and failure conditions;
- commands sufficient to reproduce published artifacts;
- an explicit distinction between implementation, measurement, and
  interpretation.

Experiment-specific implementation details belong in the experiment directory.
An experiment may omit directories it does not require, but its public
interfaces, assumptions, commands, and evidence must remain documented.

## Engineering principles

- Keep control, simulation, data collection, evaluation, and presentation
  behind explicit interfaces.
- Make timing, units, coordinate frames, and action limits part of the system
  contract.
- Store experiment-defining values in versioned configuration.
- Compare systems under shared initial conditions and constraints.
- Preserve raw observations so metrics and visualizations can be regenerated.
- Test components independently before evaluating complete systems.
- Distinguish planned, implemented, reproduced, measured, and compared work.
- Publish limitations alongside results.

## Development workflow

1. Select an experiment from the [experiment index](README.md#experiments).
2. Read its local README and configuration.
3. Create or update the environment with `uv sync`.
4. Run its documented entry point.
5. Run the associated automated checks.
6. Change the configuration or implementation within that experiment.
7. Regenerate evidence using the documented evaluation path.

Run all repository tests from the repository root:

```bash
uv run pytest -q
```

Before submitting a change:

```bash
git diff --check
uv run pytest -q
```

New experiments should be self-contained, executable, and independently
verifiable. They should declare their interfaces and assumptions without
silently changing the contracts established by earlier experiments.
