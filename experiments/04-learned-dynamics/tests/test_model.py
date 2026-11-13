import pytest


def test_model_shape_seed_and_float32_contract():
    import torch
    from learned_dynamics.model import ResidualMLP
    from learned_dynamics.types import ModelSpec
    first, second = ResidualMLP(ModelSpec(), seed=3), ResidualMLP(ModelSpec(), seed=3)
    assert first(torch.ones((2, 3), dtype=torch.float32)).shape == (2, 2)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters(), strict=True))
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in first.parameters())
    assert not torch.equal(next(first.parameters()), next(ResidualMLP(ModelSpec(), seed=4).parameters()))


@pytest.mark.parametrize("width,history", [(True, 0), (0, 0), (32, True), (32, 2)])
def test_model_spec_rejects_invalid_architecture(width, history):
    from learned_dynamics.types import ModelSpec
    with pytest.raises(ValueError):
        ModelSpec(width, history)


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "empty"])
def test_model_rejects_invalid_input(kind):
    import torch
    from learned_dynamics.model import ResidualMLP
    from learned_dynamics.types import ModelSpec
    values = {"shape": torch.ones(3), "dtype": torch.ones((1, 3), dtype=torch.float64),
              "nan": torch.full((1, 3), float("nan")), "empty": torch.ones((0, 3))}
    with pytest.raises(ValueError):
        ResidualMLP(ModelSpec())(values[kind])
