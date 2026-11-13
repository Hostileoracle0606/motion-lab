import pytest


def test_cpu_is_explicit_and_other_devices_are_rejected():
    from learned_dynamics.devices import require_device
    assert require_device("cpu").type == "cpu"
    for name in ("auto", "cuda", "mps:1", None):
        with pytest.raises(ValueError):
            require_device(name)


def test_mps_unavailable_does_not_fallback(monkeypatch):
    import torch
    from learned_dynamics.devices import require_device
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="MPS"):
        require_device("mps")


def test_mps_cpu_fallback_setting_is_rejected(monkeypatch):
    from learned_dynamics.devices import require_device
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(ValueError, match="fallback"):
        require_device("mps")


def test_actual_mps_forward_backward_adam():
    import torch
    from learned_dynamics.devices import require_device
    from learned_dynamics.model import ResidualMLP
    from learned_dynamics.types import ModelSpec
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available in this runtime; no GPU result established")
    model = ResidualMLP(ModelSpec()).to(require_device("mps"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.ones((4, 3), device="mps")
    before = next(model.parameters()).detach().clone()
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    torch.mps.synchronize()
    assert torch.isfinite(loss).item()
    assert not torch.equal(before, next(model.parameters()).detach())
