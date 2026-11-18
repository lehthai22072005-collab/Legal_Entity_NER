"""Advance the learning-rate schedule only after an actual optimizer update."""
import torch


def optimizer_step(model, optimizer, scaler, scheduler=None):
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    # GradScaler lowers its scale when a nonfinite gradient skips the update.
    # Equal or increased scale means optimizer.step was performed.
    updated = scaler.get_scale() >= before
    if updated and scheduler is not None:
        scheduler.step()
    return updated
