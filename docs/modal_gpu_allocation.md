# Modal GPU Allocation Audit

This document captures how Signal provisions GPUs on Modal today and why we keep a
single Modal app rather than deploying separate apps per GPU tier.

## Runtime structure

- `modal_runtime/training_session.py` defines the stateful training container and
  now exports GPU-specific aliases using `TrainingSession.with_options(...)`. Each
  alias (for example `TrainingSession_A100_80GB_4`) pins a concrete `gpu` string so
  Modal launches the container on the correct hardware.
- `main.py` resolves the requested `gpu_config` to the matching Modal class name
  before calling `modal.Cls.from_name`, ensuring API requests land on the
  corresponding alias and therefore the right GPU.
- `api/gpu_allocator.py` performs the automatic sizing heuristic. When it selects
  a configuration such as `H100:8`, the API lookup immediately reuses the alias
  exported by the runtime module.

## Why we prefer aliases over multiple apps

Modal's `Cls.with_options(...)` clones a deployment with a different resource
configuration without duplicating the underlying code. By exporting one alias per
supported GPU, we keep deployment simple:

1. We only build and ship a single Docker image and Modal app (`signal`).
2. Variants share the same class implementation, so fixes ship once and apply to
   every GPU target.
3. Provisioning logic can remain declarative—selecting a GPU string is all that is
   needed to pivot to a new tier.

If we deployed distinct Modal apps per GPU, we would incur additional operational
work (multiple deploys, additional secrets wiring, more names to manage) without a
functional gain because `with_options` already provides dedicated instances under
unique names.

## Operational guidance

- To add a new GPU tier, append it to the `_GPU_VARIANTS` mapping in
  `modal_runtime/training_session.py` and extend the lookup table in `main.py`.
- Keep the allocator in `api/gpu_allocator.py` synchronized so automatic sizing can
  reach the newly supported variant.
- When rolling out allocator changes, redeploy the Modal app once; the aliases
  update automatically because they live in the same source module.
