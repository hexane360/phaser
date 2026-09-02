# Changelog

## 0.3.0 — 2026-09-02

Changes since [v0.2](https://github.com/hexane360/phaser/tree/v0.2) (2025-09-05).

### Major features

- **PyTorch backend** (#31, #45). A fourth array backend alongside numpy, cupy and jax,
  with autodiff support for the gradient engine and Apple MPS device support.
- **Arbitrary MTF support** (#58). Real- and reciprocal-space convolution machinery
  (`prepare_convolve2d`/`prepare_convolve2d_recip`, DCT-based `'reflect'` boundaries,
  separable-filter fast paths), a `Filter` algebra (products, sums, scalar multiplication),
  EMPAD preset MTFs, an `apply_mtf` preprocessing hook, and MTF in the forward model of
  both the gradient and conventional engines.
- **Web UI overhaul** (#54). Topic pub/sub websocket protocol (`web/pubsub.py`,
  `web/views.py`), tabbed/dockable dashboard with drag & drop and persisted layouts,
  migration from `wasm-array` to `@hexane360/plotlib`, new widgets (object amplitude &
  phase, sum & stack; probe modes in real and reciprocal space; scan positions),
  heartbeat/reconnect handling, version information, and an error page.
- **New CLI structure** (#46). `phaser/cli/` with lazily-loaded subcommands and declared
  optional dependencies, plus two interactive tools: `process_empad` (EMPAD XML metadata
  fixup) and `calc_drift` (linear scan drift measurement), backed by a persisted YAML
  `Config` (`utils/config.py`).
- **Arbitrary aberration support for the initial probe** (#43), validated against Kirkland.
- **New data loaders**: Gatan/DM4 (#32, requires `rsciio`) and Nion, including scan rotation
  extraction from Gatan metadata.
- **HDF5 state format v0.2** (#55). Reads remain backwards
  compatible with v0.1.
- **Granular loss reporting** (#35). `progress` is now one series per loss term (detector
  error plus each named regularizer), threaded through both engines, the observer and the
  web UI.
- **Synthetic/constrained position gradients**: `positions_affine` and `positions_line` allow constraining position updates.

### Major bugfixes

- Websocket compression disabled via a monkeypatch on the pinned `hypercorn`, fixing server
  hangs and long response times (#54).
- Fixed slow compilation on the conventional engines, plus general conventional-engine fixes
  and gradient-engine speedups.
- Fixed underflow NaNs on torch/MPS, the adam solver's `nu` dtype, and `remove_linear_ramp`
  on torch (#45).
- Fixed `torch.func.grad` with non-tensor aux types, and `recip_grid` breaking tracing on torch.
- Compact grouping algorithm rewritten: recursive bisection seeding instead of kmeans, with
  vectorized refinement. Substantially better with a large number of groups.
- Exceptions are now passed to `Observer.close()`, and job failures propagate to the web UI.
- Scan affine transform is applied pre-rotation.
- Position updates are no longer scaled by grouping.
- Fixed `drop_nan_patterns` with tilt (#32).

### Minor features

- `nonneg_object_phase` constraint, with an example plan (#42).
- `clamp_object_amplitude` accepts an optional lower bound (#38).
- Observer reporting for position & tilt updates, elapsed time and UTC time, and a detailed
  error breakdown.
- Py4DSTEM layout added to the default HDF5 search path.
- `offset` post-load hook.
- Better `convolve1d` on jax and torch, and better `xp.pad` support on torch (#52).
- Warning when the ground truth is too small for reliable cross-correlation in
  `align_object_to_ground_truth` (#41).

### Packaging

- Version is now dynamic, read from `phaser.__version__`; the entry point moved to
  `phaser.cli:cli`.
- Dropped support for numpy 1.x (now `numpy>=2.0,<2.7`).
- New `torch` and `cupy13` extras; jax widened to `<0.11`; `py-pane` bumped to 0.11.7.
- New dependencies: `frozendict`, `optree`, `platformdirs`, `lxml`.
- `hypercorn` pinned exactly in `web` feature.