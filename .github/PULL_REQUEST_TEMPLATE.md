## What this changes

<!-- One paragraph. What capability moves, and why. -->

## How it was validated

<!-- Which ground truth (analytic field from radar_palette.testing, or a
     held-out real volume), which baseline it was compared against, and the
     numbers. "Tests pass" is not validation for a new operator. -->

## Checklist

- [ ] `ruff check .` and `ruff format --check .` clean
- [ ] `pytest` passes
- [ ] New public names exported via the subpackage's `__all__`
- [ ] Docstrings are numpydoc and state units
- [ ] Optional dependencies guarded at the call site and gated in tests
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Commits carry `Co-authored-by` trailers for all contributors

Co-authored-by: Scott Collis <scollis.acrf@gmail.com>
Co-authored-by: Claude <noreply@anthropic.com>
