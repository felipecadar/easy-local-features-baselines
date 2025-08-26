# How to add a new extractor

1. Create `src/easy_local_features/feature/baseline_<name>.py` with a subclass of `BaseExtractor`.
2. Implement `detectAndCompute`, `detect` (if detector), `compute`, `to`, and `has_detector`.
3. Add the name to `available_extractors` in `src/easy_local_features/__init__.py`.
4. Provide downloads or weight handling as needed.
5. Add tests under `tests/`.
