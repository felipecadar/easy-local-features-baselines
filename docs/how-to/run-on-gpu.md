# Run on GPU

Most extractors can be moved to GPU via `.to("cuda")`.

```python
ex = getExtractor("aliked")
ex.to("cuda")
```

If a detector or matcher doesn’t support `.to`, it will stay on CPU.
