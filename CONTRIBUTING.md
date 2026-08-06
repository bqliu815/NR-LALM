# Contributing

Please open an issue before changing a paper configuration or work
accounting rule. Bug fixes should include a focused regression test and must
preserve the paper-facing method names NR-LALM and NR-LALM+SOC.

For local checks:

```bash
python -m pip install -e ".[test]"
pytest
python scripts/validate_release.py
```

Do not commit downloaded data, raw result records, generated figures,
scheduler logs, credentials, or machine-specific paths.
