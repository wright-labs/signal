# Publishing the Frontier Signal SDK to PyPI

## Prerequisites

1. Install build tools:

```bash
pip install build twine
```

2. Create accounts:
   - PyPI: https://pypi.org/account/register/
   - TestPyPI (for testing): https://test.pypi.org/account/register/

3. Configure API tokens:
   - Create API tokens on PyPI/TestPyPI
   - Add to `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-...

[testpypi]
username = __token__
password = pypi-...
```

## Build the Package

From the `client/` directory:

```bash
cd /path/to/frontier-signal/client

# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build the package
python -m build
```

This creates:
- `dist/frontier_signal-0.1.0.tar.gz` (source distribution)
- `dist/frontier_signal-0.1.0-py3-none-any.whl` (wheel)

## Test Locally

Install the package locally to test:

```bash
# Install in editable mode
pip install -e .

# Or install from wheel
pip install dist/frontier_signal-0.1.0-py3-none-any.whl
```

Test the installation:

```python
import frontier_signal
from frontier_signal import SignalClient

print(frontier_signal.__version__)  # Should print: 0.1.0
```

## Publish to TestPyPI (Recommended First)

```bash
python -m twine upload --repository testpypi dist/*
```

Test installation from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ frontier-signal
```

## Publish to PyPI (Production)

Once you've tested on TestPyPI:

```bash
python -m twine upload dist/*
```

Users can then install with:

```bash
pip install frontier-signal
```

## Version Updates

When releasing a new version:

1. Update version in `frontier_signal/_version.py`:
```python
__version__ = "0.2.0"
```

2. Update version in `pyproject.toml`:
```toml
version = "0.2.0"
```

3. Tag the release in git:
```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

4. Build and publish as described above

## Automated Publishing with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      
      - name: Build package
        working-directory: ./client
        run: python -m build
      
      - name: Publish to PyPI
        working-directory: ./client
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: python -m twine upload dist/*
```

Add your PyPI API token as a GitHub secret named `PYPI_API_TOKEN`.

## Verification

After publishing, verify the package:

1. Check the PyPI page: https://pypi.org/project/frontier-signal/
2. Install in a fresh environment:
```bash
python -m venv test_env
source test_env/bin/activate
pip install frontier-signal
python -c "import frontier_signal; print(frontier_signal.__version__)"
```

## Troubleshooting

### Package name already exists
If `frontier-signal` is taken, update the name in `pyproject.toml`:
```toml
name = "frontier-signal-ml"  # or another available name
```

### Import name
The package name on PyPI (`frontier-signal`) is different from the import name (`frontier_signal`). This is standard practice for packages with hyphens:
- Install: `pip install frontier-signal`
- Import: `import frontier_signal` or `from frontier_signal import SignalClient`

### Missing files in distribution
Ensure `MANIFEST.in` includes all necessary files. Check what's included:
```bash
tar -tzf dist/frontier_signal-0.1.0.tar.gz
```

## Yanking a Release

If you publish a broken version:

```bash
# Yank the version (makes it unavailable for new installs)
# This requires manual action on PyPI website
```

Then publish a fixed version with an incremented version number.
