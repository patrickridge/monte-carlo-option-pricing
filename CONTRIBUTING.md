# Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

## Setup

```bash
git clone https://github.com/patrickridge/monte-carlo-option-pricing.git
cd monte-carlo-option-pricing
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest -q
```

All tests must pass before submitting a pull request.

## C++ Extension

The C++ backend is optional. See [README.md](README.md#c-acceleration-optional-macos) for build instructions.
