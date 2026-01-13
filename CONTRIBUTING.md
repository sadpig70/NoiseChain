# Contributing to NoiseChain

Thank you for your interest in contributing to the NoiseChain project!

[🇰🇷 Korean Version (한국어)](CONTRIBUTING_ko.md)

## Development Setup

```bash
# Clone repository
git clone https://github.com/sadpig70/NoiseChain.git
cd NoiseChain

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Code Style

- **Formatter**: Ruff
- **Type Checker**: mypy
- **Line Length**: 100 characters

```bash
# Lint code
ruff check src/

# Check types
mypy src/noisechain/
```

## Commit Convention

```
<type>: <subject>

<body>
```

### Type

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `chore`: Build tasks, configuration changes

### Example

```
feat: Add correlation signature verification

- Implement CorrelationSignature.verify()
- Add cosine similarity threshold
- Update tests
```

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Commit your changes following the convention.
3. Verify that all tests pass (`pytest`).
4. Create a Pull Request (PR).

## Contact

- 📧 <sadpig70@gmail.com>
- 🔗 [GitHub Issues](https://github.com/sadpig70/NoiseChain/issues)
