# Contribution Guidelines

## How to Contribute

We welcome contributions to the Table Comparison project! Here's how you can help:

### Types of Contributions

1. **Bug Reports** - Found a bug? Please report it!
2. **Feature Requests** - Have an idea for improvement?
3. **Code Contributions** - Submit pull requests for fixes or features
4. **Documentation** - Help improve our documentation
5. **Testing** - Add test cases or improve existing ones

### Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/table-comparison.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Set up development environment: `make dev-setup`

### Development Setup

```bash
# Install dependencies
make install-dev

# Check dependencies
make check-deps

# Run tests
make test

# Format code
make format

# Run linting
make lint
```

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and small
- Add type hints where appropriate

### Testing

- Write tests for new features
- Ensure all tests pass before submitting
- Aim for good test coverage
- Test with different data types and edge cases

### Submitting Changes

1. Make sure all tests pass: `make test`
2. Format your code: `make format`
3. Run linting: `make lint`
4. Commit with descriptive messages
5. Push to your fork
6. Create a pull request

### Pull Request Guidelines

- Describe what your PR does
- Reference any related issues
- Include tests for new functionality
- Update documentation if needed
- Keep PRs focused and atomic

### Reporting Issues

When reporting bugs, please include:
- Python version
- OS and version
- Ollama version (if applicable)
- Steps to reproduce
- Expected vs actual behavior
- Error messages or logs

### Feature Requests

For feature requests, please describe:
- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered
- How it fits with the project goals

## Development Guidelines

### Code Organization

```
src/
├── embeddings.py     # Embedding generation
├── comparisons.py    # Comparison methods
├── utils.py         # Utility functions
└── visualizations.py # Plotting and charts

tests/
├── test_embeddings.py
├── test_comparisons.py
└── test_utils.py
```

### Adding New Comparison Methods

1. Add method to `TableComparator` class
2. Follow existing patterns for similarity matrices
3. Include proper error handling
4. Add comprehensive tests
5. Update documentation

### Adding New Embedding Models

1. Update `EMBEDDING_MODELS` in `config.py`
2. Test with different model dimensions
3. Add fallback handling
4. Update documentation

### Performance Considerations

- Use numpy arrays for matrix operations
- Cache embeddings when possible
- Consider memory usage for large datasets
- Profile performance-critical code

## Questions?

- Open an issue for questions
- Check existing issues and documentation
- Contact maintainers for guidance

Thank you for contributing to the Table Comparison project!