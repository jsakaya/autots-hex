# AutoTS Hexagonal Redesign

This repository hosts the hexagonal (ports-and-adapters) reimplementation of AutoTS using 
[Darts](https://github.com/unit8co/darts) as the unified forecasting backend. Development follows 
the roadmap captured in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

## Development Setup
1. Ensure Python 3.10 or newer is available.
2. Create a virtual environment and install package dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```
3. Run the test suite:
   ```bash
   pytest
   ```

## Project Structure
```
core/
  domain/          # Entities, value objects, domain services
  application/     # Application services and orchestrators
ports/             # Abstract interfaces for adapters
adapters/
  primary/         # CLI, API, or UI adapters
  secondary/       # Darts, persistence, logging, execution adapters
config/            # Dependency wiring and configuration profiles
tests/             # Unit, integration, and regression tests
```

## Contributing
Feature work is delivered in small, fully tested increments. See the plan for the current phase 
and the acceptance criteria for each milestone.
