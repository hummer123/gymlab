# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive reinforcement learning laboratory that combines:
- **OpenAI Gym fork** (`gym/`): Core RL environment framework with various environments
- **Algorithm implementations**: Multiple RL algorithms from traditional to deep learning approaches
- **Custom environments**: 21Card game and Markov Decision Process implementations
- **Educational focus**: Well-documented implementations suitable for learning and research

## Common Commands

### Gym Framework Development (in `gym/` directory)

```bash
# Install gym with all dependencies
pip install -e .[all]

# Install specific environment groups
pip install -e .[classic_control,box2d,toy_text]  # Basic environments
pip install -e .[mujoco]                           # Physics simulation
pip install -e .[atari]                            # Atari games

# Install development dependencies
pip install -e .[testing]

# Run pre-commit hooks (formatting, linting, type checking)
pre-commit run --all-files

# Run specific tools manually
black gym/
flake8 gym/
isort gym/
pyright --project=gym/pyproject.toml
```

### Testing

```bash
# From project root - run gym test suite
cd gym && python -m pytest tests/ -v

# Run specific test modules
cd gym && python -m pytest tests/test_core.py -v
cd gym && python -m pytest tests/spaces/ -v
cd gym && python -m pytest tests/envs/ -v

# Run tests with coverage
cd gym && python -m pytest tests/ --cov=gym

# Run single test file
cd gym && python -m pytest tests/test_core.py::test_step -v
```

### Algorithm Training

```bash
# DQN on CartPole
cd cartPole && python dqn_cartPole.py

# Policy Gradient on CartPole
cd policyGradient && python cartpole_pg.py

# Actor-Critic on CartPole
cd policyGradient && python cartpole_ac.py
```

## Architecture Overview

### Core Components

1. **Gym Framework (`gym/gym/`)**:
   - `core.py`: Base `Env` and `Wrapper` classes defining the RL API
   - `envs/`: Environment implementations organized by type
     - `classic_control/`: CartPole, MountainCar, etc.
     - `mujoco/`: Physics-based environments
     - `box2d/`: 2D physics environments
     - `toy_text/`: Simple text-based environments
   - `spaces/`: Action and observation space definitions
   - `wrappers/`: Environment modification utilities

2. **Algorithm Implementations**:
   - **CartPole (`cartPole/`)**: Q-learning discretization, DQN implementations
   - **Policy Gradient (`policyGradient/`)**: REINFORCE, Actor-Critic algorithms
   - **Custom Games (`21Card/`, `mdp/`)**: Custom environment implementations

3. **Visualization (`plot/`)**: Real-time plotting and SBUS data simulation

### Key Patterns

- **Standard Gym API**: All environments follow `reset()`, `step()`, `render()` interface
- **Model Persistence**: Agents save best models to `./model/` directories using PyTorch
- **Training Loop Pattern**: Consistent training/evaluation separation with periodic model saving
- **Configuration Dictionaries**: Hyperparameters passed as config dicts to agents
- **Exception Handling**: Robust error handling in training loops with detailed logging

### Dependencies

- **Core**: numpy>=1.18.0, cloudpickle>=1.2.0, gym_notices>=0.0.4
- **Rendering**: pygame==2.1.0, opencv-python>=3.0, matplotlib>=3.0
- **Physics**: box2d-py==2.3.5, mujoco==2.2, mujoco_py<2.2,>=2.1
- **ML**: PyTorch (used in algorithm implementations)
- **Testing**: pytest==7.0.1
- **Code Quality**: black, flake8, isort, pydocstyle, pyright

## Development Workflow

### Adding New Environments

1. Inherit from `gym.Env` in `gym/gym/core.py`
2. Implement required methods: `step()`, `reset()`, `render()`, `close()`
3. Define `action_space`, `observation_space`, `reward_range`
4. Register environment in `gym/gym/envs/registration.py`
5. Add tests in `gym/tests/envs/`

### Adding New Algorithms

1. Create agent class in appropriate directory (`cartPole/`, `policyGradient/`, etc.)
2. Follow existing patterns for model saving/loading
3. Implement training loop with periodic evaluation
4. Save best model using `./model/best_[algorithm]_model.pth` convention
5. Use configuration dictionaries for hyperparameters

### Code Quality Standards

- **Formatting**: Black code formatter
- **Linting**: Flake8 with per-file ignores in `gym/.pre-commit-config.yaml`
- **Import Sorting**: isort with black profile
- **Type Checking**: pyright with basic mode (imports ignored due to optional dependencies)
- **Documentation**: Google-style docstrings (pydocstyle)

### Testing Approach

- **Comprehensive Coverage**: 61 test files covering all major components
- **Environment Testing**: Each environment has dedicated tests
- **API Compliance**: Core Gym API thoroughly tested
- **Integration Testing**: Wrapper and space functionality tested together
- **Test Structure**: Uses pytest with custom test utilities

## File Structure Conventions

```
algoritmDirectory/
├── algorithm_agent.py          # Base agent implementation
├── algorithm_env.py           # Training script with environment setup
├── netAgent.py               # Neural network architectures (if applicable)
├── utils.py                  # Algorithm-specific utilities
└── model/                    # Directory for saved models
    ├── best_[name]_model.pth # Best performing model
    └── final_[name]_model.pth# Final trained model
```

## Important Notes

- **Python Support**: Supports Python 3.6-3.10, with 3.7+ recommended
- **Platform Support**: Linux/macOS officially supported
- **Model Paths**: Use relative paths like `./model/best_model.pth` for consistency
- **Error Handling**: Implement robust exception handling in training loops
- **Evaluation**: Separate training and evaluation phases with model loading
- **Visualization**: Use matplotlib for training curves and performance plots