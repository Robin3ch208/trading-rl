# forex_rl_trading

Clean, professional R&D repository for FX trading with PPO (Stable-Baselines3) on EURUSD tick data.

## Goals (R&D / R&D context)
- Reproducible research: configs + deterministic seeds + traceable results
- Professional codebase: package layout, tests, typing (pragmatic), CI
- Produce a LaTeX report describing the scientific/technical work

## Repo layout
- `src/forex_rl_trading/`: python package (env + features)
- `tests/`: unit + smoke tests
- `train_forex_ppo.py`: training entrypoint (SB3 PPO)

## Installation
Base install (env + features):

```bash
python -m pip install -e .
```

Dev tools (tests/lint):

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Training dependencies (SB3 + Torch):

```bash
python -m pip install -e ".[rl]"
```

## Run training
Put your (non-versioned) tick data CSV in `data/`, then:

```bash
python train_forex_ppo.py
```



