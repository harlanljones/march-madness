# March Madness Bracket Predictor

A PyTorch-based ML pipeline that predicts NCAA March Madness tournament outcomes using historical data from Kaggle's [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) competition.

## How It Works

The model is a small feedforward neural network (BracketNet) trained on 20+ years of tournament results (1985-2025, ~2,500 games). For each possible matchup it computes 23 features as team-A-minus-team-B differences -- seed, offensive/defensive efficiency, Massey ordinal rank, shooting percentages, turnover rate, and more -- then outputs a win probability.

**Architecture:** Input(23) &rarr; 128 &rarr; 64 &rarr; 32 &rarr; 1 (sigmoid), with BatchNorm and Dropout at each layer.

Three bracket generation modes are supported:

- **Deterministic** -- always pick the higher-probability team
- **Probability** -- show win probabilities for every matchup
- **Monte Carlo** -- simulate the tournament 10,000 times and report championship frequencies

## Setup

Requires Python 3.12+.

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data

Install the Kaggle CLI and download the 2026 competition dataset:

```bash
uv pip install kaggle
export KAGGLE_USERNAME="your-username"
export KAGGLE_KEY="your-api-key"
kaggle competitions download -c march-machine-learning-mania-2026 -p data/raw/
cd data/raw && unzip march-machine-learning-mania-2026.zip && rm march-machine-learning-mania-2026.zip
```

You need to [accept the competition rules](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/rules) before downloading.

The dataset includes:

| File | Rows | Description |
|------|------|-------------|
| MRegularSeasonDetailedResults.csv | 124,031 | Game-level box scores (2003-2026) |
| MRegularSeasonCompactResults.csv | 198,079 | Score-only results (1985-2026) |
| MNCAATourneyCompactResults.csv | 2,585 | Historical tournament results (1985-2025) |
| MNCAATourneyDetailedResults.csv | 1,449 | Tournament box scores (2003-2025) |
| MNCAATourneySeeds.csv | 2,626 | Tournament seeds (1985-2025) |
| MMasseyOrdinals.csv | 5,819,228 | Third-party team rankings (2003-2026) |
| MTeams.csv | 381 | Team IDs and names |
| MSeasons.csv | 42 | Season metadata (1985-2026) |

2026 tournament seeds will be added to `MNCAATourneySeeds.csv` after Selection Sunday.

## Usage

### Train

Train on 2003-2023, validate on 2024-2025:

```bash
python main.py train --data-dir data/raw --train-end 2023 --val-start 2024 --val-end 2025
```

Train a 5-model ensemble:

```bash
python main.py train --data-dir data/raw --train-end 2023 --val-start 2024 --val-end 2025 --ensemble 5
```

Models are saved to `outputs/models/`.

### Evaluate

Evaluate against known tournament results:

```bash
python main.py evaluate --model outputs/models/best.pt --data-dir data/raw --season-start 2025 --season-end 2025
```

Outputs log loss, accuracy, and an optional calibration plot (`--calibration path.png`).

### Generate Bracket

Once 2026 seeds are available:

```bash
python main.py bracket --model outputs/models/best.pt --season 2026 --method deterministic
python main.py bracket --model outputs/models/best.pt --season 2026 --method monte_carlo --simulations 10000
```

Brackets are printed to the terminal and saved as JSON to `outputs/brackets/`.

## Features (23 per matchup)

All features are computed as Team A minus Team B differences (Team A = lower TeamID by Kaggle convention):

| Feature | Source |
|---------|--------|
| Seed difference | MNCAATourneySeeds |
| Net/offensive/defensive efficiency | MRegularSeasonDetailedResults |
| Massey ordinal rank | MMasseyOrdinals |
| Effective FG%, 3PT%, FT% | DetailedResults |
| Turnover rate, opponent turnover rate | DetailedResults |
| Win %, scoring margin, PPG, PAPG | CompactResults |
| Free throw rate, offensive rebound % | DetailedResults |
| Road win %, assist rate, steal/block rate | DetailedResults |
| Opponent effective FG% | DetailedResults |
| Scoring consistency, last-10 momentum | Derived |

Data augmentation enforces symmetry: each (A, B) matchup is also trained as (B, A) with negated features and flipped label.

## Project Structure

```
march-madness/
├── main.py                    # CLI: train / evaluate / bracket
├── src/
│   ├── utils.py               # Seed parsing, team name lookup
│   ├── data_loader.py         # Load Kaggle CSVs, aggregate season stats
│   ├── features.py            # 23-feature matchup vector engineering
│   ├── dataset.py             # PyTorch Dataset, flip augmentation, scaling
│   ├── model.py               # BracketNet (3-layer NN)
│   ├── train.py               # Training loop, early stopping, ensemble, save/load
│   ├── evaluate.py            # Log loss, accuracy, calibration, ESPN scoring
│   └── bracket.py             # Bracket simulation (deterministic / Monte Carlo)
├── tests/                     # 100 tests covering all modules
├── data/raw/                  # Kaggle CSVs (not committed)
├── outputs/models/            # Saved .pt checkpoints
├── outputs/brackets/          # Generated bracket JSON
└── requirements.txt
```

## Tests

```bash
python -m pytest tests/ -v
```

100 tests validate seed parsing, stat aggregation, feature antisymmetry, model output ranges, training convergence, save/load roundtrips, ESPN scoring, bracket structure, and Monte Carlo simulation.

## Training Details

- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=7)
- **Early stopping:** patience=20, restores best weights
- **Temporal split:** train on past seasons, validate on recent seasons (no data leakage)
- **Targets:** log loss < 0.50 (seed-only baseline ~0.55), accuracy 74-76%

## 2026 Season Preview

Top teams by net efficiency (offensive - defensive points per 100 possessions):

| Team | Win% | Net Eff | Off Eff | Def Eff |
|------|------|---------|---------|---------|
| Duke | .931 | +29.5 | 121.7 | 92.2 |
| Michigan | .931 | +28.8 | 122.7 | 93.9 |
| Arizona | .931 | +26.4 | 120.3 | 93.8 |
| St Louis | .893 | +25.7 | 121.8 | 96.2 |
| Gonzaga | .903 | +25.1 | 119.2 | 94.1 |
| Florida | .793 | +23.3 | 120.8 | 97.5 |
| Houston | .828 | +21.9 | 116.8 | 95.0 |
| Iowa St | .828 | +21.8 | 118.9 | 97.1 |
| Connecticut | .900 | +21.1 | 117.7 | 96.6 |
| St Mary's CA | .867 | +20.8 | 118.1 | 97.3 |
