# Lab 2 — Reddit Network Analysis (Dead Internet Theory)

Analyses three Reddit interaction networks to test the Dead Internet Theory:
- **r/SubSimulatorGPT2** — GPT-2 bots (explicit sandbox)
- **r/changemyview** — human users
- **IRA Bots** — Russian Internet Research Agency accounts (archived dataset)

## Setup

```bash
pip install networkx pandas matplotlib seaborn requests
```

## Run

```bash
python main.py
```

This will:
1. Collect comment data from Reddit (r/SubSimulatorGPT2 and r/changemyview) and load the IRA archived dataset
2. Build directed weighted interaction graphs for each network
3. Compute centrality measures (degree, betweenness, closeness, eigenvector)
4. Generate correlation matrices and visualisations
5. Run alternate specifications for robustness checks
6. Save all outputs to `outputs/`

## Output files

| File | Description |
|------|-------------|
| `outputs/01_degree_distributions.png` | Degree distribution histograms |
| `outputs/02_centrality_boxplots.png` | Centrality measure boxplots |
| `outputs/03_correlation_heatmaps.png` | Centrality correlation matrices |
| `outputs/04_network_*.png` | Force-directed network graphs |
| `outputs/05_summary_stats.png` | Summary statistics bar chart |
| `outputs/centrality_*.csv` | Per-node centrality values |
| `outputs/corr_*.csv` | Correlation matrices |

## File structure

```
main.py                 # Entry point — runs full pipeline
config.py               # Output directory and request headers
data_collection.py      # Reddit API + IRA CSV collection
network_construction.py # Builds NetworkX DiGraphs
topography.py           # Network structure description
centrality.py           # Centrality computation and heatmaps
correlations.py         # Correlation matrices
visualizations.py       # All plots
alternate_specs.py      # Robustness checks (5 specifications)
```
