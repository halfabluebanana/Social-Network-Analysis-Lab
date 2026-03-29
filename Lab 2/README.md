# Lab 2 — Reddit Network Analysis (Dead Internet Theory)

Analyses three Reddit interaction networks to test the Dead Internet Theory:
- **r/SubSimulatorGPT2** — GPT-2 bots (explicit sandbox)
- **r/changemyview** — human users
- **IRA Bots** — Russian Internet Research Agency accounts (archived dataset)

## Lab Questions

**1.** Find a complete social network, preferably one with at least some attributes about the nodes with it. (If you simply have a social network, but no real attributes, you will need to pick an additional network to compare that first one to.)

Describe the social network(s) to me, in terms of how it was collected, what it represents and so forth. Also give me basic topography of the network: the nature of the ties; direction of ties; overall density; and if attributes are with the network, the distribution of the categories and variables of those attributes.

**2.** Calculate degree centrality (in- and out-degree, too, if you have such data); closeness centrality; betweenness centrality; and eigenvector centrality. Correlate those measures of centrality. Highlight which nodes are most central and least central, along different dimensions.

**3.** Now, do 1 of the following, but not both:

3a. If you have a network with attribute data, then state some hypothesis about how an attribute may be related to some (or all of the) measures of centrality. Explain why you think these two variables should be related.

3b. If you don't have a network with attribute data, then pick another network to compare your first network against. Calculate all of the same measures as above for Network #2. Consider if normalization is appropriate for any of these measures. Then state some hypothesis about why some (or all of the) measures of centrality in one network will be the same or different from the second network. Explain why you think these two networks should be similar or different.

In either case, when you are done above, then consider alternate specifications of your variables and codings and decisions and models. What would you want to consider changing and why. If you can, report on what are the consequences of those changes?

**4.** Lastly, give your best conclusion as to what you learned from your analysis. Did it make sense, given your initial expectations? Why? Why not?

---

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

- `outputs/01_degree_distributions.png` — Degree distribution histograms
- `outputs/02_centrality_boxplots.png` — Centrality measure boxplots
- `outputs/03_correlation_heatmaps.png` — Centrality correlation matrices
- `outputs/04_network_*.png` — Force-directed network graphs
- `outputs/05_summary_stats.png` — Summary statistics bar chart
- `outputs/centrality_*.csv` — Per-node centrality values
- `outputs/corr_*.csv` — Correlation matrices

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
