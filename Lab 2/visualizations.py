import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import OUTPUT_DIR

BOT_COLOR = "#e74c3c"
HUM_COLOR = "#3498db"
IRA_COLOR = "#f39c12"


def plot_all(df_bots, df_humans, G_bots, G_humans, corr_bots, corr_humans,
             df_ira=None, G_ira=None, corr_ira=None):
    measures  = ["Norm Total Degree", "Norm Betweenness", "Norm Closeness", "Norm Eigenvector"]
    has_ira   = df_ira is not None

    networks = [
        (df_bots,   "Bots (SubSimGPT2)",    BOT_COLOR),
        (df_humans, "Humans (changemyview)", HUM_COLOR),
    ]
    if has_ira:
        networks.append((df_ira, "IRA Bots", IRA_COLOR))

    # ── 1. Degree distributions ───────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Degree Distribution: All Networks", fontsize=14, fontweight="bold")
    for ax, m in zip(axes, ["Norm Total Degree", "Norm In-Degree", "Norm Out-Degree"]):
        for df, label, color in networks:
            ax.hist(df[m], bins=30, alpha=0.5, color=color, label=label, density=True)
        ax.set_title(m.replace("_", " ").title())
        ax.set_xlabel("Centrality Value"); ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_degree_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ 01_degree_distributions.png")

    # ── 2. Centrality boxplots ────────────────
    frames = []
    for df, label, _ in networks:
        tmp = df[measures].copy(); tmp["network"] = label; frames.append(tmp)
    melted = pd.concat(frames).melt(id_vars="network", var_name="measure", value_name="value")

    palette = {label: color for _, label, color in networks}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Centrality Measures: All Networks", fontsize=14, fontweight="bold")
    for ax, m in zip(axes.flat, measures):
        sns.boxplot(data=melted[melted["measure"] == m],
                    x="network", y="value", hue="network",
                    palette=palette, legend=False, ax=ax)
        ax.set_title(m.capitalize()); ax.set_xlabel(""); ax.set_ylabel("Value")
        ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_centrality_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ 02_centrality_boxplots.png")

    # ── 3. Correlation heatmaps ───────────────
    cols       = ["Norm In-Degree", "Norm Out-Degree", "Norm Total Degree", "Norm Betweenness", "Norm Closeness", "Norm Eigenvector"]
    n_heatmaps = 3 if has_ira else 2
    fig, axes  = plt.subplots(1, n_heatmaps, figsize=(7 * n_heatmaps, 6))
    fig.suptitle("Centrality Correlation Matrices", fontsize=14, fontweight="bold")

    heatmap_data = [
        (axes[0], corr_bots,   "Bots — r/SubSimulatorGPT2"),
        (axes[1], corr_humans, "Humans — r/changemyview"),
    ]
    if has_ira:
        heatmap_data.append((axes[2], corr_ira, "IRA Bots"))

    for ax, corr, title in heatmap_data:
        sns.heatmap(corr.loc[cols, cols], annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_correlation_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ 03_correlation_heatmaps.png")

    # ── 4. Network graphs ─────────────────────
    graph_data = [
        (G_bots,   df_bots,   "Bots — r/SubSimulatorGPT2", BOT_COLOR),
        (G_humans, df_humans, "Humans — r/changemyview",    HUM_COLOR),
    ]
    if has_ira:
        graph_data.append((G_ira, df_ira, "IRA Bots", IRA_COLOR))
    for G, df, label, color in graph_data:
        _plot_network(G, df, label, color)

    # ── 5. Summary stats bar chart ────────────
    stat_keys = ["Density", "Reciprocity", "Avg Betweenness", "Avg Closeness", "Avg Eigenvector"]
    def stats_for(G, df):
        return [
            nx.density(G),
            nx.reciprocity(G),
            df["Norm Betweenness"].mean(),
            df["Norm Closeness"].mean(),
            df["Norm Eigenvector"].mean(),
        ]

    all_stats = [
        ("Bots",   stats_for(G_bots,   df_bots),   BOT_COLOR),
        ("Humans", stats_for(G_humans, df_humans), HUM_COLOR),
    ]
    if has_ira:
        all_stats.append(("IRA", stats_for(G_ira, df_ira), IRA_COLOR))

    n_groups = len(stat_keys)
    n_bars   = len(all_stats)
    w        = 0.8 / n_bars
    x        = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (label, vals, color) in enumerate(all_stats):
        offset = (i - n_bars / 2 + 0.5) * w
        bars   = ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=6)

    ax.set_xticks(x); ax.set_xticklabels(stat_keys)
    ax.set_ylabel("Value")
    ax.set_title("Network Summary Comparison", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_summary_stats.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ 05_summary_stats.png")


def _plot_network(G, df, label, accent_color, max_nodes=120):
    """Force-directed graph: size=degree, color=betweenness."""
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    G_vis = G.subgraph(largest_wcc).copy()
    if G_vis.number_of_nodes() > max_nodes:
        top   = sorted(G_vis.nodes(), key=lambda n: G_vis.degree(n), reverse=True)[:max_nodes]
        G_vis = G_vis.subgraph(top).copy()

    pos      = nx.spring_layout(G_vis, k=1.2, iterations=60, seed=42)
    nodes_in = [n for n in G_vis.nodes() if n in df.index]
    sizes    = [max(df.loc[n, "Norm Total Degree"] * 4000, 30) for n in nodes_in]
    colors   = [df.loc[n, "Norm Betweenness"] for n in nodes_in]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#0f0f1a"); fig.patch.set_facecolor("#0f0f1a")

    nx.draw_networkx_edges(G_vis, pos, ax=ax, alpha=0.12,
                           edge_color="#cccccc", arrows=True, arrowsize=6, width=0.4)
    sc = nx.draw_networkx_nodes(G_vis, pos, ax=ax,
                                nodelist=nodes_in, node_size=sizes,
                                node_color=colors, cmap=plt.cm.plasma, alpha=0.9)
    top10 = df.loc[nodes_in].nlargest(10, "Norm Total Degree").index.tolist()
    nx.draw_networkx_labels(G_vis, pos, {n: n for n in top10 if n in pos},
                            ax=ax, font_size=7, font_color="white")

    plt.colorbar(sc, ax=ax, label="Betweenness Centrality", fraction=0.025, pad=0.02)
    ax.set_title(f"{label}\n(size=degree centrality · color=betweenness)",
                 color="white", fontsize=12, pad=12)
    ax.axis("off")

    fname = label.lower().replace(" ", "_").replace("/", "").replace("—", "")[:40]
    fpath = f"{OUTPUT_DIR}/04_network_{fname}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ 04_network_{fname}.png")
