import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import OUTPUT_DIR


def compute_centrality(G, label, save_heatmap=True):
    print(f"\n{'='*50}")
    print(f"CENTRALITY MEASURES — {label}")
    print(f"{'='*50}")

    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    G_sub = G.subgraph(largest_wcc).copy()

    # === Compute Centrality Measures (Normalized) ===
    norm_degree_centrality     = nx.degree_centrality(G_sub)
    norm_in_degree_centrality  = nx.in_degree_centrality(G_sub)
    norm_out_degree_centrality = nx.out_degree_centrality(G_sub)
    norm_betweenness_centrality = nx.betweenness_centrality(G_sub, normalized=True)
    norm_closeness_centrality  = nx.closeness_centrality(G_sub)

    try:
        norm_eigenvector_centrality = nx.eigenvector_centrality(G_sub, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("  Eigenvector did not converge — using degree as proxy")
        norm_eigenvector_centrality = norm_degree_centrality

    # === Compute Centrality Measures (Unnormalized) ===
    raw_in_degree    = dict(G_sub.in_degree())
    raw_out_degree   = dict(G_sub.out_degree())
    raw_total_degree = {n: raw_in_degree[n] + raw_out_degree[n] for n in G_sub.nodes()}
    raw_betweenness  = nx.betweenness_centrality(G_sub, normalized=False)

    # === Create DataFrame ===
    df_centrality = pd.DataFrame({
        "Node":               list(G_sub.nodes()),
        "Network":            label,
        "Norm In-Degree":     [norm_in_degree_centrality[n]   for n in G_sub.nodes()],
        "Norm Out-Degree":    [norm_out_degree_centrality[n]  for n in G_sub.nodes()],
        "Norm Total Degree":  [norm_degree_centrality[n]      for n in G_sub.nodes()],
        "Norm Betweenness":   [norm_betweenness_centrality[n] for n in G_sub.nodes()],
        "Norm Closeness":     [norm_closeness_centrality[n]   for n in G_sub.nodes()],
        "Norm Eigenvector":   [norm_eigenvector_centrality[n] for n in G_sub.nodes()],
        "Raw In-Degree":      [raw_in_degree[n]               for n in G_sub.nodes()],
        "Raw Out-Degree":     [raw_out_degree[n]              for n in G_sub.nodes()],
        "Raw Total Degree":   [raw_total_degree[n]            for n in G_sub.nodes()],
        "Raw Betweenness":    [raw_betweenness[n]             for n in G_sub.nodes()],
        "Comment Count":      [G_sub.nodes[n].get("comment_count", 0) for n in G_sub.nodes()],
        "Avg Score":          [G_sub.nodes[n].get("avg_score", 0)     for n in G_sub.nodes()],
    }).set_index("Node")

    print(f"\n  Nodes in largest component: {len(df_centrality)}")
    print(f"\n=== Centrality Measures — First 5 Rows ===")
    print(df_centrality[["Norm In-Degree","Norm Out-Degree","Norm Total Degree",
                          "Norm Betweenness","Norm Closeness","Norm Eigenvector"]].head().to_string())

    # === Correlation Matrix ===
    norm_cols = ["Norm In-Degree","Norm Out-Degree","Norm Total Degree",
                 "Norm Betweenness","Norm Closeness","Norm Eigenvector"]
    corr = df_centrality[norm_cols].corr()
    print(f"\n=== Correlation Matrix of Centrality Measures — {label} ===")
    print(corr.round(3).to_string())

    # === Heatmap ===
    if save_heatmap:
        fname = label.lower().replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title(f"Correlation Matrix of Centrality Measures\n{label}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/centrality_corr_{fname}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Heatmap saved: centrality_corr_{fname}.png")

    return df_centrality


def highlight_central_nodes(df, label, top_n=5):
    print(f"\n{'='*50}")
    print(f"KEY NODES — {label}")
    print(f"{'='*50}")

    # map normalized measure → its raw counterpart for display
    raw_map = {
        "Norm Total Degree":  "Raw Total Degree",
        "Norm In-Degree":     "Raw In-Degree",
        "Norm Out-Degree":    "Raw Out-Degree",
        "Norm Betweenness":   "Raw Betweenness",
        "Norm Closeness":     None,
        "Norm Eigenvector":   None,
    }

    for measure in raw_map:
        raw_col = raw_map[measure]
        print(f"\n  {measure} — Top {top_n}:")
        for node, val in df[measure].nlargest(top_n).items():
            raw = f"  (raw: {df.loc[node, raw_col]:.0f})" if raw_col else ""
            print(f"    {str(node):<35} {val:.5f}{raw}")
        print(f"  {measure} — Bottom {top_n}:")
        for node, val in df[measure].nsmallest(top_n).items():
            raw = f"  (raw: {df.loc[node, raw_col]:.0f})" if raw_col else ""
            print(f"    {str(node):<35} {val:.5f}{raw}")
