import networkx as nx
import numpy as np
from collections import defaultdict
from centrality import compute_centrality


def _build_score_weighted(edges, label):
    """Build a DiGraph where edge weight = sum of karma scores instead of reply count."""
    G = nx.DiGraph()
    score_weights = defaultdict(int)
    for e in edges:
        score_weights[(e["source"], e["target"])] += max(e.get("score", 0), 0)
    for (src, tgt), weight in score_weights.items():
        if weight > 0:
            G.add_edge(src, tgt, weight=weight)
    return G


def alternate_specs(G_bots, G_humans, df_bots, df_humans, G_ira=None, df_ira=None,
                    edges_bots=None, edges_humans=None, edges_ira=None):
    print(f"\n{'='*50}")
    print("ALTERNATE SPECIFICATIONS")
    print(f"{'='*50}")

    networks = [("Bots", G_bots, df_bots), ("Humans", G_humans, df_humans)]
    if G_ira is not None:
        networks.append(("IRA", G_ira, df_ira))

    # Spec 1: Remove low-degree nodes (near-isolates)
    print("\n[Spec 1] Remove nodes with degree < 2 (near-isolates)...")
    for label, G, _ in networks:
        low = [n for n, d in G.degree() if d < 2]
        G2  = G.copy(); G2.remove_nodes_from(low)
        print(f"  {label}: removed {len(low)} nodes → "
              f"{G2.number_of_nodes()} remain, "
              f"density {nx.density(G2):.5f}")

    # Spec 2: Edge weight distribution
    print("\n[Spec 2] Edge weight distribution...")
    for label, G, _ in networks:
        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        multi   = sum(w > 1 for w in weights)
        print(f"  {label}: avg weight={np.mean(weights):.2f}, "
              f"max={max(weights)}, "
              f"multi-edges={100*multi/len(weights):.1f}%")

    # Spec 3: Undirected
    print("\n[Spec 3] Treat as undirected...")
    for label, G, _ in networks:
        Gu   = G.to_undirected()
        ccs  = list(nx.connected_components(Gu))
        frac = max(len(c) for c in ccs) / Gu.number_of_nodes()
        print(f"  {label}: {len(ccs)} components, "
              f"largest = {frac:.1%} of nodes, "
              f"density = {nx.density(Gu):.5f}")

    # Spec 4: Normalised centrality comparison
    print("\n[Spec 4] Mean centrality — normalised comparison...")
    header_labels = ["Bots", "Humans"] + (["IRA"] if G_ira is not None else [])
    headers = "".join(f"{label:>12}" for label in header_labels)
    print(f"  {'Measure':<15}{headers}")
    print(f"  {'-'*55}")

    dfs = [(df_bots, "Bots"), (df_humans, "Humans")]
    if df_ira is not None:
        dfs.append((df_ira, "IRA"))

    for m in ["Norm Total Degree", "Norm Betweenness", "Norm Closeness", "Norm Eigenvector"]:
        vals = "".join(f"{df[m].mean():>12.5f}" for df, _ in dfs)
        print(f"  {m:<22}{vals}")

    # Spec 5: Score-weighted edges vs count-weighted edges
    if edges_bots and edges_humans:
        print("\n[Spec 5] Score-weighted edges (weight = karma sum) vs count-weighted...")
        print(f"  {'Network':<10} {'Measure':<22} {'Count-weighted':>16} {'Score-weighted':>16} {'Diff':>10}")
        print(f"  {'-'*70}")

        raw_networks = [("Bots", G_bots, df_bots, edges_bots),
                        ("Humans", G_humans, df_humans, edges_humans)]
        if G_ira is not None and edges_ira:
            raw_networks.append(("IRA", G_ira, df_ira, edges_ira))

        for label, G_count, df_count, edges in raw_networks:
            G_score = _build_score_weighted(edges, label)
            if G_score.number_of_nodes() < 2:
                print(f"  {label}: insufficient score-weighted edges to compare")
                continue
            try:
                df_score = compute_centrality(G_score, f"{label}_score", save_heatmap=False)
            except Exception as e:
                print(f"  {label}: could not compute score-weighted centrality ({e})")
                continue

            for m in ["Norm Total Degree", "Norm Betweenness", "Norm Closeness"]:
                c_val = df_count[m].mean()
                s_val = df_score[m].mean()
                diff  = s_val - c_val
                print(f"  {label:<10} {m:<22} {c_val:>16.5f} {s_val:>16.5f} {diff:>+10.5f}")
        print()
        print("  Interpretation: positive diff = score-weighting raises centrality")
        print("  (nodes connected by high-karma replies become more central)")
