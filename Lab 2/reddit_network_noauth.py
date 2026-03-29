"""
Dead Internet Theory — Reddit Network Analysis
Lab 2, Option 2, Route 3b

NO API CREDENTIALS NEEDED — uses Reddit's public JSON endpoints.

Compares:
  Network 1 (Bots)   — r/SubSimulatorGPT2
  Network 2 (Humans) — r/changemyview
"""

import requests
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict
import time
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

HEADERS = {"User-Agent": "network_lab_academic/0.1"}
OUTPUT_DIR = "/Users/adelinesetiawan/Documents/Columbia Classes/Social Network Analysis/Lab/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. DATA COLLECTION — public JSON API
# ─────────────────────────────────────────────

def fetch_posts(subreddit, n_posts=10):
    """Fetch hot posts from a subreddit using public JSON endpoint."""
    posts = []
    after = None

    while len(posts) < n_posts:
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
        if after:
            url += f"&after={after}"

        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            data = r.json()["data"]
            batch = data["children"]
            if not batch:
                break
            posts.extend(batch)
            after = data.get("after")
            if not after:
                break
            time.sleep(1)  # polite delay
        except Exception as e:
            print(f"Error fetching posts from {subreddit}: {e}")
            break

    return posts[:n_posts]


def fetch_comments(subreddit, post_id, post_title):
    """Fetch full comment tree for a single post."""
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json?limit=500"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data[1]["data"]["children"]  # index 0 is post, 1 is comments
    except Exception as e:
        print(f"Error fetching comments for {post_id}: {e}")
        return []


def parse_comment_tree(comments, parent_author=None, edges=None, node_data=None):
    if edges is None:
        edges = []
    if node_data is None:
        node_data = {}
    """
    Recursively walk comment tree, extract user→user reply edges.
    Reddit comment trees can be deeply nested.
    """
    for item in comments:
        if item["kind"] != "t1":  # t1 = comment
            continue

        data = item["data"]
        author = data.get("author", "[deleted]")

        if author in ("[deleted]", "AutoModerator", None):
            continue

        # track node attributes
        if author not in node_data:
            node_data[author] = {
                "comment_count": 0,
                "total_score": 0,
            }
        node_data[author]["comment_count"] += 1
        node_data[author]["total_score"] += data.get("score", 0)

        # create edge if this is a reply to another user
        if parent_author and parent_author != author:
            edges.append({
                "source": author,         # replier
                "target": parent_author,  # replied-to
                "score": data.get("score", 0)
            })

        # recurse into replies
        replies = data.get("replies", "")
        if isinstance(replies, dict):
            reply_comments = replies["data"]["children"]
            parse_comment_tree(reply_comments, author, edges, node_data)

    return edges, node_data


def collect_network(subreddit, n_posts=15, label=""):
    """Full collection pipeline for one subreddit."""
    print(f"\n{'='*50}")
    print(f"📥 Collecting r/{subreddit} ({label})")
    print(f"{'='*50}")

    posts = fetch_posts(subreddit, n_posts=n_posts)
    print(f"  Fetched {len(posts)} posts")

    all_edges = []
    all_nodes = {}

    for i, post in enumerate(posts):
        pdata = post["data"]
        post_id = pdata["id"]
        title = pdata["title"][:60]
        author = pdata.get("author", "[deleted]")

        print(f"  [{i+1}/{len(posts)}] {title}...")

        comments = fetch_comments(subreddit, post_id, title)
        edges, nodes = parse_comment_tree(comments, parent_author=author,
                                          edges=[], node_data={})

        # merge node data
        for user, stats in nodes.items():
            if user not in all_nodes:
                all_nodes[user] = {"comment_count": 0, "total_score": 0}
            all_nodes[user]["comment_count"] += stats["comment_count"]
            all_nodes[user]["total_score"]   += stats["total_score"]

        all_edges.extend(edges)
        time.sleep(1.5)  # rate limiting — be polite

    print(f"\n  ✅ Total: {len(all_edges)} edges, {len(all_nodes)} unique users")
    return all_edges, all_nodes


# ─────────────────────────────────────────────
# 2. NETWORK CONSTRUCTION
# ─────────────────────────────────────────────

def build_network(edges, node_data, label):
    """Build directed weighted NetworkX graph."""
    G = nx.DiGraph()

    # aggregate edge weights
    edge_weights = defaultdict(int)
    for e in edges:
        edge_weights[(e["source"], e["target"])] += 1

    for (src, tgt), weight in edge_weights.items():
        G.add_edge(src, tgt, weight=weight)

    # node attributes
    for node in G.nodes():
        nd = node_data.get(node, {})
        G.nodes[node]["comment_count"] = nd.get("comment_count", 0)
        cc = nd.get("comment_count", 1)
        G.nodes[node]["avg_score"] = nd.get("total_score", 0) / max(cc, 1)
        G.nodes[node]["network"] = label

    print(f"\n🔗 Network '{label}'")
    print(f"   Nodes       : {G.number_of_nodes()}")
    print(f"   Edges       : {G.number_of_edges()}")
    print(f"   Density     : {nx.density(G):.5f}")
    try:
        print(f"   Reciprocity : {nx.reciprocity(G):.4f}")
    except:
        print(f"   Reciprocity : N/A")

    return G


# ─────────────────────────────────────────────
# 3. TOPOGRAPHY
# ─────────────────────────────────────────────

def describe_network(G, label):
    print(f"\n{'='*50}")
    print(f"TOPOGRAPHY — {label}")
    print(f"{'='*50}")

    degrees    = dict(G.degree())
    in_deg     = dict(G.in_degree())
    out_deg    = dict(G.out_degree())

    print(f"Nodes              : {G.number_of_nodes()}")
    print(f"Edges              : {G.number_of_edges()}")
    print(f"Density            : {nx.density(G):.6f}")
    try:
        print(f"Reciprocity        : {nx.reciprocity(G):.4f}")
    except:
        pass
    print(f"Avg total degree   : {np.mean(list(degrees.values())):.2f}")
    print(f"Avg in-degree      : {np.mean(list(in_deg.values())):.2f}")
    print(f"Avg out-degree     : {np.mean(list(out_deg.values())):.2f}")
    print(f"Max degree node    : {max(degrees, key=degrees.get)} "
          f"(degree={max(degrees.values())})")

    wccs = list(nx.weakly_connected_components(G))
    print(f"Weakly conn comps  : {len(wccs)}")
    print(f"Largest component  : {max(len(c) for c in wccs)} nodes "
          f"({100*max(len(c) for c in wccs)/G.number_of_nodes():.1f}%)")


# ─────────────────────────────────────────────
# 4. CENTRALITY
# ─────────────────────────────────────────────

def compute_centrality(G, label):
    print(f"\n📐 Computing centrality — '{label}'...")

    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    G_sub = G.subgraph(largest_wcc).copy()

    degree_c     = nx.degree_centrality(G_sub)
    in_degree_c  = nx.in_degree_centrality(G_sub)
    out_degree_c = nx.out_degree_centrality(G_sub)
    betweenness_c = nx.betweenness_centrality(G_sub, normalized=True)
    closeness_c  = nx.closeness_centrality(G_sub)

    try:
        eigenvector_c = nx.eigenvector_centrality(G_sub, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("  ⚠ Eigenvector did not converge — using degree as proxy")
        eigenvector_c = degree_c

    rows = []
    for node in G_sub.nodes():
        rows.append({
            "node":          node,
            "network":       label,
            "degree":        degree_c.get(node, 0),
            "in_degree":     in_degree_c.get(node, 0),
            "out_degree":    out_degree_c.get(node, 0),
            "betweenness":   betweenness_c.get(node, 0),
            "closeness":     closeness_c.get(node, 0),
            "eigenvector":   eigenvector_c.get(node, 0),
            "comment_count": G_sub.nodes[node].get("comment_count", 0),
            "avg_score":     G_sub.nodes[node].get("avg_score", 0),
        })

    df = pd.DataFrame(rows).set_index("node")
    print(f"  ✅ Done — {len(df)} nodes in largest component")
    return df


def highlight_central_nodes(df, label, top_n=5):
    print(f"\n{'='*50}")
    print(f"KEY NODES — {label}")
    print(f"{'='*50}")
    for measure in ["degree", "betweenness", "closeness", "eigenvector"]:
        print(f"\n  {measure.upper()} — Top {top_n}:")
        for node, val in df[measure].nlargest(top_n).items():
            print(f"    {str(node):<35} {val:.5f}")
        print(f"  {measure.upper()} — Bottom {top_n}:")
        for node, val in df[measure].nsmallest(top_n).items():
            print(f"    {str(node):<35} {val:.5f}")


# ─────────────────────────────────────────────
# 5. CORRELATIONS
# ─────────────────────────────────────────────

def correlate_centrality(df, label):
    measures = ["degree", "in_degree", "out_degree",
                "betweenness", "closeness", "eigenvector"]
    corr = df[measures].corr()
    print(f"\n{'='*50}")
    print(f"CENTRALITY CORRELATIONS — {label}")
    print(f"{'='*50}")
    print(corr.round(3).to_string())
    return corr


# ─────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_all(df_bots, df_humans, G_bots, G_humans,
             corr_bots, corr_humans):

    BOT_COLOR   = "#e74c3c"
    HUM_COLOR   = "#3498db"
    measures    = ["degree", "betweenness", "closeness", "eigenvector"]

    # ── 6a. Degree distributions ──────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Degree Distribution: Bots vs Humans", fontsize=14, fontweight="bold")
    for ax, m in zip(axes, ["degree", "in_degree", "out_degree"]):
        ax.hist(df_bots[m],   bins=30, alpha=0.6, color=BOT_COLOR,
                label="Bots (SubSimGPT2)", density=True)
        ax.hist(df_humans[m], bins=30, alpha=0.6, color=HUM_COLOR,
                label="Humans (changemyview)", density=True)
        ax.set_title(m.replace("_", " ").title())
        ax.set_xlabel("Centrality Value"); ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_degree_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ 01_degree_distributions.png")

    # ── 6b. Centrality boxplots ───────────────
    df_b = df_bots[measures].copy();   df_b["network"] = "Bots"
    df_h = df_humans[measures].copy(); df_h["network"] = "Humans"
    melted = pd.concat([df_b, df_h]).melt(id_vars="network",
                                           var_name="measure",
                                           value_name="value")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Centrality Measures: Bots vs Humans",
                 fontsize=14, fontweight="bold")
    palette = {"Bots": BOT_COLOR, "Humans": HUM_COLOR}
    for ax, m in zip(axes.flat, measures):
        sns.boxplot(data=melted[melted["measure"] == m],
                    x="network", y="value", palette=palette, ax=ax)
        ax.set_title(m.capitalize()); ax.set_xlabel(""); ax.set_ylabel("Value")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_centrality_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ 02_centrality_boxplots.png")

    # ── 6c. Correlation heatmaps ──────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Centrality Correlation Matrices",
                 fontsize=14, fontweight="bold")
    cols = ["degree","in_degree","out_degree","betweenness","closeness","eigenvector"]
    for ax, corr, title in [
        (ax1, corr_bots,   "Bots — r/SubSimulatorGPT2"),
        (ax2, corr_humans, "Humans — r/changemyview")
    ]:
        sns.heatmap(corr.loc[cols, cols], annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_correlation_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ 03_correlation_heatmaps.png")

    # ── 6d. Network graphs ────────────────────
    for G, df, label, color in [
        (G_bots,   df_bots,   "Bots — r/SubSimulatorGPT2", BOT_COLOR),
        (G_humans, df_humans, "Humans — r/changemyview",    HUM_COLOR),
    ]:
        _plot_network(G, df, label, color)

    # ── 6e. Summary stats bar chart ───────────
    stats = {
        "Density":         [nx.density(G_bots),     nx.density(G_humans)],
        "Reciprocity":     [nx.reciprocity(G_bots), nx.reciprocity(G_humans)],
        "Avg Betweenness": [df_bots["betweenness"].mean(),
                            df_humans["betweenness"].mean()],
        "Avg Closeness":   [df_bots["closeness"].mean(),
                            df_humans["closeness"].mean()],
        "Avg Eigenvector": [df_bots["eigenvector"].mean(),
                            df_humans["eigenvector"].mean()],
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(stats)); w = 0.35
    b1 = ax.bar(x - w/2, [v[0] for v in stats.values()],
                w, label="Bots",   color=BOT_COLOR, alpha=0.85)
    b2 = ax.bar(x + w/2, [v[1] for v in stats.values()],
                w, label="Humans", color=HUM_COLOR, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(list(stats.keys()))
    ax.set_ylabel("Value")
    ax.set_title("Network Summary: Bots vs Humans", fontweight="bold")
    ax.legend()
    ax.bar_label(b1, fmt="%.4f", padding=3, fontsize=7)
    ax.bar_label(b2, fmt="%.4f", padding=3, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_summary_stats.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ 05_summary_stats.png")


def _plot_network(G, df, label, accent_color, max_nodes=120):
    """Force-directed graph: size=degree, color=betweenness."""
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    G_vis = G.subgraph(largest_wcc).copy()
    if G_vis.number_of_nodes() > max_nodes:
        top = sorted(G_vis.nodes(),
                     key=lambda n: G_vis.degree(n), reverse=True)[:max_nodes]
        G_vis = G_vis.subgraph(top).copy()

    pos = nx.spring_layout(G_vis, k=1.2, iterations=60, seed=42)
    nodes_in = [n for n in G_vis.nodes() if n in df.index]
    sizes  = [max(df.loc[n, "degree"] * 4000, 30) for n in nodes_in]
    colors = [df.loc[n, "betweenness"] for n in nodes_in]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#0f0f1a"); fig.patch.set_facecolor("#0f0f1a")

    nx.draw_networkx_edges(G_vis, pos, ax=ax, alpha=0.12,
                           edge_color="#cccccc", arrows=True,
                           arrowsize=6, width=0.4)
    sc = nx.draw_networkx_nodes(G_vis, pos, ax=ax,
                                nodelist=nodes_in,
                                node_size=sizes,
                                node_color=colors,
                                cmap=plt.cm.plasma,
                                alpha=0.9)
    top10 = df.loc[nodes_in].nlargest(10, "degree").index.tolist()
    nx.draw_networkx_labels(G_vis, pos,
                            {n: n for n in top10 if n in pos},
                            ax=ax, font_size=7, font_color="white")

    plt.colorbar(sc, ax=ax, label="Betweenness Centrality",
                 fraction=0.025, pad=0.02)
    ax.set_title(f"{label}\n(size=degree centrality · color=betweenness)",
                 color="white", fontsize=12, pad=12)
    ax.axis("off")

    fname = label.lower().replace(" ", "_").replace("/","").replace("—","")[:40]
    fpath = f"{OUTPUT_DIR}/04_network_{fname}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ 04_network_{fname}.png")


# ─────────────────────────────────────────────
# 7. ALTERNATE SPECIFICATIONS
# ─────────────────────────────────────────────

def alternate_specs(G_bots, G_humans, df_bots, df_humans):
    print(f"\n{'='*50}")
    print("ALTERNATE SPECIFICATIONS")
    print(f"{'='*50}")

    # Spec 1: Remove low-degree nodes (isolates + near-isolates)
    print("\n[Spec 1] Remove nodes with degree < 2 (near-isolates)...")
    for label, G in [("Bots", G_bots), ("Humans", G_humans)]:
        low = [n for n, d in G.degree() if d < 2]
        G2  = G.copy(); G2.remove_nodes_from(low)
        print(f"  {label}: removed {len(low)} nodes → "
              f"{G2.number_of_nodes()} remain, "
              f"density {nx.density(G2):.5f}")

    # Spec 2: Weighted vs unweighted
    print("\n[Spec 2] Edge weight distribution...")
    for label, G in [("Bots", G_bots), ("Humans", G_humans)]:
        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        multi   = sum(w > 1 for w in weights)
        print(f"  {label}: avg weight={np.mean(weights):.2f}, "
              f"max={max(weights)}, "
              f"multi-edges={100*multi/len(weights):.1f}%")

    # Spec 3: Undirected
    print("\n[Spec 3] Treat as undirected...")
    for label, G in [("Bots", G_bots), ("Humans", G_humans)]:
        Gu   = G.to_undirected()
        ccs  = list(nx.connected_components(Gu))
        frac = max(len(c) for c in ccs) / Gu.number_of_nodes()
        print(f"  {label}: {len(ccs)} components, "
              f"largest = {frac:.1%} of nodes, "
              f"density = {nx.density(Gu):.5f}")

    # Spec 4: Normalised centrality comparison
    print("\n[Spec 4] Mean centrality — normalised comparison...")
    print(f"  {'Measure':<15} {'Bots':>10} {'Humans':>10} {'Ratio B/H':>12}")
    print(f"  {'-'*50}")
    for m in ["degree", "betweenness", "closeness", "eigenvector"]:
        b = df_bots[m].mean()
        h = df_humans[m].mean()
        ratio = b / h if h > 0 else float("inf")
        print(f"  {m:<15} {b:>10.5f} {h:>10.5f} {ratio:>12.3f}")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    print("🕸  Dead Internet Theory — Reddit Network Analysis")
    print("=" * 50)

    # ── Collect ───────────────────────────────
    edges_bots,   nodes_bots   = collect_network("SubSimulatorGPT2",
                                                  n_posts=15,
                                                  label="Bots")
    edges_humans, nodes_humans = collect_network("changemyview",
                                                  n_posts=15,
                                                  label="Humans")

    # ── Build ─────────────────────────────────
    G_bots   = build_network(edges_bots,   nodes_bots,
                             "r/SubSimulatorGPT2 (Bots)")
    G_humans = build_network(edges_humans, nodes_humans,
                             "r/changemyview (Humans)")

    describe_network(G_bots,   "r/SubSimulatorGPT2 (Bots)")
    describe_network(G_humans, "r/changemyview (Humans)")

    # ── Centrality ────────────────────────────
    df_bots   = compute_centrality(G_bots,   "Bots")
    df_humans = compute_centrality(G_humans, "Humans")

    highlight_central_nodes(df_bots,   "Bots")
    highlight_central_nodes(df_humans, "Humans")

    # ── Correlations ──────────────────────────
    corr_bots   = correlate_centrality(df_bots,   "Bots")
    corr_humans = correlate_centrality(df_humans, "Humans")

    # ── Visualisations ────────────────────────
    print("\n🎨 Generating visualisations...")
    plot_all(df_bots, df_humans, G_bots, G_humans,
             corr_bots, corr_humans)

    # ── Alternate specs ───────────────────────
    alternate_specs(G_bots, G_humans, df_bots, df_humans)

    # ── Save CSVs ─────────────────────────────
    df_bots.to_csv(f"{OUTPUT_DIR}/centrality_bots.csv")
    df_humans.to_csv(f"{OUTPUT_DIR}/centrality_humans.csv")
    corr_bots.to_csv(f"{OUTPUT_DIR}/corr_bots.csv")
    corr_humans.to_csv(f"{OUTPUT_DIR}/corr_humans.csv")

    print(f"\n✅ All done. Outputs in {OUTPUT_DIR}/")
    print("   Plots : 01_degree_distributions.png")
    print("           02_centrality_boxplots.png")
    print("           03_correlation_heatmaps.png")
    print("           04_network_bots*.png")
    print("           04_network_humans*.png")
    print("           05_summary_stats.png")
    print("   Data  : centrality_bots.csv, centrality_humans.csv")
    print("           corr_bots.csv, corr_humans.csv")


if __name__ == "__main__":
    main()
