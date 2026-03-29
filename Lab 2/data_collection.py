import io
import contextlib
import json
import requests
import time
import pandas as pd
import networkx as nx
from collections import defaultdict
from config import HEADERS, OUTPUT_DIR

IRA_COMMENTS_URL = "https://raw.githubusercontent.com/ALCC01/reddit-suspicious-accounts/master/data/comments.csv"


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
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching posts: {e}")
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
    """
    Recursively walk comment tree, extract user→user reply edges.
    Reddit comment trees can be deeply nested.
    """
    if edges is None:
        edges = []
    if node_data is None:
        node_data = {}

    for item in comments:
        if item["kind"] != "t1":  # t1 = comment
            continue

        data = item["data"]
        author = data.get("author", "[deleted]")

        if author in ("[deleted]", "AutoModerator", None):
            continue

        if author not in node_data:
            node_data[author] = {"comment_count": 0, "total_score": 0}
        node_data[author]["comment_count"] += 1
        node_data[author]["total_score"] += data.get("score", 0)

        if parent_author and parent_author != author:
            edges.append({
                "source": author,
                "target": parent_author,
                "score": data.get("score", 0)
            })

        replies = data.get("replies", "")
        if isinstance(replies, dict):
            reply_comments = replies["data"]["children"]
            parse_comment_tree(reply_comments, author, edges, node_data)

    return edges, node_data


def collect_network(subreddit, n_posts=15, label=""):
    """Full collection pipeline for one subreddit."""
    print(f"\n{'='*50}")
    print(f"Collecting r/{subreddit} ({label})")
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

        for user, stats in nodes.items():
            if user not in all_nodes:
                all_nodes[user] = {"comment_count": 0, "total_score": 0}
            all_nodes[user]["comment_count"] += stats["comment_count"]
            all_nodes[user]["total_score"]   += stats["total_score"]

        all_edges.extend(edges)
        time.sleep(1.5)

    print(f"\n  Total: {len(all_edges)} edges, {len(all_nodes)} unique users")
    return all_edges, all_nodes


def collect_ira_network():
    """
    Build edge/node data from the archived IRA Reddit comment dataset.
    Edges are constructed the same way as parse_comment_tree:
      - top-level comment (parent_id starts with t3_) → edge to post author (link_author)
      - reply to another comment (parent_id starts with t1_) → look up parent comment
        author within the dataset; skip if parent not in dataset (non-IRA user)
    This gives us IRA-bot→IRA-bot edges, capturing how they amplified each other.
    """
    print(f"\n{'='*50}")
    print("Collecting IRA bot network (archived CSV)")
    print(f"{'='*50}")

    df = pd.read_csv(IRA_COMMENTS_URL)
    df = df[df['author.name'].notna()]
    df = df[~df['author.name'].isin(['[deleted]', 'AutoModerator'])]

    print(f"  Loaded {len(df)} comments from {df['author.name'].nunique()} IRA accounts")

    # index by fullname for fast parent lookup
    id_to_author = dict(zip(df['fullname'], df['author.name']))

    edges    = []
    node_data = {}

    for _, row in df.iterrows():
        author    = row['author.name']
        parent_id = str(row['parent_id'])
        score     = row.get('score', 0)

        # track node
        if author not in node_data:
            node_data[author] = {"comment_count": 0, "total_score": 0}
        node_data[author]["comment_count"] += 1
        node_data[author]["total_score"]   += score

        # resolve parent author
        if parent_id.startswith("t3_"):
            # top-level comment — parent is the post author
            parent_author = row.get('link_author', None)
        elif parent_id.startswith("t1_"):
            # reply — look up parent comment in dataset
            parent_author = id_to_author.get(parent_id, None)
        else:
            parent_author = None

        if parent_author and parent_author not in ('[deleted]', 'AutoModerator') \
                and parent_author != author:
            edges.append({
                "source": author,
                "target": parent_author,
                "score":  score
            })

    print(f"  Total: {len(edges)} edges, {len(node_data)} unique IRA users")
    return edges, node_data


def summarise(edges, node_data, label):
    edge_weights = defaultdict(int)
    for e in edges:
        edge_weights[(e["source"], e["target"])] += 1
    G = nx.DiGraph()
    for (src, tgt), w in edge_weights.items():
        G.add_edge(src, tgt, weight=w)

    # === Basic Network Properties ===
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density   = nx.density(G)

    print(f"\n{'='*50}")
    print(f"NETWORK SUMMARY — {label}")
    print(f"{'='*50}")
    print(f"Number of Nodes: {num_nodes}")
    print(f"Number of Edges: {num_edges}")
    print(f"Network Density: {density:.4f}")
    try:
        print(f"Reciprocity    : {nx.reciprocity(G):.4f}")
    except Exception:
        pass

    # === Attribute Data Overview ===
    df_att = pd.DataFrame.from_dict(node_data, orient="index")
    df_att.index.name = "user"
    df_att["avg_score"]   = df_att["total_score"] / df_att["comment_count"].clip(lower=1)
    df_att["in_degree"]   = pd.Series(dict(G.in_degree()))
    df_att["out_degree"]  = pd.Series(dict(G.out_degree()))

    print(f"\n=== Attribute Data Overview — {label} ===")
    print(df_att[["comment_count", "avg_score", "in_degree", "out_degree"]].describe().round(3).to_string())


if __name__ == "__main__":
    edges_bots,   nodes_bots   = collect_network("SubSimulatorGPT2", n_posts=15, label="Bots")
    edges_humans, nodes_humans = collect_network("changemyview",      n_posts=15, label="Humans")
    edges_ira,    nodes_ira    = collect_ira_network()

    # print to console AND write to txt
    summary_path = f"{OUTPUT_DIR}/network_summary.txt"
    all_output = []
    for edges, nodes, label in [
        (edges_bots,   nodes_bots,   "r/SubSimulatorGPT2 (Bots)"),
        (edges_humans, nodes_humans, "r/changemyview (Humans)"),
        (edges_ira,    nodes_ira,    "IRA Bots"),
    ]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summarise(edges, nodes, label)
        output = buf.getvalue()
        print(output, end="")
        all_output.append(output)
    with open(summary_path, "w") as f:
        f.write("".join(all_output))

    # save per-network attribute CSVs
    for edges, nodes, label, fname in [
        (edges_bots,   nodes_bots,   "r/SubSimulatorGPT2 (Bots)", "summary_bots.csv"),
        (edges_humans, nodes_humans, "r/changemyview (Humans)",    "summary_humans.csv"),
        (edges_ira,    nodes_ira,    "IRA Bots",                   "summary_ira.csv"),
    ]:
        ew = defaultdict(int)
        for e in edges: ew[(e["source"], e["target"])] += 1
        G = nx.DiGraph()
        for (s, t), w in ew.items(): G.add_edge(s, t, weight=w)
        df_csv = pd.DataFrame.from_dict(nodes, orient="index")
        df_csv.index.name = "user"
        df_csv["avg_score"]  = df_csv["total_score"] / df_csv["comment_count"].clip(lower=1)
        df_csv["in_degree"]  = pd.Series(dict(G.in_degree()))
        df_csv["out_degree"] = pd.Series(dict(G.out_degree()))
        df_csv.to_csv(f"{OUTPUT_DIR}/{fname}")

    # cache raw data for downstream modules
    with open(f"{OUTPUT_DIR}/raw_edges_bots.json",   "w") as f: json.dump(edges_bots,   f)
    with open(f"{OUTPUT_DIR}/raw_nodes_bots.json",   "w") as f: json.dump(nodes_bots,   f)
    with open(f"{OUTPUT_DIR}/raw_edges_humans.json", "w") as f: json.dump(edges_humans, f)
    with open(f"{OUTPUT_DIR}/raw_nodes_humans.json", "w") as f: json.dump(nodes_humans, f)
    with open(f"{OUTPUT_DIR}/raw_edges_ira.json",    "w") as f: json.dump(edges_ira,    f)
    with open(f"{OUTPUT_DIR}/raw_nodes_ira.json",    "w") as f: json.dump(nodes_ira,    f)
    print(f"\nSummary saved to {OUTPUT_DIR}/network_summary.txt")
    print(f"CSVs saved to    {OUTPUT_DIR}/summary_bots.csv, summary_humans.csv, summary_ira.csv")
