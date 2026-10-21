"""Network Analysis: PageRank, Betweenness Centrality, and Narrative Propagation."""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("NetworkAnalysis")

# KOL = Key Opinion Leader. Users scoring above this PageRank threshold are flagged.
KOL_PAGERANK_THRESHOLD = 0.05
KOL_BETWEENNESS_THRESHOLD = 0.05


class NetworkAnalyzer:
    """
    Graph-based influence and propagation analysis engine.

    Builds a directed weighted social graph from interaction edges
    (sourced from Neo4j or synthetic data) and computes:
    - PageRank: node influence / reach
    - Betweenness Centrality: information bottleneck / bridge role
    - Key Opinion Leader (KOL) classification
    - Narrative propagation path tracing
    """

    def __init__(self) -> None:
        self._graph: Optional[nx.DiGraph] = None

    def build_graph(self, edges: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build a directed weighted graph from a list of edge dicts.

        Expected edge format:
            {"source": str, "target": str, "weight": float}
        """
        G = nx.DiGraph()
        for e in edges:
            src = e.get("source", "")
            tgt = e.get("target", "")
            wt = float(e.get("weight", 1.0))
            if src and tgt:
                if G.has_edge(src, tgt):
                    G[src][tgt]["weight"] += wt
                else:
                    G.add_edge(src, tgt, weight=wt)
        self._graph = G
        logger.info(f"Social graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return G

    def compute_pagerank(
        self,
        G: Optional[nx.DiGraph] = None,
        alpha: float = 0.85,
        max_iter: int = 100,
    ) -> Dict[str, float]:
        """
        Compute PageRank scores for all nodes.

        Returns:
            dict mapping hashed_user_id -> PageRank score (float)
        """
        G = G or self._graph
        if G is None or G.number_of_nodes() == 0:
            logger.warning("Empty graph — no PageRank computed.")
            return {}

        try:
            scores = nx.pagerank(G, alpha=alpha, max_iter=max_iter, weight="weight")
            logger.info(f"PageRank computed for {len(scores)} nodes.")
            return scores
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank did not converge. Returning uniform scores.")
            n = G.number_of_nodes()
            return {node: 1.0 / n for node in G.nodes()}

    def compute_betweenness_centrality(
        self, G: Optional[nx.DiGraph] = None, normalized: bool = True
    ) -> Dict[str, float]:
        """
        Compute Betweenness Centrality for all nodes.

        Returns:
            dict mapping hashed_user_id -> betweenness score (float)
        """
        G = G or self._graph
        if G is None or G.number_of_nodes() == 0:
            logger.warning("Empty graph — no betweenness centrality computed.")
            return {}

        scores = nx.betweenness_centrality(G, normalized=normalized, weight="weight")
        logger.info(f"Betweenness Centrality computed for {len(scores)} nodes.")
        return scores

    def identify_kols(
        self,
        pagerank_scores: Dict[str, float],
        betweenness_scores: Dict[str, float],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Identify Key Opinion Leaders (KOLs) based on combined influence.

        Ranking = 0.6 * PageRank + 0.4 * Betweenness (normalized to max).
        """
        if not pagerank_scores:
            return []

        all_nodes = set(pagerank_scores) | set(betweenness_scores)

        # Normalize each score to [0, 1]
        max_pr = max(pagerank_scores.values(), default=1.0) or 1.0
        max_bt = max(betweenness_scores.values(), default=1.0) or 1.0

        kol_scores = []
        for node in all_nodes:
            pr = pagerank_scores.get(node, 0.0)
            bt = betweenness_scores.get(node, 0.0)
            combined = 0.6 * (pr / max_pr) + 0.4 * (bt / max_bt)
            kol_scores.append({
                "hashed_user_id": node,
                "pagerank": pr,
                "betweenness": bt,
                "influence_score": round(combined, 6),
                "is_kol": (pr >= KOL_PAGERANK_THRESHOLD or bt >= KOL_BETWEENNESS_THRESHOLD),
            })

        kol_scores.sort(key=lambda x: x["influence_score"], reverse=True)
        logger.info(f"KOL identification: {sum(1 for k in kol_scores if k['is_kol'])} KOLs among {len(kol_scores)} users.")
        return kol_scores[:top_n]

    def trace_narrative_propagation(
        self,
        G: Optional[nx.DiGraph],
        source_user_id: str,
        max_hops: int = 5,
    ) -> Dict[str, Any]:
        """
        Trace narrative propagation paths from a source user node.

        Returns:
            dict with:
            - source: the origin user
            - reachable_nodes: all nodes reachable within max_hops
            - propagation_paths: shortest paths from source to each reachable node
            - clusters: weakly connected components for cluster analysis
        """
        G = G or self._graph
        if G is None:
            return {"source": source_user_id, "reachable_nodes": [], "propagation_paths": {}, "clusters": []}

        if source_user_id not in G:
            logger.warning(f"Source node '{source_user_id}' not found in graph.")
            return {"source": source_user_id, "reachable_nodes": [], "propagation_paths": {}, "clusters": []}

        # BFS to find all nodes reachable within max_hops
        reachable = {}
        for target in nx.single_source_shortest_path_length(G, source_user_id, cutoff=max_hops):
            if target != source_user_id:
                reachable[target] = nx.shortest_path(G, source_user_id, target)

        # Weakly connected components (clusters)
        wcc = [list(c) for c in nx.weakly_connected_components(G)]

        logger.info(
            f"Propagation trace from '{source_user_id[:8]}...': "
            f"{len(reachable)} reachable nodes within {max_hops} hops."
        )
        return {
            "source": source_user_id,
            "reachable_nodes": list(reachable.keys()),
            "propagation_paths": reachable,
            "clusters": wcc,
            "hop_count": max_hops,
        }

    def run_full_analysis(
        self,
        edges: List[Dict[str, Any]],
        top_n_kols: int = 10,
    ) -> Dict[str, Any]:
        """
        Convenience method: build graph, compute all metrics, return summary.
        """
        G = self.build_graph(edges)
        pagerank = self.compute_pagerank(G)
        betweenness = self.compute_betweenness_centrality(G)
        kols = self.identify_kols(pagerank, betweenness, top_n=top_n_kols)

        return {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "pagerank": pagerank,
            "betweenness": betweenness,
            "kols": kols,
        }
