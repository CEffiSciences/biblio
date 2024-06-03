import textwrap
from typing import Any, Callable
from .services.semantic_scholar import PaperID, References, ReferencesByPaper, Papers
from .threat import ThreatScores

import networkx as nx

from .clusters import Clusters, ClusterID


def median(lst: list[float]) -> float:
    lst = sorted(lst)
    n = len(lst)
    if n % 2 == 1:
        return lst[n // 2]
    else:
        return (lst[n // 2 - 1] + lst[n // 2]) / 2


def filter_references_inside_papers(
    papers: Papers, references: ReferencesByPaper
) -> ReferencesByPaper:
    """
    Filter references to only be between between an element of papers and another element of papers
    """
    return ReferencesByPaper(
        papers={
            k: References(
                references=[
                    cit
                    for cit in v.references
                    if cit.citedPaper and cit.citedPaper.paperId in papers.papers
                ]
            )
            for k, v in references.papers.items()
        }
    )


def get_clusters_by_paper(clusters: Clusters) -> dict[PaperID, ClusterID]:
    result: dict[PaperID, ClusterID] = {}
    for cluster in clusters.clusters.values():
        for point in cluster.points:
            result[point.paper_id] = cluster.index

    return result


def get_references_between_clusters(
    references: ReferencesByPaper, clusters_by_paper: dict[PaperID, ClusterID]
):
    result: dict[tuple[ClusterID, ClusterID], list[References.Reference]] = {}

    for src, links in references.papers.items():
        for link in links.references:
            assert link.citedPaper and link.citedPaper.paperId

            trg = link.citedPaper.paperId

            if src not in clusters_by_paper or trg not in clusters_by_paper:
                continue

            srcClass = clusters_by_paper[src]
            trgClass = clusters_by_paper[trg]

            if srcClass == -1 or trgClass == -1:
                continue

            result.setdefault((srcClass, trgClass), []).append(link)

    return result


def default_keep(references: list[References.Reference]):
    return (
        any(i.isInfluential for i in references)
        or len({i.citedPaper.paperId for i in references if i.citedPaper}) >= 3
    )


default_palette = {
    "viral": "red",
    "bacterial": "green",
    "toxin": "blue",
    "fungal": "yellow",
    "prion": "purple",
}


def default_width_line(
    references: list[References.Reference], influential_width_line: int = 8
):
    allTrgs = {i.citedPaper.paperId for i in references if i.citedPaper}

    influential = any(i.isInfluential for i in references)
    return (
        min(len(allTrgs), influential_width_line) / 2
        if not influential
        else influential_width_line
    )


def generate_graph(
    references_by_papers: ReferencesByPaper,
    papers: Papers,
    clusters: Clusters,
    threats: ThreatScores,
    keep_function: Callable[[list[References.Reference]], bool] = default_keep,
    width_line_function: Callable[
        [list[References.Reference]], float
    ] = default_width_line,
    palette: dict[str, str] = default_palette,
    threshold_threat_color: float = 0.75,
    remove_isolated_nodes: bool = True,
    influence_weight: float | None = 1.0 / 8,
) -> nx.DiGraph:
    references_by_papers = filter_references_inside_papers(papers, references_by_papers)
    clusters_by_papers = get_clusters_by_paper(clusters)
    references_between_clusters = get_references_between_clusters(
        references_by_papers, clusters_by_papers
    )

    def keep(edge: tuple[ClusterID, ClusterID]) -> bool:
        if edge[0] == edge[1]:
            return False

        references = references_between_clusters.get(edge)
        if not references:
            return False

        return keep_function(references)

    clusters_num = clusters.clusters.keys()
    clusters_num = [
        src
        for src in clusters_num
        if src != -1
        and not remove_isolated_nodes
        or any(keep((src, trg)) for trg in clusters_num if src != trg)
    ]

    graph = nx.DiGraph()
    graph.add_nodes_from(clusters_num)

    for (src, trg), references in references_between_clusters.items():
        if src in clusters_num and trg in clusters_num and keep((src, trg)):
            graph.add_edge(src, trg, data=references)

    def get_edge_attrs(edge: tuple[ClusterID, ClusterID]) -> dict[str, Any]:
        return dict(
            penwidth=width_line_function(references_between_clusters[edge]),
            weight=width_line_function(references_between_clusters[edge])
            * influence_weight
            if influence_weight
            else 1.0,
        )

    nx.set_edge_attributes(graph, {edge: get_edge_attrs(edge) for edge in graph.edges})

    def get_node_attrs(cluster: ClusterID) -> dict[str, Any]:
        colors = [
            palette[k]
            for k in threats.threat_scores[cluster].model_fields
            if median(getattr(threats.threat_scores[cluster], k))
            > threshold_threat_color
        ]

        return dict(
            fillcolor=":".join(colors),
            style="wedged" if len(colors) > 1 else "filled",
        )

    nx.set_node_attributes(
        graph, {cluster: get_node_attrs(cluster) for cluster in clusters_num}
    )

    graph = nx.relabel_nodes(
        graph,
        {
            i: "\n".join(textwrap.wrap(clusters.clusters[i].name, width=20))
            for i in graph.nodes
        },
    )

    return graph
