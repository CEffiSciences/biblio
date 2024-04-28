import csv
import networkx as nx
from os import environ as env
import pydantic
from services.semantic_scholar import Papers,CitationsByPaper,Citations
import itertools
import copy
import textwrap

class TNSE(pydantic.BaseModel):
    class PaperPoint(pydantic.BaseModel):
        paper_id: str
        cluster_label: int
        x: float
        y: float
        z: float
    points: dict[str, PaperPoint]

###
# Load all data
###
with open(f'./{env.get("DATA_PATH")}/paper_embeddings_clusters.csv') as f:
    reader = csv.DictReader(f)
    tnse = TNSE(points={
        i['Paper ID']: TNSE.PaperPoint(
            paper_id=i['Paper ID'],
            cluster_label=i['Cluster Label'],
            x=i['t-SNE Dim 1'],
            y=i['t-SNE Dim 2'],
            z=i['t-SNE Dim 3']
        ) 
        for i in reader
    })

papers = Papers.model_validate_json(open(f'./{env.get("DATA_PATH")}/papers.json').read()).papers

citations = CitationsByPaper.model_validate_json(open(f'./{env.get("DATA_PATH")}/citations.json').read()).papers


# Filter for citations that are in bioterrorism papers or empty
# citations = dict(filter(lambda pid: pid[0] in papers and len(pid[1].citations) > 0, citations.items()))

# Filter for citations that are in bioterrorism papers
citations = {k: v for k, v in citations.items() for cit in v.citations if cit.citedPaper.paperId in papers}
for k, v in citations.items():
    for cit in v.citations:
        cit.currentPaper = Citations.Citation.PaperRef(paperId=k)

clusters_link = dict()

for src, links in citations.items():
    for link in links.citations:
        trg = link.citedPaper.paperId
        if any(i not in tnse.points for i in [src, trg]):
            continue
        srcClass = tnse.points[src].cluster_label
        trgClass = tnse.points[trg].cluster_label
        if any(i == -1 for i in [srcClass, trgClass]):
            continue
        clusters_link.setdefault((srcClass, trgClass), []).append(link)
# %%
cluster_threat_score = dict()

with open(f'./{env.get("DATA_PATH")}/cluster_problem_scores.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    while True:
        ten_experts = list(sorted(itertools.islice(reader, 10), key=lambda i: float(i['Score'])))
        if not ten_experts:
            break
        cluster = cluster_threat_score.setdefault(int(ten_experts[0]['Cluster']), dict())
        cluster[ten_experts[0]['Threat Type']] = (float(ten_experts[4]['Score']) + float(ten_experts[5]['Score'])) / 2

# %%
def keep(edge):
    if edge[0] == edge[1]:
        return False

    dat = clusters_link.get(edge)
    if not dat:
        return False

    return any(i.isInfluential for i in dat) or len({i.citedPaper.paperId for i in dat}) >= 3

with open(f'./{env.get("DATA_PATH")}/cluster_labels.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    clusters = {int(i['Cluster_ID']): '\n'.join(textwrap.wrap(i['Cluster_Name'], width=20)) for i in reader}

clusters_num = clusters.keys()
clusters_num = [src for src in clusters_num if any(keep((src, trg)) for trg in clusters_num if src != trg)]

cluster_graph = nx.DiGraph()
cluster_graph.add_nodes_from(clusters_num)

# https://coolors.co/0081a7-00afb9-fdfcdc-fed9b7-f07167
color_palet = '0081a7-00afb9-fdfcdc-fed9b7-f07167'
colors_dict = {
    "Viral": '#' + color_palet.split('-')[0], # viral | blue - cerulean
    "Bacterial": '#' + color_palet.split('-')[1], # Bacterial | light blue - verdigris
    "Toxin": '#' + color_palet.split('-')[2], # Toxin | light yellow
    "Fungal": '#' + color_palet.split('-')[3], # Fungal | light orange
    "Prion": '#' + color_palet.split('-')[4], # Prion | red - bittersweet
}

for (src, trg), dat in clusters_link.items():
    allSrcs = {i.currentPaper.paperId for i in dat}
    allTrgs = {i.citedPaper.paperId for i in dat}
    if src in clusters_num and trg in clusters_num and keep((src, trg)):
        cluster_graph.add_edge(src, trg, data=dat)

def get_attr(edge):
    dat = clusters_link[edge]
    allTrgs = {i.citedPaper.paperId for i in dat}

    influential = any(i.isInfluential for i in dat)
    alpha = 1 if keep(edge) else 0

    return dict(
        penwidth=min(len(allTrgs), 8) / 2 if not influential else 8,
        constraint=keep(edge),
        color='black' if alpha > 0.1 else 'transparent',
    )

draw_graph = cluster_graph
nx.set_edge_attributes(draw_graph, {edge: get_attr(edge) for edge in draw_graph.edges})

# def get_attr_pc(cluster, cat):
#     score = cluster_threat_score[cluster][cat]

#     assert colors_dict.get(cat) is not None or score < 0.85

#     return dict(
#         style = 'invis' if score < 0.85 else 'dashed',
#         color=colors_dict.get(cat),
#         constraint=True,
#         weight=1000 if score > 0.85 else 1,
#     )

def get_colors(cluster):
    colors =[colors_dict[k] for k, score in cluster_threat_score[cluster].items() if score > 0.75]
    return dict(
        fillcolor=':'.join(colors),
        style='wedged' if len(colors) > 1 else 'filled',
    )

nx.set_node_attributes(draw_graph, {cluster: get_colors(cluster) for cluster in cluster_threat_score})

draw_graph = draw_graph.subgraph(clusters_num)
component = max(nx.strongly_connected_components(draw_graph), key=len)
draw_graph = draw_graph.subgraph(component)
draw_graph = nx.relabel_nodes(cluster_graph, clusters)

graphviz = nx.nx_agraph.to_agraph(draw_graph)
graphviz.graph_attr.update()
graphviz.node_attr.update(fontsize=40, fontfamily='FreeMono', fontweight="bold")
graphviz.edge_attr.update(dir='back')
graphviz.draw(f'./{env.get("DATA_PATH")}/cluster_graph.png', prog='dot')