from .services.semantic_scholar import PaperID, Papers
from .lang import Translations

from collections import defaultdict
import click
from tqdm import tqdm
from .services.openai import client

import pydantic

from sklearn.manifold import TSNE
import hdbscan

from sentence_transformers import SentenceTransformer


ClusterID = int


class Clusters(pydantic.BaseModel):
    class Cluster(pydantic.BaseModel):
        class Point(pydantic.BaseModel):
            paper_id: PaperID
            embedding: tuple[float, ...]
            tsne: tuple[float, float, float]
            index: int

        index: int
        points: list[Point] = []
        name: str

    clusters: dict[ClusterID, Cluster] = {}


def generate_cluster_title(
    translations: Translations,
    cluster: Clusters.Cluster,
    limit: int = 20,
) -> str:
    def get_title(paper_id: PaperID) -> str:
        return translations.translations[paper_id].title

    titles = "\n".join(get_title(point.paper_id) for point in cluster.points[:limit])
    prompt = f"""
        Based on the following titles of research papers, generate a concise and informative title for a research axis that encapsulates the common theme. The title should be succinct, informative, and consist of 3 to 10 words. Do not use a colon (:). Here are some examples of good research axis titles:

        - Integrated Syndromic Surveillance for Enhanced Public Health Preparedness
        - Innovative Strategies for Rapid Vaccine Development Against Bioterrorism Agents
        - Advances Detection, Prevention, and Vaccine Development for Ebola and Other Hemorrhagic Fever Viruses
        - Evaluation of Biorisk Threats in Food and Waterborne Pathogens

        Given these titles of research papers in the cluster:
        {titles}

        Generate a research axis title:
        """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
        temperature=0.0,
        stop=["\n"],
        logit_bias={"25": -20, "1058": -20},
    )

    assert response.choices[0].message.content
    return response.choices[0].message.content


def cluster_papers(
    papers: Papers,
    translations: Translations,
    n_dims: int = 3,
    min_cluster_size: int = 10,
    min_samples: int = 7,
) -> Clusters:
    papers = Papers(
        papers={
            i: j for i, j in papers.papers.items() if i in translations.translations
        }
    )

    def get_abstract(paper: Papers.Paper) -> str:
        return translations.translations[paper.paperId].abstract

    abstracts = [get_abstract(paper) for paper in papers.papers.values()]

    click.echo("Computing embeddings of abstracts")
    embedder = SentenceTransformer("all-mpnet-base-v2")
    embeddings = embedder.encode(
        abstracts,
        show_progress_bar=True,
    )

    click.echo("Computing t-SNE of embeddings")
    tsne = TSNE(n_components=n_dims, random_state=0, verbose=1)
    embeddings_tSNE = tsne.fit_transform(embeddings)  # type: ignore

    click.echo("Computing hdbscan of t-SNE embeddings")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    labels = clusterer.fit_predict(embeddings_tSNE)

    result_raw = defaultdict(list[Clusters.Cluster.Point])

    for index, (label, paper_id, embedding, tsne) in list(
        enumerate(zip(labels, papers.papers, embeddings, embeddings_tSNE))
    ):
        result_raw[int(label)].append(
            Clusters.Cluster.Point(
                paper_id=paper_id,
                embedding=embedding.tolist(),  # type: ignore
                tsne=tsne.tolist(),
                index=index,
            )
        )

    result = Clusters()
    for label, cluster in tqdm(
        list(result_raw.items()), desc="Generating cluster titles"
    ):
        cluster = Clusters.Cluster(points=cluster, name="", index=label)
        cluster.name = (
            generate_cluster_title(translations, cluster) if label != -1 else "Noise"
        )

        result.clusters[label] = cluster

    return result
