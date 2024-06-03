import hashlib
from typing import IO
from .services.semantic_scholar import ReferencesByPaper, Papers
from .lang import Translations
from .clusters import Clusters
from .threat import ThreatScores

import pathlib

from tqdm import tqdm
import click

from plotly import graph_objects as go
import matplotlib.colors
import networkx as nx


from .services import semantic_scholar
from .env import env

from . import lang
from . import clusters
from . import threat
from . import graph


@click.group(context_settings={"show_default": True})
def cli():
    pass


@cli.command()
@click.option("--query", default="bioterrorism")
@click.option("--limit", default=100)
@click.option("--out", default=env.path.papers, type=click.File("w"))
def fetch_papers(*, query: str, limit: int, out: IO[str]):
    pathlib.Path(out.name).parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"Fetching papers for {query}...")

    papers = semantic_scholar.fetch_papers(
        query=query,
        limit=limit,
    )

    click.echo(f"Writing papers to {out.name}")
    out.write(papers.model_dump_json(indent=2))


@cli.command()
@click.option("--papers_input", default=env.path.papers, type=click.File("r"))
@click.option("--out", default=env.path.translations, type=click.File("w"))
def generate_translations(*, papers_input: IO[str], out: IO[str]):
    pathlib.Path(out.name).parent.mkdir(parents=True, exist_ok=True)

    papers = Papers.model_validate_json(papers_input.read())

    translations = Translations(
        translations={
            name: Translations.Translation(abstract=paper.abstract, title=paper.title)
            for name, paper in papers.papers.items()
            if paper.abstract
        }
        | {
            name: lang.translate_paper_to_en(paper)
            for name, paper in tqdm(
                [
                    (k, v)
                    for k, v in papers.papers.items()
                    if v.abstract and not lang.is_text_in_language(v.abstract, "en")
                ],
                desc="Translating abstracts",
            )
            if paper.abstract
            if not lang.is_text_in_language(paper.abstract, "en")
        }
    )

    click.echo(f"Writing translations to {out.name}")
    out.write(translations.model_dump_json(indent=2))


@cli.command()
@click.option("--papers_input", default=env.path.papers, type=click.File("r"))
@click.option(
    "--cache_folder",
    default=env.path.cache_folder,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        path_type=pathlib.Path,
    ),
)
@click.option("--cache/--no-cache", default=True)
@click.option("--limit", default=100)
@click.option("--out", default=env.path.references, type=click.File("w"))
def fetch_references(
    *,
    papers_input: IO[str],
    cache_folder: pathlib.Path,
    cache: bool,
    limit: int,
    out: IO[str],
):
    pathlib.Path(out.name).parent.mkdir(parents=True, exist_ok=True)
    papers = Papers.model_validate_json(papers_input.read())

    cache_folder.mkdir(parents=True, exist_ok=True)
    cache_file = (
        cache_folder / hashlib.md5(papers.model_dump_json().encode()).hexdigest()
    )
    references = (
        ReferencesByPaper.model_validate_json(cache_file.read_text())
        if cache_file.exists() and cache
        else ReferencesByPaper(papers={})
    )

    for name, paper in tqdm(papers.papers.items(), desc="Fetching references"):
        if name in references.papers:
            continue

        references.papers[name] = semantic_scholar.fetch_references(
            paper_id=paper.paperId,
            limit=limit,
        )
        cache_file.write_text(references.model_dump_json(indent=2))

    click.echo(f"Writing references to {out.name}")
    out.write(references.model_dump_json(indent=2))


@cli.command()
@click.option("--papers_input", default=env.path.papers, type=click.File("r"))
@click.option(
    "--translations_input", default=env.path.translations, type=click.File("r")
)
@click.option("--out_json", default=env.path.clusters, type=click.File("w"))
@click.option("--out_html", default=env.path.clusters_html, type=click.File("w"))
@click.option("--n_dims", default=3)
@click.option("--min_sample_size", default=10)
@click.option("--min_samples", default=7)
def generate_clusters(
    *,
    papers_input: IO[str],
    translations_input: IO[str],
    out_json: IO[str],
    out_html: IO[str],
    min_sample_size: int,
    min_samples: int,
    n_dims: int,
):
    pathlib.Path(out_json.name).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_html.name).parent.mkdir(parents=True, exist_ok=True)

    papers = Papers.model_validate_json(papers_input.read())
    translations = Translations.model_validate_json(translations_input.read())

    clusters_papers = clusters.cluster_papers(
        papers,
        translations,
        min_cluster_size=min_sample_size,
        min_samples=min_samples,
        n_dims=n_dims,
    )

    click.echo(f"Writing clusters to {out_json.name}")
    out_json.write(clusters_papers.model_dump_json(indent=2))

    click.echo(f"Writing interactive clusters to {out_html.name}")

    fig = go.Figure()
    colors = iter(matplotlib.colors.cnames.values())
    for cluster in clusters_papers.clusters.values():
        if cluster.index == -1:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=[i.tsne[0] for i in cluster.points],
                y=[i.tsne[1] for i in cluster.points],
                z=[i.tsne[2] for i in cluster.points],
                mode="markers",
                marker=dict(
                    size=5,
                    line=dict(width=0.5),
                    opacity=0.8,
                    color=next(colors),
                ),
                name=cluster.name,
            )
        )
    fig.write_html(out_html)


@cli.command()
@click.option("--clusters_input", default=env.path.clusters, type=click.File("r"))
@click.option("--out", default=env.path.threat_scores, type=click.File("w"))
def generate_threat_scores(
    *,
    clusters_input: IO[str],
    out: IO[str],
):
    pathlib.Path(out.name).parent.mkdir(parents=True, exist_ok=True)
    clusters = Clusters.model_validate_json(clusters_input.read())

    scores = ThreatScores(
        threat_scores={
            i: threat.threat_score(j.name)
            for i, j in tqdm(clusters.clusters.items(), desc="Evaluating threat scores")
        }
    )

    out.write(scores.model_dump_json(indent=2))


@cli.command()
@click.option("--papers_input", default=env.path.papers, type=click.File("r"))
@click.option("--references_input", default=env.path.references, type=click.File("r"))
@click.option("--clusters_input", default=env.path.clusters, type=click.File("r"))
@click.option("--threat_input", default=env.path.threat_scores, type=click.File("r"))
@click.option("--threshold_threat_color", default=0.75)
@click.option(
    "--remove_isolated_nodes/--no-remove_isolated_nodes", default=True, is_flag=True
)
@click.option("--influence_weight", default=1.0 / 8)
@click.option("--out", default=env.path.graph, type=click.File("wb"))
def generate_graph(
    *,
    papers_input: IO[str],
    references_input: IO[str],
    clusters_input: IO[str],
    threat_input: IO[str],
    threshold_threat_color: float,
    remove_isolated_nodes: bool,
    influence_weight: float,
    out: IO[bytes],
):
    pathlib.Path(out.name).parent.mkdir(parents=True, exist_ok=True)

    papers = Papers.model_validate_json(papers_input.read())
    references = ReferencesByPaper.model_validate_json(references_input.read())
    clusters = Clusters.model_validate_json(clusters_input.read())
    threat = ThreatScores.model_validate_json(threat_input.read())

    click.echo("Generating graph...")
    todraw = graph.generate_graph(
        papers=papers,
        references_by_papers=references,
        clusters=clusters,
        threats=threat,
        threshold_threat_color=threshold_threat_color,
        remove_isolated_nodes=remove_isolated_nodes,
        influence_weight=influence_weight,
    )
    graphviz = nx.nx_agraph.to_agraph(todraw)

    graphviz.graph_attr.update()
    graphviz.node_attr.update(fontsize=40, fontfamily="FreeMono", fontweight="bold")
    graphviz.edge_attr.update(dir="back")
    click.echo(f"Writing graph to {out.name}")
    graphviz.draw(out, prog="dot")


@cli.command()
@click.pass_context
def pipeline(ctx: click.Context):
    ctx.invoke(fetch_papers)
    ctx.invoke(generate_translations)
    ctx.invoke(generate_clusters)
    ctx.invoke(generate_threat_scores)
    ctx.invoke(fetch_references)
    ctx.invoke(generate_graph)
