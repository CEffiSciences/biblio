import pathlib
import pydantic
import environs

envf = environs.Env()
envf.read_env()
envf.read_env(".env.local")


class Env(pydantic.BaseModel):
    semantic_api_key: str = envf.str("SEMANTIC_SCHOLAR_API_KEY")

    class Path(pydantic.BaseModel):
        data: pathlib.Path = pathlib.Path(envf.str("DATA_PATH"))
        papers: pathlib.Path = data / "papers.json"
        translations: pathlib.Path = data / "translations.json"
        cache_folder: pathlib.Path = data
        references: pathlib.Path = data / "references.json"
        clusters: pathlib.Path = data / "clusters.json"
        clusters_html: pathlib.Path = data / "clusters.html"
        threat_scores: pathlib.Path = data / "threat_scores.json"
        graph: pathlib.Path = data / "graph.png"

    path: Path = Path()


env = Env()
env.path.data.mkdir(parents=True, exist_ok=True)
