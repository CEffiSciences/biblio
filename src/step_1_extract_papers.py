from services import semantic_scholar
from os import environ as env

papers = semantic_scholar.fetch_papers(
    query='bioterrorims',
    limit=1
)

semantic_scholar.dump(papers, f'{env.get("DATA_PATH")}papers.json')