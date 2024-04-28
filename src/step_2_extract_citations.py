from services import semantic_scholar
from os import environ as env
import json
from tqdm import tqdm

papers = json.load(open(f'{env.get("DATA_PATH")}/papers.json'))

# citations = {}

# for paper, idx in papers:
#     citations[idx] = semantic_scholar.fetch_citations(
#         paper_id=paper['paperId']
#     )

citations = []

for paper, idx in papers:
    citations.append(semantic_scholar.fetch_citations(
        paper_id=paper['paperId']
    ))
    
semantic_scholar.dump(citations, f'{env.get("DATA_PATH")}/citations.json')
