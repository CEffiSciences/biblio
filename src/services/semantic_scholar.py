import pydantic
from semanticscholar import SemanticScholar,PaginatedResults
import json

try:
    sch = SemanticScholar(api_key=open('.semantic_key').read().strip())
except:
    raise Exception('Provide ".semantic_key" with your api key credentials')

class Papers(pydantic.BaseModel):
    class Paper(pydantic.BaseModel):
        paperId: str
        title: str
        referenceCount: int
        citationCount: int
        influentialCitationCount: int
        fieldsOfStudy: list | None
        s2FieldsOfStudy: list
        publicationTypes: list[str] | None

        class Journal(pydantic.BaseModel):
            name: str | None = None
            volume: str | None = None
        journal: Journal | None
        
    papers: dict[str, Paper]

PAPER_FIELDS = Papers.Paper.model_fields.keys()

class Citations(pydantic.BaseModel):
    class Citation(pydantic.BaseModel):
        class PaperRef(pydantic.BaseModel):
            paperId: str | None = None
        paperId: str | None = None
        intents: list[str]
        isInfluential: bool
        currentPaper: PaperRef | None = None
        citedPaper: PaperRef | None = None
    citations: list[Citation]

CITATION_FIELDS = Citations.Citation.model_fields.keys()

class CitationsByPaper(pydantic.BaseModel):
    papers: dict[str, Citations]

def fetch_papers(query: str, fields: list[str] = PAPER_FIELDS, limit = None):
    res = sch.search_paper(
        query=query,
        fields=fields,
        limit=limit if limit else 100,
        bulk=True,
    )
    while res._has_next_page():
        res.next_page()
    return res

def fetch_citations(paper_id: int, fields: list[str] = CITATION_FIELDS, limit = None):
    res = sch.get_paper_citations(
        paper_id=paper_id,
        fields=CITATION_FIELDS,
        limit=limit if limit else 100,
    )
    while res._has_next_page():
        res.next_page()
    return res

def dump(data: PaginatedResults, file_name: str):
    if isinstance(data, list):
        dmp = [d._data for d in data]
    else:
        dmp = data._data
    with open(file_name, 'w') as f:
        json.dump(dmp, f)