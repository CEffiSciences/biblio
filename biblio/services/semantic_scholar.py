from typing import Iterable
import pydantic

from semanticscholar.SemanticScholarException import NoMorePagesException
from semanticscholar import SemanticScholar

from ..env import env

sch = SemanticScholar(api_key=env.semantic_api_key)

PaperID = str


class Papers(pydantic.BaseModel):
    class Paper(pydantic.BaseModel):
        paperId: PaperID
        abstract: str | None
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

    papers: dict[PaperID, Paper]


def fetch_papers(
    query: str,
    fields: Iterable[str] = Papers.Paper.model_fields.keys(),
    limit: int = 100,
) -> Papers:
    res = sch.search_paper(
        query=query,
        fields=list(fields),
        limit=limit,
        bulk=True,
    )

    result = {}
    while True:
        result |= {i["paperId"]: i for i in res._data}

        try:
            res.next_page()
        except NoMorePagesException:
            break
    return Papers(papers=result)


class References(pydantic.BaseModel):
    class Reference(pydantic.BaseModel):
        class PaperRef(pydantic.BaseModel):
            paperId: PaperID | None = None

        paperId: PaperID | None = None
        citedPaper: PaperRef
        intents: list[str]
        isInfluential: bool

    references: list[Reference]


class ReferencesByPaper(pydantic.BaseModel):
    papers: dict[PaperID, References]


def fetch_references(
    paper_id: str,
    fields: Iterable[str] = (
        i for i in References.Reference.model_fields.keys() if i not in {"citedPaper"}
    ),
    limit: int = 100,
) -> References:
    res = sch.get_paper_references(
        paper_id=paper_id,
        fields=list(fields),
        limit=limit,
    )

    result_raw = []
    while True:
        result_raw += res._data
        try:
            res.next_page()
        except NoMorePagesException:
            break

    return References(references=result_raw)
