from .services.semantic_scholar import PaperID

import pydantic

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException
import nltk

from .services import semantic_scholar
from .services.openai import client

DetectorFactory.seed = 0
nltk.download("punkt")


def is_text_in_language(text: str, language: str) -> bool:
    try:
        if len(text.strip()) == 0:
            return False

        return any(
            lang.lang == language and lang.prob > 0.99 for lang in detect_langs(text)
        )

    except LangDetectException:
        return False


class Translations(pydantic.BaseModel):
    class Translation(pydantic.BaseModel):
        abstract: str
        title: str

    translations: dict[PaperID, Translation]


def translate_paper_to_en(
    paper: semantic_scholar.Papers.Paper,
) -> Translations.Translation:
    if not paper.abstract:
        raise ValueError("Paper does not have an abstract")

    prompt = (
        f"Translate or extract an English version of this abstract : '{paper.abstract}'"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    translated_abstract = response.choices[0].message.content or ""

    prompt = f"Translate this title in English : '{paper.title}'"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    translated_title = response.choices[0].message.content or ""

    return Translations.Translation(
        abstract=translated_abstract,
        title=translated_title,
    )
