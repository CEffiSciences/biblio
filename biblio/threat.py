from .clusters import ClusterID

import pydantic

import re

from .services.openai import client


# TODO: Should be refactored to use function calls so that threat_scores axes can be easily modified, and that it is overall more robust
# Unsure this method of prompting for experts really work, list of important leverages should probably be GPT-generated in the first place
class ThreatScores(pydantic.BaseModel):
    class ThreatScore(pydantic.BaseModel):
        prion: list[float] = []
        viral: list[float] = []
        bacterial: list[float] = []
        toxin: list[float] = []
        fungal: list[float] = []

    threat_scores: dict[ClusterID, ThreatScore] = {}


def threat_score(label: str) -> ThreatScores.ThreatScore:
    prompt = f"""
    Imagine a panel of ten experts in the research axis '{label}', each evaluating how directly their research axis contributes to mitigating bioterrorist threats on a scale from 0 to 1, where 0 indicates no direct contribution and 1 indicates a direct and substantial contribution to reducing the threat.

    Research Axis: '{label}'

    Each expert provides a score answering the question: "How directly does your research axis contribute to reducing the bioterrorist threat of type [type]?"

    Please provide five scores for each expert, corresponding to the following threat types:
    1. Viral Threats
    2. Bacterial Threats
    3. Toxin-Based Threats
    4. Fungal Threats
    5. Prion-Based Threats

    Format the response exactly as follows:
    Expert 1: Viral - X, Bacterial - X, Toxin - X, Fungal - X, Prion - X
    Expert 2: Viral - X, Bacterial - X, Toxin - X, Fungal - X, Prion - X
    ...
    Expert 10: Viral - X, Bacterial - X, Toxin - X, Fungal - X, Prion - X
    """
    response = (
        client.chat.completions.create(
            model="gpt-4-turbo",  # Could be gpt-4o, is faster and cheaper, but results are less consistent with gpt-4o, and gpt-4-turbo has higher variance
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            stop=["\n\n"],
        )
        .choices[0]
        .message.content
    )
    assert response

    result = ThreatScores.ThreatScore()
    score_pattern = re.compile(
        r"Expert \d+: Viral - (?P<viral>\d\.\d+), Bacterial - (?P<bacterial>\d\.\d+), Toxin - (?P<toxin>\d\.\d+), Fungal - (?P<fungal>\d\.\d+), Prion - (?P<prion>\d\.\d+)"
    )

    for match in re.finditer(score_pattern, response):
        for i in ThreatScores.ThreatScore.model_fields:
            lst = getattr(result, i, [])
            lst.append(float(match.group(i)))

    return result
