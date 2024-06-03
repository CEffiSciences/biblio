# What is it
This project creates cartography representation of axes of research. See output/graph.png and output/clusters.html for example of what this looks like when applied to bioterrorism.


# How to install
To install first install [graphviz](https://graphviz.org/), for instance through your distribution's package manager
Then do:

```
poetry install
```

Then you can just simply do
```
poetry run biblio
```

or if your environment is activated
```
biblio
```

Copy .env to .env.local and populate it with your own values for OPENAI_API_KEY and SEMANTIC_SCHOLAR_API_KEY

# How to use
In code documentation is pending, parameters should be somewhat transparent in the meantime.
If you do not need parametrization and just want to produce outputs, you may do:
```
biblio pipeline
```

which will call all necessary commands in order

The order in which the function needs to be called:
- fetch-papers: Fetch all papers associated with a keyword or query
- generate-translation: Translate all papers in English
- generate-clusters: Clusters the papers depending on their word embedding and produces output/clusters.html
- generate-threat-scores: Evaluate how much each clusters contribute to different axes of impact (TODO: This is the weakest part of this process yet, refactor)
- fetch-references: Get all references mentionned by all articles from fetch-papers
- generate-graph: Produces output/graph.png
