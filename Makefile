-include params.env
export

common:
	@python src/common.py

extract_papers: common
	@python src/step_1_extract_papers.py

extract_citations: common
	@python src/step_2_extract_citations.py

generate_grah: common
	@python src/step_4_generate_graph.py