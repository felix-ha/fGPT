.PHONY: docker_build
docker_build:
	docker build . -t fgpt

.PHONY: docker_run
docker_run:
	docker run -v $(pwd):/app fgpt

.PHONY: docker_shell
docker_shell:
	docker run  -it -v $(pwd):/app --entrypoint bash fgpt

.PHONY: pre_commit
pre_commit:
	black . 
	pytest --exitfirst

.PHONY: data_pipeline
data_pipeline:
	python dask_pipeline.py --ratio=1.0 --full
