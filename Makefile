.PHONY: docker_build
docker_build:
	docker build . -t dev-fgpt

.PHONY: docker_run
docker_run:
	docker run -v $(pwd):/app dev-fgpt

.PHONY: pre_commit
pre_commit:
	black . 
	pytest --exitfirst
