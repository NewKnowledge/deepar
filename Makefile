build:
	@echo "Building Image"
	DOCKER_BUILDKIT=1 docker build -t deepar . -f Dockerfile

build-reproduce:
	@echo "Building Reproduce Image"
	DOCKER_BUILDKIT=1 docker build -t reproduce-amazon -f Dockerfile.repro .

run:
	@echo "Running"
	docker-compose run deepar

notebook:
	@echo "Starting a Jupyter notebook"
	docker-compose run -p 8888:8888 deepar \
		jupyter notebook --ip=0.0.0.0 \
		--NotebookApp.token='' --NotebookApp.password='' \
		--no-browser --allow-root \
		--notebook-dir="notebooks"

notebook-reproduce:
	@echo "Starting a Jupyter notebook to reproduce Amazon results" 
	docker-compose -f docker-compose.reproduce.yml run -p 8888:8888 reproduce-amazon \
		jupyter notebook --ip=0.0.0.0 \
		--NotebookApp.token='' --NotebookApp.password='' \
		--no-browser --allow-root \
		--notebook-dir="notebooks"

test:
	@echo "Building Image"
	DOCKER_BUILDKIT=1 docker build -t deepar -f Dockerfile .

	@echo "Running Tests"
	docker-compose run deepar pytest --color=yes -s tests/