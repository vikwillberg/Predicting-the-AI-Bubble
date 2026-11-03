.PHONY: install engine data validate all test linkedin clean

install:
	pip install -r requirements.txt

engine:
	cd engine && go build -o ../bin/lppls ./cmd/lppls

data:
	python -m icarus ingest

validate:
	python -m icarus validate

all: engine
	python -m icarus all

test: engine
	cd engine && go test ./...
	python -m pytest -q

linkedin:
	python -m icarus linkedin

clean:
	rm -rf bin data/processed/*.duckdb data/processed/artifacts .pytest_cache
