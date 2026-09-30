.PHONY: install install-colab validate captions prepare train generate evaluate app test lint

install:
	pip install -r requirements.txt
	pip install -e .

install-colab:
	pip install -r requirements-colab.txt
	pip install -e .

validate:
	python scripts/validate_dataset.py

captions:
	python scripts/generate_captions.py --write-metadata

prepare:
	python scripts/prepare_dataset.py

train:
	python scripts/train_lora.py --config configs/lora_sd15.yaml

generate:
	python scripts/generate.py --prompt "$(PROMPT)"

evaluate:
	python scripts/evaluate.py --input outputs/generated --reference data/reference

app:
	python app/app.py

test:
	pytest -v

lint:
	python -m pyflakes src scripts app || true
