set dotenv-load := false

PY := "python3"
VENV_DIR := ".venv"

default:
    @just --list

venv:
    {{PY}} -m venv {{VENV_DIR}}

install: venv
    {{VENV_DIR}}/bin/python -m pip install -r requirements.txt

run:
    {{VENV_DIR}}/bin/uvicorn app.main:app --reload

analyze VIDEO:
    curl -F "file=@{{VIDEO}}" http://127.0.0.1:8000/analyze

# Visual ball-detection diagnostics: YOLO + motion + final result (see docs/flow.md §9)
eval-detection *ARGS:
    {{VENV_DIR}}/bin/python scripts/eval_detection.py {{ARGS}}

clean:
    rm -rf {{VENV_DIR}} __pycache__ .pytest_cache outputs/*.json
