## Table Tennis Analyzer (POC)

Minimal FastAPI API that accepts a table tennis video upload and returns simple rally/hit statistics.

### Docs

- Pipeline flow: `docs/flow.md`

### Setup

Recommended (uses `justfile`):

```bash
just install
```

### Run

```bash
just run
```

### API

- `POST /analyze` (multipart form upload field name: `file`)

Example:

```bash
just analyze /path/to/video.mp4
```

### Fallback (no `just`)

If you don’t want to use `just`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

