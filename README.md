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

Then open **http://127.0.0.1:8000/** for the web UI (upload, history, result detail), or use the JSON API below.

### Web UI (Jinja2)

- **/** — upload video (POST ke `/upload`, redirect to results)
- **/history** — list of analyze from `outputs/*.json`
- **/results/{id}** — detail one analyze (without heavy `track` payload)

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

