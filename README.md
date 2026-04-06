## Table Tennis Analyzer (POC)

Minimal FastAPI API that accepts a table tennis video upload and returns simple rally/hit statistics.

### Docs

- Pipeline flow: `docs/flow.md`
- Video overlay (rally + hit): `docs/overlay-video.md`

### Visual eval (detection debug)

Generate side-by-side PNGs (YOLO candidates, frame-diff motion, final pick + table ROI) for a video:

```bash
just eval-detection uploads/<analysis_id>.mp4 --samples 30
```

Optional: `--out <dir>` (default `eval_out`). Requires the same venv as `just install` / `just run`.

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
- **/results/{id}** — detail one analyze (without heavy `track` payload); video player uses **overlay** (`/overlays/{id}`) when present

### API

- `POST /analyze` (multipart form upload field name: `file`)

Example:

```bash
just analyze /path/to/video.mp4
```

After uploading via the web UI, use the file under `uploads/` with `just eval-detection` to inspect detection behaviour frame-by-frame.

### Fallback (no `just`)

If you don’t want to use `just`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

