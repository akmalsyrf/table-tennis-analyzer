# Pipeline Flow (POC)

Dokumen ini menjelaskan alur pemrosesan dari video sampai keluar statistik sederhana, sesuai kode saat ini.

## Ringkasan singkat

`video` → (OpenCV decode) → `frames` (di-sampling + resize) → **per effective frame:** `table ROI` (metadata) + `ball detection` (YOLO + frame differencing) → `tracking` → `event detection` → `stats` → JSON response + `outputs/<video_stem>.json` + optional overlay MP4.

## Flowchart

![Pipeline flow](images/pipeline-flow.png)

Sumber diagram (untuk regenerate): `docs/pipeline-flow.mmd`.

```bash
npx -y @mermaid-js/mermaid-cli -i docs/pipeline-flow.mmd -o docs/images/pipeline-flow.png -b white -w 1400
```

## Detail per tahap (mengacu ke kode)

### 1) API: upload → simpan file

- Endpoint: `POST /analyze` (JSON) dan alur web `POST /upload`
- Input: multipart form field `file`
- Output JSON (`AnalyzeResponse`): `total_rallies`, `rallies[]` (`hits`, `duration`). **Tidak** ada field `roi` di response API.

Implementasi API: `app/routes/analyze.py` — video disimpan ke `uploads/`, lalu `analyze_video(video_path, outputs_dir)`.

### 2) Video → frames (decode + sampling + resize)

Implementasi: `app/services/video_service.py` (`analyze_video`, `_collect_detections_and_rois`).

- `cv2.VideoCapture`, baca `fps_native`
- `frame_step = max(1, round(fps_native / target_fps))` — default `target_fps` di `app/config.py` (`PipelineTuning`, biasanya **15**)
- `fps_effective = fps_native / frame_step`
- Hanya frame dengan `frame_idx % frame_step == 0`
- Resize jika lebar > `max_width` (default **960**)

### 3) Table ROI (per frame, metadata saja)

Implementasi: `app/cv/table_roi.py` (`detect_table_roi_frame`).

- Mendeteksi area meja dengan **Canny + Hough lines** (garis putih meja), bukan HSV “blob biru”.
- Hasil `TableROI | None` disimpan per frame di output JSON (`track[].roi`), dan dipakai overlay debug.
- **Tidak** memfilter deteksi bola — bola tidak dipotong oleh ROI.

Parameter tuning lama `TableRoiTuning` di config masih ada untuk kompatibilitas; deteksi edge-based saat ini mengabaikan sebagian besar field tersebut (lihat docstring di `detect_table_roi_frame`).

### 4) Ball detection (YOLO + motion)

Implementasi: `app/cv/detection.py` (`BallDetector`), `app/cv/motion.py` (`MotionDetector`).

**YOLO (utama):**

- Model default: `yolov8n.pt`
- Filter confidence `yolo_conf` (default ~0.30), ukuran bbox (`min_box_area` / `max_box_area`)
- Prioritas class **32** (COCO “sports ball”) bila tersedia

**Frame differencing:**

- `prev_gray` vs `curr_gray`: `absdiff` → threshold → contour kecil sebagai `MotionBlob`
- Skor kandidat YOLO digabung dengan **konfirmasi motion** di dekat pusat bbox (`motion_match_radius`)
- Jika YOLO tidak menghasilkan apa pun: **fallback** ke blob motion terbaik (confidence rendah, `source="motion"`)

**Output:** `Detection(x, y, conf, area, source)` — `source` salah satu `yolo`, `yolo+motion`, `motion`.

Jika YOLO sama sekali tidak ter-load, dipakai path HSV fallback lama (`_detect_fallback`).

### 5) Tracking (single-object, prediksi + cooldown)

Implementasi: `app/cv/tracking.py` (`track_positions`).

- Prediksi posisi berikutnya dari kecepatan dua titik terakhir
- Terima deteksi jika jarak ke prediksi ≤ `max_jump_px`
- Setelah `reacquire_cooldown` miss berturut-turut, tracker reset (mencegah loncat ke false positive jauh)

Output: list `TrackPoint | None` dengan field `x`, `y`, `conf`, `source`.

### 6) Event detection → rallies + hits

Implementasi: `app/cv/events.py` (`compute_rallies`).

- Rally start / end / `hit_frames` seperti sebelumnya (missing frames, perubahan arah)
- Semua threshold bekerja pada timeline **FPS efektif**

### 7) Stats + output JSON

Di `app/services/video_service.py`:

- Response API: `total_rallies`, `rallies[]`
- File `outputs/<video_stem>.json`:
  - `fps_native`, `fps_effective`, `frame_step`, `resize_max_width`
  - `debug`: `num_frames`, `num_detections`, `num_tracked`
  - `track`: per indeks effective frame, objek dengan `x`, `y`, `conf`, `source`, dan **`roi`** (dict play area + `table_*`, atau `null` jika meja tidak terdeteksi) — cocok untuk klip kamera yang berganti (wide / close-up / replay)

### 8) Video overlay (review visual)

- `outputs/overlays/<video_stem>.mp4`
- Detail: `docs/overlay-video.md`
- Termasuk kotak debug **ROI** (magenta) dan **TABLE** (cyan) per frame bila ROI terdeteksi

### 9) Evaluasi visual (offline)

Script: `scripts/eval_detection.py` — cuplikan frame dengan panel YOLO + motion + hasil akhir + ROI.

```bash
just eval-detection uploads/<id>.mp4 --samples 30
```

Lihat juga `justfile` recipe `eval-detection`.

## Parameter yang paling sering di-tuning

Di `app/config.py` (`PipelineTuning` dan turunannya):

| Area | Field | Efek kasar |
|------|--------|------------|
| Sampling | `target_fps`, `max_width` | Akurasi vs kecepatan |
| YOLO | `yolo_conf`, `max_box_area`, `min_box_area` | Miss vs false positive |
| Motion | `motion_match_radius` | Seberapa dekat motion harus ke bbox YOLO |
| Track | `max_jump_px`, `reacquire_cooldown` | Kelancaran track vs penolakan loncatan |
| Rally | `missing_end_seconds`, `missing_end_min_frames`, `direction_change_threshold` | Panjang rally / sensitivitas hit |

Untuk table ROI edge-based, parameter masih di dalam `table_roi.py` (Canny/Hough); belum semua diekspos ke `TableRoiTuning`.
