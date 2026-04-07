# Video overlay (rally + hit)

Dokumen ini menjelaskan **video overlay** yang dihasilkan setelah analisis: apa isinya, di mana file-nya, bagaimana hubungannya dengan pipeline, dan batasannya.

## Tujuan

Overlay dipakai untuk **review visual** tanpa harus membaca angka saja:

- Melihat posisi bola yang di-track (titik kuning).
- Melihat **rally** mana yang sedang aktif dan **jumlah hit** kumulatif untuk rally itu.
- Melihat **momen “HIT”** (inferensi dari perubahan arah gerakan bola), bukan ground truth dari wasit.

## Kapan overlay dibuat

Setelah `analyze_video()` selesai menghitung `track` dan `rallies`, generator overlay dipanggil dari `app/services/video_service.py`. Jika penulisan overlay gagal (codec tidak tersedia, disk penuh, dll.), analisis JSON/API tetap bisa sukses; error dicatat di log.

## Lokasi file

- Direktori: `outputs/overlays/`
- Nama file: `<stem_upload>.mp4`, sama dengan stem file video di `uploads/` (misalnya UUID analisis: `outputs/overlays/<analysis_id>.mp4`).

Stem mengikuti `video_path.stem` dari file yang diunggah, sehingga cocok dengan `outputs/<analysis_id>.json` ketika upload web memakai nama `uploads/<analysis_id>.mp4`.

## Pemutaran di web UI

Halaman hasil (`/results/{analysis_id}`) menampilkan **video overlay** jika file tersebut ada:

- Sumber video: `GET /overlays/{analysis_id}`
- Jika overlay tidak ada, UI jatuh ke video asli: `GET /videos/{analysis_id}`.

## Apa yang digambar di setiap frame

Implementasi: `app/services/overlay_service.py` (`generate_rally_hit_overlay_video`).

| Elemen | Deskripsi |
|--------|-----------|
| Kotak **ROI** (magenta) | Play area hasil `detect_table_roi_frame` untuk frame itu (`rois_by_frame[track_i]`). Tidak digambar jika ROI `None`. |
| Kotak **TABLE** (cyan) | Perkiraan permukaan meja (`table_x`, `table_y`, `table_w`, `table_h`) dari objek `TableROI` yang sama. |
| Titik kuning | Pusat bola dari `track[track_i]` jika tidak `None` (setelah tracking, bukan raw detector). |
| Teks atas | `Rally X/Y \| Hits Z` — rally ke-*X* dari *Y* total, *Z* = hit kumulatif untuk rally itu (sama dengan logika `compute_rallies`). |
| Bar bawah | Timeline: latar abu-abu, segmen oranye = interval `[start_frame, end_frame]` rally aktif, garis putih = posisi saat ini di timeline. |
| Teks **HIT** + lingkaran hijau | Hanya pada indeks frame yang ada di `Rally.hit_frames` (inferensi arah berubah drastis). |

## Timeline penting: “effective frames”

Overlay **bukan** satu-satu dengan setiap frame asli video:

- Pipeline hanya memproses frame yang lolos sampling (`frame_step` dari `fps_native` menuju `target_fps`, lihat `app/services/video_service.py`).
- Panjang `track` = jumlah **effective frame**.
- Indeks `track_i` di overlay sama dengan indeks di `track` dan dengan `start_frame` / `end_frame` / `hit_frames` di `Rally` (`app/cv/events.py`).

Artinya: durasi video overlay mengikuti **FPS efektif** (`fps_effective`), dan jumlah frame overlay ≈ panjang `track`, bukan jumlah frame native.

## Codec dan browser

Default-nya, OpenCV mencoba membuka `VideoWriter` dengan codec berurutan:

- Linux: `mp4v` dulu (supaya tidak memicu error/noise ffmpeg ketika probing H.264), lalu `avc1`, `H264`, `X264`.
- Non-Linux: `avc1`, `H264`, `X264`, lalu fallback `mp4v`.

Jika ingin memaksa probing H.264 dulu (lebih ramah untuk pemutaran di browser), set env `OVERLAY_PREFER_H264=1`.

Catatan: jika hanya `mp4v` yang berhasil, beberapa browser bisa bermasalah — kalau perlu, uji di Chrome/Firefox atau konversi manual dengan ffmpeg.

### Transcode otomatis (disarankan untuk web)

Jika `ffmpeg` tersedia di PATH, generator overlay akan mencoba **transcode otomatis** menjadi:

- H.264 (`libx264`)
- `-pix_fmt yuv420p`
- `-movflags +faststart` (supaya video bisa mulai diputar sebelum full download)

Kamu bisa mematikan transcode dengan env `OVERLAY_TRANSCODE=0`.

## Hubungan dengan JSON analisis

File `outputs/<id>.json` berisi `track` (per effective frame: posisi bola, `conf`, `source`, dan `roi` atau `null`), `rallies` (ringkas: `hits`, `duration`), dan metadata `fps_effective` / `frame_step`. Field `hit_frames` **tidak** diserialisasi ke JSON file; overlay memakai data `Rally` langsung di memori saat analisis. Response API `POST /analyze` hanya mengembalikan `total_rallies` dan `rallies` — detail track/ROI hanya di file JSON di `outputs/`.

## Referensi cepat kode

- Generator overlay: `app/services/overlay_service.py`
- Pemanggilan setelah analisis: `app/services/video_service.py`
- Rally + `hit_frames`: `app/cv/events.py` (`compute_rallies`)
- Route streaming overlay: `app/routes/web.py` (`GET /overlays/{analysis_id}`)
- Template pemutar: `app/templates/result.html`
