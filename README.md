# facefilter-scoreboard

Pipeline that scores face filters by fusing FaceNet similarity and LPIPS style distance, then exports matrix+raw Excel reports with auto-formatting.

## Features
- Detects source images automatically (shortest filename per folder).
- Computes FaceNet cosine similarity (512-dim embeddings) and LPIPS-VGG style distance.
- Dynamically mixes scores (FaceNet-only when style distance < 0.02, otherwise 0.3/0.7 weights).
- Generates a single Excel file with matrix & raw sheets, centered values, and red highlights for scores >= 80.
- Skips sensitive data (`input/`, `output/`, `.txt`) via `.gitignore` to keep the repo lightweight.

## Project Structure
```
filter/
├── facenet_pipeline.py       # Main pipeline script
├── README.md                 # Project overview (this file)
├── LICENSE                   # MIT License
└── .gitignore                # Ignore rules (input/output/results/etc)
```

## Requirements
- Python 3.10+
- pip packages: `torch`, `torchvision`, `facenet-pytorch`, `lpips`, `pandas`, `openpyxl`, `Pillow`

## Usage
1. Activate your Python environment (example uses an existing venv):
   ```powershell
   & "$env:USERPROFILE\Desktop\privacy-aware-face-recognition\venv\Scripts\Activate.ps1"
   ```
2. Place per-person folders (each with one source + variants) inside `filter/input/`.
3. Run the pipeline:
   ```powershell
   & "$env:USERPROFILE\Desktop\privacy-aware-face-recognition\venv\Scripts\python.exe" `
     "$env:USERPROFILE\Desktop\filter\facenet_pipeline.py"
   ```
4. Collect results from `filter/output/facenet_scores.xlsx`.

## Notes
- Change LPIPS backbone via `--lpips-net` (default `vgg`, also accepts `alex` or `squeeze`).
- Adjust sensitivity using `--style-threshold` (default `0.02`) and `--style-distance-cap` (default `0.2`).
- The repo intentionally excludes raw images and Excel outputs; only code/config remains.

## License
Released under the MIT License. See [LICENSE](LICENSE) for details.
