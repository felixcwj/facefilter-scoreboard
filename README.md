# facefilter-scoreboard

Pipeline that scores face filters by fusing FaceNet similarity and LPIPS style distance, then exports matrix+raw Excel reports with auto-formatting.

## Features
- Detects the source image automatically (shortest filename per folder).
- Computes FaceNet cosine similarity (512-dim embeddings) and LPIPS-VGG style distance.
- Dynamically mixes scores (FaceNet-only when style distance < 0.02, otherwise 0.3/0.7 weights).
- Generates a single Excel file with matrix & raw sheets, centered values, and red highlights for scores >= 80.
- Keeps the repo lightweight via `.gitignore` (inputs, outputs, Excel artifacts excluded).

## Project Structure
```
filter/
├── facenet_pipeline.py   # Main pipeline script
├── README.md             # Project overview (this file)
├── LICENSE               # MIT License
└── .gitignore            # Ignore rules (input/output/results/etc)
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

## CLI Options

| Option | Default | Description |
| --- | --- | --- |
| `--input-dir` | `input` | Root folder that holds per-person subfolders. |
| `--output-dir` | `output` | Destination folder for the Excel workbook. |
| `--excel-name` | `facenet_scores.xlsx` | Name of the generated workbook. |
| `--style-threshold` | `0.02` | LPIPS distance threshold to trigger style weighting. |
| `--style-distance-cap` | `0.2` | Max LPIPS distance mapped to 0% style similarity. |
| `--lpips-net` | `vgg` | LPIPS backbone (`vgg`, `alex`, `squeeze`). |

## Sample Workflow
1. Drop two folders such as `input/haerin/` and `input/no/` (each with one source image + variants that share suffixes like `_h`, `_n`, etc.).
2. Run the pipeline once—Excel will contain a `matrix` sheet (filters × sources) and a `raw` sheet (all intermediate values).
3. Re-run whenever you add new folders; existing sources remain while new ones append automatically.
4. Inspect `raw` to diagnose why certain filters stayed red (≥80%)—it includes `face_percent`, `style_percent`, LPIPS distances, and applied weights.

## Notes
- Repository intentionally excludes raw images, Word notes, and Excel outputs so the repo stays lightweight.
- The Excel formatting (center alignment, red ≥80) happens automatically—no macro required.

## Roadmap / Ideas
- Add CLI flag to dump CSV summaries alongside Excel.
- Provide a tiny synthetic sample dataset so new users can test without uploading real faces.
- Integrate optional CLIP-based score for even stronger style awareness.

## License
Released under the MIT License. See [LICENSE](LICENSE) for details.
