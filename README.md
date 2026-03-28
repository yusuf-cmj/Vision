# Image Enhancement Pipeline — Assignment

## Quick Start

```bash
# 1. Activate virtual environment (Windows)
venv\Scripts\activate

# 2. Launch Jupyter
jupyter notebook

# 3. Open Vision_Enhancement.ipynb
# 4. Run All Cells: Kernel > Restart & Run All
# 5. Export to PDF: File > Save and Export Notebook As > HTML
#    (then print HTML to PDF from browser — most reliable on Windows)
```

## Project Structure

```
Vision/
├── venv/                        # Python virtual environment
├── data/
│   ├── raw/                     # Original images (auto-generated on first run)
│   └── results/                 # All output figures and metrics CSV
│       ├── figure01_dataset.png
│       ├── intensity_ops/       # Figures 3–6
│       ├── spatial_filters/     # Figures 7–9
│       ├── pipelines/           # Figures 10–13
│       ├── failure_cases/       # Figures 14–15
│       └── metrics_table.csv    # MSE / PSNR / SSIM for 4 controlled pairs
├── src/
│   ├── dataset_prep.py          # Image loading + degradation functions
│   ├── intensity_ops.py         # Gamma, stretch, HE, CLAHE, piecewise linear
│   ├── spatial_filters.py       # Mean, Gaussian, Median, Bilateral, Unsharp, Laplacian
│   ├── pipelines.py             # Pipeline 1 (grayscale) + Pipeline 2 (CIE Lab)
│   ├── metrics.py               # MSE, PSNR, SSIM
│   └── visualization.py         # All plotting helpers
├── Vision_Enhancement.ipynb     # MAIN REPORT — run this
└── requirements.txt
```

## Reinstall dependencies (if needed)

```bash
venv\Scripts\pip install --only-binary :all: -r requirements.txt
```
