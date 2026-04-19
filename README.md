# 🌲 Carbon Accounting Tool

> Satellite imagery → Canopy Height → Forest Classification → Carbon Stock
> Powered by Meta's CHMv2 (DINOv3) model and Chave 2014 allometry.

---

## What it does

Given a geographic bounding box, this tool:

1. **Fetches satellite patches** from ESRI World Imagery (~0.6m/px)
2. **Fetches elevation data** from Copernicus DEM GLO-30 via Planetary Computer
3. **Predicts canopy height** per patch using Meta's CHMv2 (DINOv3-ViT + DPT head)
4. **Classifies forest type** from elevation (Sal / Chir Pine / Oak / Alpine)
5. **Calculates biomass & carbon** using Chave 2014 allometric equations
6. **Outputs** per-layer GeoTIFFs and a 6-panel diagnostic visualization

```
ESRI tiles ──► CHMv2 inference ──► Canopy Height
                                        │
COP DEM ─────► Forest Classification ──┤
                                        ▼
                              AGB + Carbon Density
                               GeoTIFFs + PNG viz
```

---

## Forest Zones (Uttarakhand)

| Elevation | Forest Type | Wood Density |
|-----------|-------------|--------------|
| 0–1000 m | Sal Forest (*Shorea robusta*) | 0.82 g/cm³ |
| 1000–1800 m | Chir Pine (*Pinus roxburghii*) | 0.49 g/cm³ |
| 1800–2800 m | Oak/Banj (*Quercus*) | 0.72 g/cm³ |
| 2800 m+ | High Alpine | 0.60 g/cm³ |

---

## Setup

```bash
micromamba env create -f environment.yml -y
micromamba activate carbon_tool
huggingface-cli login   # one-time, for CHMv2 weights
```

---

## Usage

### Step 1 — Fetch data (same bbox for both)

```bash
python scripts/fetch_esri_patches.py --bbox 78.05 30.44 78.09 30.47 --zoom 18
python scripts/fetch_dem_patches.py  --bbox 78.05 30.44 78.09 30.47 --zoom 18
```

### Step 2 — Run the pipeline

```bash
python run.py                  # processes 2 patches by default
python run.py --n 5            # process 5 patches
python run.py --n 1 --out_dir data/output/test
```

### Output

```
data/output/
├── tifs/<patch_name>/
│   ├── chm.tif             # Canopy height (m)
│   ├── dem.tif             # Elevation (m)
│   ├── forest_class.tif    # 1=Sal 2=Pine 3=Oak 4=Alpine
│   ├── agb.tif             # Above-ground biomass (Mg/ha)
│   └── carbon_density.tif  # Carbon stock (MgC/ha)
└── visualizations/
    └── <patch_name>_carbon.png   # 6-panel diagnostic figure
```

---

## Carbon Equations

```
DBH     = a × H^b                              (regional allometry)
AGB     = 0.0673 × (ρ × DBH² × H)^0.976       (Chave 2014)
Carbon  = AGB × 0.47                           (IPCC carbon fraction)
```

---

## Stack

- **Model** — `facebook/dinov3-vitl16-chmv2-dpt-head` (HuggingFace)
- **Imagery** — ESRI World Imagery (zoom 18, ~0.6 m/px)
- **DEM** — Copernicus DEM GLO-30 via Microsoft Planetary Computer
- **Device** — Apple Silicon MPS / CUDA / CPU
