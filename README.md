# CHMv2 Carbon Accounting Platform - Run Flow

Complete workflow for processing Dehradun-Mussoorie region with real ESRI tiles and Planetary Computer DEM.

## Prerequisites

```bash
# 1. Create environment
micromamba env create -f environment.yml -y
micromamba activate chmv2_v2

# 2. Verify PyTorch MPS (Apple Silicon)
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# 3. Login to HuggingFace (for CHMv2 weights)
huggingface-cli login
# OR set token:
# export HF_TOKEN="your_token_here"
```

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA SOURCES                                   │
├──────────────────────────────┬──────────────────────────────────────────────┤
│  ESRI World Imagery          │  Planetary Computer                          │
│  Zoom 18 (~0.6m/pixel)      │  COP DEM GLO-30 (30m native)                 │
│  XYZ tiles                   │  STAC API query                              │
└──────────────────────────────┴──────────────────────────────────────────────┘
              │                              │
              ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STREAM LOADER (stream_loader.py)                          │
│                                                                              │
│  1. Fetch ESRI tiles for each 512x512 chunk                                │
│  2. Mosaic multiple tiles → 512x512 RGB @ target resolution                  │
│  3. Query Planetary Computer STAC for DEM                                   │
│  4. Resample 30m DEM → 512x512 @ target resolution                          │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INFERENCE ENGINE (inference_engine.py)                    │
│                                                                              │
│  Phase A (ABA - Area-Based Approach):                                       │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│    │   RGB Input  │───▶│  CHMv2 Model │───▶│  CHM Output  │               │
│    │  (512x512x3) │    │  DINOv3-ViT  │    │  (512x512)   │               │
│    └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                    │                        │
│  Phase B (Classification):                                                   │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│    │  DEM Input   │───▶│ Altitudinal  │───▶│ Forest Class │               │
│    │  (512x512)   │    │   Zonation   │    │  (512x512)   │               │
│    └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                    │                        │
│  Phase C (Carbon Accounting):                                                │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│    │ CHM + Forest │───▶│   Allometry  │───▶│   Carbon     │               │
│    │    Class     │    │  Chave 2014  │    │   Density    │               │
│    └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                    │                        │
│         ┌──────────────────────────────────────────┘                        │
│         ▼                                                                    │
│    ┌─────────────────────────────────────────────┐                         │
│    │  DINOv3 Embeddings (PCA visualization)    │                         │
│    │  (1024 tokens → 64x64 spatial → 512x512)   │                         │
│    └─────────────────────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DATACUBE OUTPUT (Zarr/Xarray)                            │
│                                                                              │
│  Variables per 512x512 chunk:                                              │
│    - red, green, blue      : uint8   (RGB imagery)                        │
│    - dem                   : float32 (elevation meters)                   │
│    - chm                   : float32 (canopy height meters)               │
│    - forest_class          : uint8   (1=Sal, 2=Pine, 3=Oak)                 │
│    - carbon_density        : float32 (MgC/ha)                             │
│    - agb                   : float32 (Mg/ha biomass)                      │
│    - embeddings            : float32 (optional, per-patch PCA)            │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION (visualizer.py)                             │
│                                                                              │
│  Output: 4-panel diagnostic PNG                                             │
│                                                                              │
│  ┌──────────────────┐ ┌──────────────────┐                               │
│  │                  │ │                  │                               │
│  │  Original RGB    │ │ Canopy Height    │                               │
│  │  (ESRI tiles)    │ │ (CHMv2 output)   │                               │
│  │                  │ │  0-40m scale     │                               │
│  └──────────────────┘ └──────────────────┘                               │
│                                                                              │
│  ┌──────────────────┐ ┌──────────────────┐                               │
│  │                  │ │                  │                               │
│  │  Forest Class    │ │ Carbon Density   │                               │
│  │  (DEM-based)     │ │ (MgC/ha)         │                               │
│  │  Sal/Pine/Oak    │ │  Hotspots = red  │                               │
│  └──────────────────┘ └──────────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Run Commands

### Step 1: Initialize DataCube Only
```bash
python run_carbon_accounting.py --region dehradun --init-only
```
**Output:** Empty Zarr store at `data/zarr_store/dehradun_carbon.zarr`

### Step 2: Full Run (Phase A + Phase B + Visualization)
```bash
python run_carbon_accounting.py --region dehradun --phase all
```
**Process:**
1. Connects to Planetary Computer STAC API
2. Searches COP DEM GLO-30 for Dehradun bounds
3. For each 512x512 chunk:
   - Fetches ESRI tiles (mosaics if multiple needed)
   - Fetches DEM (30m → resampled to 512x512)
   - Runs CHMv2 inference (DINOv3 → CHM)
   - Classifies forest by altitude (Sal <1000m, Pine 1000-1800m, Oak 1800-2800m)
   - Calculates carbon using Chave 2014 equation
4. Saves to Zarr DataCube
5. Identifies high-carbon hotspots (Phase B)
6. Generates 4-panel visualizations

**Outputs:**
- `data/zarr_store/dehradun_carbon.zarr/` — Full DataCube
- `data/output/visualizations/` — PNG diagnostics

### Step 3: Visualization Only (existing DataCube)
```bash
python run_carbon_accounting.py --region dehradun --visualize-only
```
**Output:** Re-generates PNGs from existing Zarr store

## Altitudinal Zonation Config

Edit `config.yaml` section `altitude_zones`:

```yaml
altitude_zones:
  - name: "Sal_Forest"      # Shorea robusta
    min_alt: 0
    max_alt: 1000
    class_code: 1
    wood_density: 0.82
    allometry_a: 0.396
    allometry_b: 1.089
    
  - name: "Chir_Pine"       # Pinus roxburghii
    min_alt: 1000
    max_alt: 1800
    class_code: 2
    wood_density: 0.49
    allometry_a: 0.307
    allometry_b: 1.138
    
  - name: "Oak_Banj"        # Quercus leucotrichophora
    min_alt: 1800
    max_alt: 2800
    class_code: 3
    wood_density: 0.72
    allometry_a: 0.235
    allometry_b: 1.246
```

## Carbon Calculation Equations

### 1. DBH from Height
```
DBH = a × H^b
```
Where:
- DBH = Diameter at Breast Height (cm)
- H = Canopy Height from CHMv2 (m)
- a, b = Regional allometry coefficients

### 2. AGB from DBH
```
AGB = 0.0673 × (ρ × DBH² × H)^0.976
```
Where:
- AGB = Above Ground Biomass (Mg/ha)
- ρ = Wood density (g/cm³)
- Chave 2014 equation (tropical/moist forests)

### 3. Carbon Density
```
Carbon = AGB × 0.47
```
Where:
- 0.47 = IPCC default carbon fraction

## Expected Outputs per Chunk

For a 512×512 pixel chunk at 2m resolution (1024m × 1024m area):

| Variable | Shape | Dtype | Units | Description |
|----------|-------|-------|-------|-------------|
| red/green/blue | 512×512 | uint8 | - | RGB imagery |
| dem | 512×512 | float32 | m | Elevation |
| chm | 512×512 | float32 | m | Canopy height |
| forest_class | 512×512 | uint8 | - | 1=Sal, 2=Pine, 3=Oak |
| carbon_density | 512×512 | float32 | MgC/ha | Carbon stock |
| agb | 512×512 | float32 | Mg/ha | Biomass |
| embeddings | 1024×D | float32 | - | DINOv3 tokens (optional) |

## Troubleshooting

### DEM not available
If Planetary Computer query fails:
- Stream loader falls back to synthetic DEM
- Synthetic DEM approximates Dehradun-Mussoorie elevations
- Check `use_real_data=True` in stream_loader initialization

### ESRI tiles fail
If ESRI tiles timeout:
- Falls back to synthetic RGB based on elevation
- Synthetic RGB creates forest-like colors from DEM

### Memory issues
For large regions:
```python
# In config.yaml, reduce batch_size:
model:
  batch_size: 2  # Instead of 4
  
# Or process fewer chunks:
python run_carbon_accounting.py --region dehradun --phase a  # Phase A only
```

### Verify outputs
```python
import xarray as xr

ds = xr.open_zarr("data/zarr_store/dehradun_carbon.zarr")
print(ds)
print("\nForest class distribution:")
print(ds.forest_class.values.flatten())
print("\nCarbon range:", ds.carbon_density.min().values, "to", ds.carbon_density.max().values)
```

## Summary Command

```bash
# Complete workflow
micromamba activate chmv2_v2 && \
python run_carbon_accounting.py --region dehradun --phase all
```

Expected time for Dehradun region (~100 chunks): 15-30 minutes on M1/M2 MacBook Pro.
