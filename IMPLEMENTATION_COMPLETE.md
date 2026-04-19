# CHMv2 Carbon Accounting: FLOW-BASED ARCHITECTURE ✓

## Status: Ready to Run

Pipeline refactored to follow the correct sequential flow with unified visualization.

---

## Processing Flow (Correct Order)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: FETCH INPUT DATA                                                   │
│  ┌─────────────────────┐  ┌─────────────────────┐                          │
│  │  ESRI Satellite     │  │  DEM Elevation      │                          │
│  │  RGB Imagery        │  │  (30m → 2m)         │                          │
│  │  (512x512x3)        │  │  (512x512)          │                          │
│  └──────────┬──────────┘  └──────────┬──────────┘                          │
│             │                        │                                      │
│             └──────────┬─────────────┘                                      │
└────────────────────────┼────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: CANOPY HEIGHT MODEL (CHMv2 Model)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DINOv3-ViT → DPT Head → Canopy Height (meters)                     │   │
│  │  Input: RGB (512x512x3) → Output: CHM (512x512)                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: FOREST CLASSIFICATION (DEM-based Zonation)                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Altitude → Forest Type                                             │   │
│  │  0-1000m   → Sal Forest (Shorea robusta)                            │   │
│  │  1000-1800m → Chir Pine (Pinus roxburghii)                          │   │
│  │  1800-2800m → Oak/Banj (Quercus leucotrichophora)                   │   │
│  │  2800m+    → High Alpine                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: VISUALIZATION (Generate intermediate PNG)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  8-panel unified visualization showing:                             │   │
│  │  RGB → DEM → CHM → Forest Class                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: ABOVE-GROUND BIOMASS (Allometry Calculation)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DBH = a × Height^b  (regional coefficients)                        │   │
│  │  AGB = 0.0673 × (ρ × DBH² × H)^0.976  (Chave 2014)                  │   │
│  │  Output: AGB (Mg/ha)                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: CARBON DENSITY (47% of AGB)                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Carbon = AGB × 0.47  (IPCC default carbon fraction)                │   │
│  │  Output: Carbon Density (MgC/ha)                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: GeoTIFF Files + Unified Visualization PNG                          │
│  - outputs/geotiffs/chm.tif                                                 │
│  - outputs/geotiffs/dem.tif                                                 │
│  - outputs/geotiffs/forest_class.tif                                        │
│  - outputs/geotiffs/agb.tif                                                 │
│  - outputs/geotiffs/carbon_density.tif                                      │
│  - outputs/visualizations/complete_flow_visualization.png (8-panel)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Files Modified

### 1. **carbon_platform/inference_engine.py** (FLOW REFACTORED)
- ✓ `run_phase_a()` now follows correct sequence:
  1. Fetch ESRI + DEM together
  2. Run CHMv2 inference → canopy height
  3. Classify forest using DEM altitude
  4. Calculate AGB using allometry
  5. Calculate carbon as 47% of AGB
- ✓ Returns first chunk data for immediate visualization
- ✓ All data saved to GeoTIFF files

### 2. **carbon_platform/visualizer.py** (UNIFIED VISUALIZATION)
- ✓ `visualize_single_patch()` now creates 8-panel unified PNG:
  1. RGB Imagery (ESRI Satellite Input)
  2. DEM Elevation (Input)
  3. Canopy Height (CHMv2 Model Output)
  4. Forest Classification (DEM-based Zonation)
  5. Above-Ground Biomass (Allometry Calculation)
  6. Carbon Density (47% of AGB)
  7. DEM vs CHM Scatter
  8. Statistics Summary
- ✓ Added AGB parameter support
- ✓ Main title explains the complete flow

### 3. **run_carbon_accounting.py** (FLOW ORCHESTRATION)
- ✓ Phase A now generates unified visualization automatically
- ✓ `complete_flow_visualization.png` created after first chunk
- ✓ All steps follow the correct sequential order

### 4. **carbon_platform/__init__.py** (LAZY IMPORTS)
- ✓ Lazy imports to avoid dependency errors
- ✓ GeoTIFFManager added to exports

### 5. **carbon_platform/geotiff_manager.py** (NO CHANGES)
- ✓ Proper CRS (EPSG:3857) and Affine transforms
- ✓ LZW compression for GeoTIFFs
- ✓ Handles all layers: chm, carbon_density, forest_class, dem, agb, rgb

---

## How to Run

### Quick Test (Single Chunk)
```bash
cd /Users/neerajkaroshi/Desktop/Projects/chmv2_v2.5

# Activate environment
micromamba activate chmv2_v2

# Run with limit=1 to test the complete flow
python run_carbon_accounting.py \
    --region dehradun \
    --config config.yaml \
    --limit 1 \
    --phase all
```

### Full Run (All Chunks)
```bash
micromamba activate chmv2_v2

python run_carbon_accounting.py \
    --region dehradun \
    --config config.yaml \
    --phase all
```

### Visualize Existing Data
```bash
micromamba activate chmv2_v2

python run_carbon_accounting.py \
    --region dehradun \
    --config config.yaml \
    --visualize-only
```

---

## Expected Output

### GeoTIFF Files (`outputs/geotiffs/`)
| File | Data Type | Description |
|------|-----------|-------------|
| chm.tif | float32 | Canopy Height Model (meters) |
| dem.tif | float32 | Digital Elevation Model (meters) |
| forest_class.tif | uint8 | Forest Classification (1=Sal, 2=Pine, 3=Oak) |
| agb.tif | float32 | Above-Ground Biomass (Mg/ha) |
| carbon_density.tif | float32 | Carbon Density (MgC/ha) |
| red.tif, green.tif, blue.tif | uint8 | ESRI Satellite RGB |

### Visualization (`outputs/visualizations/`)
| File | Description |
|------|-------------|
| complete_flow_visualization.png | **8-panel unified flow visualization** |
| dem_analysis.png | DEM with hillshade, slope, aspect |
| forest_classification_map.png | Forest types with altitude zones |
| data_analysis_charts.png | 10-panel statistical analysis |
| analysis_report.txt | Text summary report |
| analysis_results.csv | Statistical data |

---

## Key Equations

### DBH from Height
```
DBH = a × H^b
```
- `a, b` = Regional allometry coefficients (from config/allometry_params.csv)
- `H` = Canopy height from CHMv2 (meters)

### AGB from DBH (Chave 2014)
```
AGB = 0.0673 × (ρ × DBH² × H)^0.976
```
- `ρ` = Wood density (g/cm³)
- `DBH` = Diameter at Breast Height (cm)
- `H` = Height (m)
- Output: AGB in Mg/ha

### Carbon from AGB
```
Carbon = AGB × 0.47
```
- `0.47` = IPCC default carbon fraction
- Output: Carbon in MgC/ha

---

## Validation Checklist

- [x] InferenceEngine follows correct flow (fetch → CHM → classify → AGB → carbon)
- [x] Visualizer creates unified 8-panel PNG
- [x] Main script orchestrates flow correctly
- [x] GeoTIFF files saved with proper georeferencing
- [x] All Python files compile without syntax errors
- [x] Lazy imports prevent dependency errors

---

## Ready to Run! 🚀

```bash
micromamba activate chmv2_v2 && \
python run_carbon_accounting.py --region dehradun --limit 1 --phase all
```
