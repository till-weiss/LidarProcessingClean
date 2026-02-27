# **LidarProcessing**  
A pipeline to process LiDAR point clouds from aerial campaigns into **Digital Elevation Models (DEMs)**. Currently DSM, DTM and CHM generation are supported.

![Example Output](Layout%201.png)


## **Project Structure**  
The code is structured around three main steps in the pipeline:  

1. **Preprocessing** (`preprocessing.py`) – Assigns point clouds to target areas, optionally aligns strips with ICP, and filters outliers.  
2. **Processing** (`processing.py`) – Converts preprocessed point clouds into **DSM, DEM, and CHM** outputs.  
3. **Validation** (`validation.py`) – Evaluates generated models against other rasters or point data.  

These steps are orchestrated by `main.py` and rely on helper methods in the `/core` directory.  
Configuration settings are stored in `/config/` and are initialized in `main.py`.

## **Folder Structure**

```
project_root/
├── 01_target_areas/           # Vector footprints of target AOIs (e.g. shapefiles or GeoJSON)
├── 02_pointclouds/            # Raw LiDAR point clouds (*.las or *.laz)
├── 03_las_footprints/         # Flight path footprints (generated if missing)
├── 04_preprocessed/           # Cleaned point clouds after outlier removal
│   ├── aligned_strips/<AOI>/  # Optional: ICP-aligned strip outputs
│   └── icp_intermediate/<AOI>/# Optional: ICP selected/overlap subsets
├── 05_results/                # Output rasters from processing
│   └── icp_logs/<AOI>/        # Optional: ICP debug logs and iteration details
│   └── <run_name>/            
│       ├── DSM/               # Digital Surface Models
│       ├── DTM/               # Digital Terrain Models
│       └── CHM/               # Canopy Height Models (DSM - DTM)
├── 06_validation/             # External raster or vector data used for validation
```

## **Pipeline Overview**  
### **1. Preprocessing**  
- Assigns LiDAR point clouds to user-defined target areas (e.g., aerial image footprints or AOIs).  
- Target areas should be stored as individual files but can be converted if necessary.  
- Optional strip-to-strip ICP alignment can be run per AOI before merge (`use_strip_icp=True`).  
- Outliers are removed using **statistical outlier removal** techniques.  

### **2. Processing**  
- Converts the preprocessed point clouds into:  
  - **Digital Surface Models (DSM)**  
  - **Digital Elevation Models (DEM)**  
  - **Canopy Height Models (CHM)**  
- Outputs are saved as **GeoTIFFs**.  

### **3. Validation** 
- Evaluates the generated models.  
- Compares against:  
  - **Single points** or **entire raster datasets**.  
  - User-defined pixel subsets or complete rasters.  
- Evaluates ground, surface, and vegetation height.  

![LiDAR Processing Workflow](lidarprocessing_workflow.png)

### **Updated Flow (with optional ICP strip alignment)**
```mermaid
flowchart TD
    A[Target AOI footprints + raw LAS/LAZ strips] --> B[Match strips to AOI]
    B --> C{use_strip_icp?}
    C -- No --> D[Use original strip list]
    C -- Yes --> E[Select ICP points per strip
(ground candidates or full cloud)]
    E --> F[Extract XY overlap per strip pair]
    F --> G[Incremental ICP
strip k -> aligned strip k-1]
    G --> H[Apply accepted transform to FULL strip
(preserve all LAS attributes)]
    H --> I[Aligned strip list]
    D --> J[Merge + clean point clouds per AOI]
    I --> J
    J --> K[Preprocessed AOI LAS]
    K --> L[DSM generation]
    K --> M[DEM generation]
    L --> N[CHM generation (optional)]
```

For efficiency, point clouds are split into **smaller chunks** and processed in **parallel**.  

## **Setup**
For an easy setup we recommend pixi. After clonting the repo simply run:
```bash
pixi install
```

Of course, you can also use the old way. The pipeline requires **Python 3.8+** and the following dependencies:  
```bash
pip install pdal laspy shapely geopandas rasterio numpy scipy tqdm
```
or using Conda:  
```bash
conda install -c conda-forge pdal laspy shapely geopandas rasterio numpy scipy tqdm
```
But the easiest way is to use Pixi, trust me!

## **Usage**  
### **1. Configure Settings**  
Before running the pipeline, update the paths and parameters in `config.py`.  
You can create **separate config files** for different datasets or use cases and select them in `main.py`. 

### **2. Run the Pipeline**  
To process LiDAR data from start to finish, simply run:  
```bash
python main.py
```
This executes all pipeline steps: **Preprocessing → Processing → Validation**.

### **3. Running Individual Steps**  
To run specific steps:  
- Convert `main.py` into a **Jupyter Notebook**.  
- Or manually **comment out** the steps you don’t need.  

### **For easy access use the TUI**.
We have inculded an interactive user Interface which allows you to configure your parameters, set your processing steps and products. To access it, in your terminal run:

```bash 
pixi run LidarProcessing
```

or, after activating your environment:
```bash
python LidarProcessing.py
```
Your output will look like this:

![ui](ui.png)

## **Coming Soon **  
**GPU Acceleration** for faster processing.  
**HPC (High-Performance Computing) Support** for large-scale datasets.  
