import os
import warnings
import numpy as np
from osgeo import gdal


class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass


class Configuration:
    """ Configuration of all parameters used for DEM/DSM creation """

    def __init__(self):

        # --------- RUN NAME ---------
        self.run_name = 'Perma-X-2023-Till'  # Custom name for this run
        self.year = 2023

        # ---------- PATHS -----------
        # Input data paths
        self.target_area_dir = '/isipd/projects/p_planetdw/data/lidar/01_target_areas/Till'  # Path to vector footprints of target areas
        self.las_files_dir = f'/isipd/projects/p_planetdw/data/lidar/02_pointclouds/{self.year}'  # Path to lidar point clouds (*.las/*.laz)
        self.las_footprints_dir = f'/isipd/projects/p_planetdw/data/lidar/03_las_footprints/{self.year}'  # Path to footprints of flight paths, if not available will be generated

        # Output directories
        self.preprocessed_dir = '/isipd/projects/p_planetdw/data/lidar/04_preprocessed'  # Path for preprocessed lidar data
        self.results_dir = '/isipd/projects/p_planetdw/data/lidar/05_results'  # Path for final DEM/DSM results
        self.validation_dir = '/isipd/projects/p_planetdw/data/lidar/06_validation'  # Path to validation data

        # ------ PREPROCESSING ------

        self.multiple_targets = False  # If target areas are saved in one gdf set to True
        self.target_name_field = 'fid'  # Field in target area gdf to use as target name

        self.preprocess_by_strip = True  # Whether to preserve strip boundaries while chunking (chunk or strip)
        self.chunk_size = 1000  # chunk size (m) for preprocessing

        # elevation outlier cap (quantile in [0–1])
        self.max_elevation_threshold = 0.99  # Higher removes more high outliers (aircraft/atmosphere). Typical 0.98–0.9995.

        # SOR parameters (Statistical Outlier Removal)
        self.sor_knn = 100  # neighbors for stats. 50–200 is common. Higher = stabler but slower.
        self.sor_multiplier = 1  # (mean + m*std). Lower (0.8–1.2) = aggressive; higher (1.5–2.5) = conservative.
        self.sor_passes = 3  # number of SOR passes per chunk. Each pass further removes outliers. Typical 2–5.

        # ELM filter (Extended Local Minimum – removes isolated low-noise points below the ground surface)
        self.elm_filter = True  # enable ELM filter
        self.elm_cell = 10.0   # cell size (m) for local minimum estimation. Smaller = finer noise detection.
        self.elm_threshold = 1.0  # points this far (m) below the cell minimum are classified as noise.

        # Radius outlier filter (removes points with too few neighbours within a given radius)
        self.radius_filter = False  # enable radius-based outlier removal
        self.radius_filter_radius = 1.0   # search radius (m). Smaller = only very isolated points removed.
        self.radius_filter_min_count = 4  # minimum neighbours within radius to keep a point.

        # ------ PREPROCESSING ------
        # ICP switch
        self.enable_icp = True

        # ICP-ready strip creation
        self.icp_use_ground_only = True
        self.icp_min_ground_points = 800
        self.icp_voxel_size = 2.0

        # SMRF ground classification
        self.smrf_window_size = 20.0
        self.smrf_slope = 0.2
        self.smrf_scalar = 2.0
        self.threshold = 0.5

        # ICP registration
        self.icp_max_correspondence_distance = 2.0
        self.icp_max_iterations = 80

        # overlap filtering before ICP
        self.icp_max_candidate_targets = 3
        self.icp_min_bbox_overlap_ratio = 0.05
        self.icp_overlap_buffer = 0.0
        self.icp_min_overlap_points = 2500

        self.icp_min_bbox_overlap_ratio = 0.05
        self.icp_min_overlap_points = 2500
        self.icp_min_fitness = 0.60
        self.icp_max_rmse = 2
        self.icp_max_shift_m = 5.0
        
        '''# ICP settings
        config.icp_min_bbox_overlap_ratio = 0.05
        config.icp_min_overlap_points = 2500
        config.icp_min_fitness = 0.6
        config.icp_max_rmse = 0.8
        config.icp_max_shift_m = 3.0
        config.icp_max_shift_xy_m = 2.0
        config.icp_max_shift_z_m = 1.5
        config.icp_overlap_buffer = 0.0
        config.icp_max_passes = 5'''


        # ------- PROCESSING --------

        self.create_DSM = True
        self.create_DEM = True
        self.create_CHM = False

        self.fill_gaps = True  # use IDW to close gaps in rasters
        self.resolution = 2  # pixel size (m). Smaller = sharper/heavier. Rule of thumb: >= sqrt(1 / points_per_m²).

        self.point_density_method = 'density'  # method to determine point density, can be 'sampling' (exact) or 'density' (fast)

        # ______ GROUND FILTERING ______

        self.smrf_filter = False  # use SMRF filter 
        self.csf_filter = True  # use cloth simulation method
        self.threshold = 0.1  # vertical tolerance (m) for extra clipping. Typical 0.5–2.

        # SMRF
        self.smrf_window_size = 20  # window size (m). Larger (15–30) removes more canopy but may bridge narrow valleys.
        self.smrf_slope = 0.2  # slope tolerance. Higher (0.2–0.4) accepts steeper ground; too high may keep low veg.
        self.smrf_scalar = 2  # elevation diff scale. 1–3 typical. Higher = more aggressive ground acceptance.

        # CSF (Cloth Simulation)
        self.csf_rigidness = 5  # cloth stiffness. 1–2 for rugged/steep; 3–4 for very flat urban.
        self.csf_iterations = 200  # steps. 200–1000. More = better fit, slower.
        self.csf_time_step = 1  # integration step. 0.5–1.0 common. Smaller = stable/accurate, slower.
        self.csf_cloth_resolution = 2  # grid spacing (m). 0.5–2 typical. Smaller = finer ground detail, heavier.

        # ------ VALIDATION ------

        self.data_type = 'raster'   # Type of validation data, can be 'raster' or 'vector' (points)
        self.validation_target = 'DSM' # product to validate, can be 'DSM', 'DEM' or 'CHM', select validation data accordingly! (DSM: higest point, DEM: ground level, CHM: height of vegetation)
        self.val_column_point = 'val_value' # column in point validation data to use for comparison
        self.val_band_raster = 1  # band index of reference raster (usually 1 unless multiband).
        self.sample_size = 100  # samples for stats. 100–10,000. Larger = more stable metrics.

        # ------ ADVANCED SETTINGS ------

        # _______ Preprocessing _______
        self.overlap = 0.0  # min overlap (fraction) between pointcloud and AOI. Typical 0.05–0.3.

        self.filter_date = False  # Filter las files by date

        self.automatic_date_parser = True # get dates from target area only for Region_Site_Date_Res_Order filenames


        self.start_date = '2023-07-10'  # Start date for filtering las files
        self.end_date = '2023-07-10'  # End date for filtering las files

        # _______ Processing _______
        self.chunk_size = 5000  # chunk size (m) for main processing. 250–1000 typical. Larger = fewer edges, more memory.
        self.chunk_overlap = 0.05  # chunk overlap (fraction) for main processing. 0.05–0.3. More reduces seam artifacts.
        self.num_workers = 8  # parallel workers. <= physical cores/RAM capacity.

        # _______ Preprocessing _______
        self.preprocess_chunk_size = 500   # chunk size (m) for preprocessing. Smaller than main = lighter SOR passes.
        self.preprocess_chunk_overlap = 0.05  # overlap fraction for preprocessing chunks. Reduces SOR edge artefacts.
        self.preprocess_reproject_vertical = False  # If True, convert vertical datum during preprocessing.
        self.preprocess_vertical_target_epsg = 3855  # Target vertical EPSG (EGM2008). Used only when preprocess_reproject_vertical is True.

        # Set overall GDAL settings
        gdal.UseExceptions()  # Enable exceptions instead of silent failures
        gdal.SetCacheMax(32000000000)  # GDAL cache bytes (~32 GB). Set to ~20–60% of available RAM for big rasters.
        warnings.filterwarnings('ignore')  # Suppress warnings

    def validate(self):
        """Validate config to catch errors early, and not during or at the end of processing"""

        # Check that required input paths exist
        for path_attr in ["target_area_dir", "las_footprints_dir", "las_files_dir"]:
            path = getattr(self, path_attr)
            if not os.path.exists(path):
                raise ConfigError(f"Invalid path: {path_attr} = {path}")

        # Create required output directories if they don't exist
        for path_attr in ["results_dir", "preprocessed_dir"]:
            path = getattr(self, path_attr)
            try:
                os.makedirs(path, exist_ok=True)
            except OSError:
                raise ConfigError(f"Unable to create folder: {path_attr} = {path}")

        if self.preprocess_reproject_vertical:
            if self.preprocess_vertical_target_epsg is None:
                raise ConfigError("preprocess_vertical_target_epsg must be set when preprocess_reproject_vertical is True")
            try:
                self.preprocess_vertical_target_epsg = int(self.preprocess_vertical_target_epsg)
            except (TypeError, ValueError):
                raise ConfigError("preprocess_vertical_target_epsg must be an integer EPSG code")

        return self
