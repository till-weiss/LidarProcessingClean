from config import get_config
from coregister import co_register_dem_pair
from change import compute_change_products
from report import save_outputs


def run_pipeline():
    cfg = get_config()

    print("=" * 70)
    print(f"Change detection workflow: {cfg.aoi_name} | {cfg.dem_type} | {cfg.ref_year}->{cfg.target_year}")
    print(f"Reference DEM: {cfg.dem_reference_path}")
    print(f"Target DEM   : {cfg.dem_target_path}")
    print(f"Output dir   : {cfg.output_dir}")
    print("=" * 70)

    coreg_data = co_register_dem_pair(cfg)
    change_data = compute_change_products(cfg, coreg_data)
    save_outputs(cfg, coreg_data, change_data)

    print("\nFinished.")


if __name__ == "__main__":
    run_pipeline()
