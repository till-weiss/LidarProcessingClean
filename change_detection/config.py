from pathlib import Path
import rasterio
import numpy as np


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

reference_dem = "data/reference_dem.tif"
target_dem = "data/target_dem.tif"

stable_mask = "data/stable_mask.tif"

output_dir = "results"

terrain_mode = "flat"   # "flat" or "sloped"


# -------------------------------------------------
# PIPELINE
# -------------------------------------------------

class CoregistrationPipeline:

    def __init__(
        self,
        reference_dem,
        target_dem,
        stable_mask=None,
        output_dir="output",
        terrain_mode="flat",
    ):

        self.reference_dem = Path(reference_dem)
        self.target_dem = Path(target_dem)

        self.stable_mask = (
            Path(stable_mask)
            if stable_mask is not None
            else None
        )

        self.output_dir = Path(output_dir)

        self.terrain_mode = terrain_mode

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -------------------------------------------------

    def load_raster(self, path):

        with rasterio.open(path) as src:

            data = src.read(1)
            profile = src.profile

        return data, profile

    # -------------------------------------------------

    def load_inputs(self):

        self.ref, self.ref_profile = self.load_raster(
            self.reference_dem
        )

        self.tgt, self.tgt_profile = self.load_raster(
            self.target_dem
        )

        if self.stable_mask is not None:

            self.mask, _ = self.load_raster(
                self.stable_mask
            )

    # -------------------------------------------------

    def compute_dh(self):

        self.dh = self.tgt - self.ref

    # -------------------------------------------------

    def apply_stable_mask(self):

        if self.stable_mask is None:

            self.stable_dh = self.dh[np.isfinite(self.dh)]

            return

        stable = self.mask > 0

        self.stable_dh = self.dh[stable]

    # -------------------------------------------------

    def estimate_vertical_shift(self):

        self.vertical_shift = np.nanmedian(
            self.stable_dh
        )

    # -------------------------------------------------

    def correct_target(self):

        self.corrected = (
            self.tgt - self.vertical_shift
        )

    # -------------------------------------------------

    def save_corrected_dem(self):

        out_path = (
            self.output_dir / "target_corrected.tif"
        )

        profile = self.tgt_profile.copy()

        profile.update(dtype="float32")

        with rasterio.open(
            out_path,
            "w",
            **profile
        ) as dst:

            dst.write(
                self.corrected.astype(np.float32),
                1,
            )

        print(f"Saved corrected DEM: {out_path}")

    # -------------------------------------------------

    def run(self):

        print("Loading inputs...")
        self.load_inputs()

        print("Computing elevation differences...")
        self.compute_dh()

        if self.terrain_mode == "flat":

            print("Applying stable terrain mask...")
            self.apply_stable_mask()

        elif self.terrain_mode == "sloped":

            print("Using all terrain")
            self.stable_dh = self.dh[np.isfinite(self.dh)]

        else:

            raise ValueError(
                f"Unknown terrain_mode: {self.terrain_mode}"
            )

        print("Estimating vertical bias...")
        self.estimate_vertical_shift()

        print(
            f"Estimated vertical shift: "
            f"{self.vertical_shift:.3f} m"
        )

        print("Correcting target DEM...")
        self.correct_target()

        print("Saving result...")
        self.save_corrected_dem()


# -------------------------------------------------
# RUN
# -------------------------------------------------

if __name__ == "__main__":

    pipeline = CoregistrationPipeline(
        reference_dem=reference_dem,
        target_dem=target_dem,
        stable_mask=stable_mask,
        output_dir=output_dir,
        terrain_mode=terrain_mode,
    )

    pipeline.run()