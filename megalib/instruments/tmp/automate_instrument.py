from pathlib import Path

# ---------- config ----------
TEMPLATE_FILE = Path("PHEMTO_config4x4_5cm.geo.setup")
OUT_DIR = Path(".")

# original CdTe-related lines in the template
MOTHER_OLD = "CdTe4x4.Mother WorldVolume"
POS_OLD = "CdTe4x4.Position 0. 0. -5.125 // Bottom Si <-> Top CdTe = 5cm"

INCLUDE_GEOM_OLD = "Include ../detectors/geometry/cdte4x4.geo"
INCLUDE_DET_OLD  = "Include ../detectors/CdTe4x4detector.det"
# ----------------------------


def float_range(start, stop, step):
    x = start
    while x < stop - 1e-9:
        yield round(x, 10)
        x += step


def main():
    template_text = TEMPLATE_FILE.read_text()

    indices = [4, 5, 6, 7, 8, 9]
    dists = list(float_range(0.5, 10.5, 0.5))  # 0.5, 1.0, ..., 9.5

    for i in indices:
        for dist in dists:
            # CdTe lines
            mother_new = f"CdTe{i}x{i}.Mother WorldVolume"
            z_value = -(dist + 0.125)
            pos_new = (
                f"CdTe{i}x{i}.Position 0. 0. {z_value:.3f} "
                f"// Bottom Si <-> Top CdTe = {dist:g}cm"
            )

            # include lines (si include left untouched)
            include_geom_new = f"Include ../detectors/geometry/cdte{i}x{i}.geo"
            include_det_new  = f"Include ../detectors/CdTe{i}x{i}detector.det"

            new_text = (
                template_text
                .replace(MOTHER_OLD,      mother_new)
                .replace(POS_OLD,         pos_new)
                .replace(INCLUDE_GEOM_OLD, include_geom_new)
                .replace(INCLUDE_DET_OLD,  include_det_new)
            )

            dist_str = f"{dist:g}"
            out_name = OUT_DIR / f"PHEMTO_config{i}x{i}_{dist_str}cm.geo.setup"
            out_name.write_text(new_text)
            print(f"Wrote {out_name}")


if __name__ == "__main__":
    main()

