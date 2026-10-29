"""Preview the UV input mode on one vHRM Studio file: how much of the body map is valid at a given size.

For a sample of frames of one *_skin_uv.h5 file it reports, per frame size S and minimum coverage, the fraction of
valid output pixels (all charts and per chart) and writes a PNG of the per-pixel valid fraction (how often each pixel
of the S x S map holds skin). Use it to choose UV.MIN_COVERAGE and the frame size (PREPROCESS.RESIZE).

The figures show coverage only; --rgb adds the mean skin colour map, which shows the face texture (identifying).

    python tools/uv_preview.py path/to/cam2_Camera_Front_skin_uv.h5 --sizes 36 72 128 --output runs/uv_preview
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.data_loader import uv_source as U  # noqa: E402


def sample_rows(status, count):
    good = np.flatnonzero(status == 1)
    if len(good) <= count:
        return good
    return good[np.linspace(0, len(good) - 1, count).round().astype(int)]


def preview(h5_path, sizes, coverages, layout, frame_count):
    prefix = U.LAYOUT_PREFIXES[layout]
    rows_out, maps = [], {}
    with h5py.File(h5_path, "r") as h5:
        status = np.asarray(h5["frame_status"][()])
        rows = sample_rows(status, frame_count)
        names = [str(name) for name in h5.attrs.get("packed_uv_chart_names", [])] if layout == "packed" else []
        regions = {"all": U.region_mask(h5, layout)}
        for name in names:
            regions[name] = U.region_mask(h5, layout, charts=(name.lower(),))
        textures = [h5[prefix + "skin_rgb"][row] for row in rows]
        mask_dataset = h5.get(prefix + "skin_mask_packed")
        skins = [np.unpackbits(mask_dataset[row], axis=-1).astype(bool) if mask_dataset is not None
                 else U.skin_texels(texture) for row, texture in zip(rows, textures)]
    for name, region in regions.items():
        skin_fraction = float(np.mean([(skin & region).sum() / max(region.sum(), 1) for skin in skins]))
        for size in sizes:
            region_count = U.pool_region(region, size)
            pooled = [U.pool_masked(texture, skin & region, size, region_count) for texture, skin in zip(textures, skins)]
            for minimum in coverages:
                valid = np.stack([coverage >= max(minimum, 1e-9) for _, coverage in pooled])
                rows_out.append({
                    "region": name, "size": size, "min_coverage": minimum,
                    "skin_texel_fraction": round(skin_fraction, 4),
                    "region_pixels": int((region_count > 0).sum()),
                    "valid_pixels_mean": round(float(valid.reshape(len(valid), -1).sum(axis=1).mean()), 1),
                    "valid_fraction_of_region": round(float(valid.sum(axis=(1, 2)).mean()
                                                            / max((region_count > 0).sum(), 1)), 4),
                })
                if name == "all":
                    maps[(size, minimum)] = (valid.mean(axis=0), np.mean([mean for mean, _ in pooled], axis=0))
    return pd.DataFrame(rows_out), maps, len(rows)


def save_figure(maps, sizes, coverages, path, show_rgb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = len(sizes) * (2 if show_rgb else 1)
    figure, axes = plt.subplots(len(coverages), columns, figsize=(3.2 * columns, 3.2 * len(coverages)),
                                squeeze=False)
    for row, minimum in enumerate(coverages):
        for column, size in enumerate(sizes):
            frequency, mean_rgb = maps[(size, minimum)]
            axis = axes[row, column * (2 if show_rgb else 1)]
            image = axis.imshow(frequency, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
            axis.set_title(f"{size}px, min cov {minimum}: valid freq.", fontsize=8)
            axis.axis("off")
            if show_rgb:
                rgb_axis = axes[row, column * 2 + 1]
                rgb_axis.imshow(np.clip(mean_rgb / 255.0, 0, 1), interpolation="nearest")
                rgb_axis.set_title(f"{size}px mean skin RGB", fontsize=8)
                rgb_axis.axis("off")
    figure.colorbar(image, ax=axes, shrink=0.6, label="fraction of frames with a valid pixel")
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5", help="A vHRM Studio *_skin_uv.h5 file.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[36, 72, 128])
    parser.add_argument("--min-coverage", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--layout", choices=sorted(U.LAYOUT_PREFIXES), default="packed")
    parser.add_argument("--frames", type=int, default=60, help="Frames sampled evenly over the file.")
    parser.add_argument("--output", default=os.path.join("runs", "uv_preview"), help="Output folder.")
    parser.add_argument("--rgb", action="store_true", help="Also draw the mean skin colour (shows the face).")
    args = parser.parse_args()

    table, maps, sampled = preview(args.h5, args.sizes, args.min_coverage, args.layout, args.frames)
    os.makedirs(args.output, exist_ok=True)
    stem = os.path.basename(args.h5).removesuffix(".h5")
    csv_path = os.path.join(args.output, f"{stem}_coverage.csv")
    png_path = os.path.join(args.output, f"{stem}_coverage.png")
    table.to_csv(csv_path, index=False)
    save_figure(maps, args.sizes, args.min_coverage, png_path, args.rgb)
    summary = table[table["region"] == "all"].drop(columns="region")
    print(f"{args.h5}: {sampled} frames sampled, {args.layout} layout\n")
    print(summary.to_string(index=False))
    charts = table[(table["region"] != "all") & (table["size"] == max(args.sizes))
                   & (table["min_coverage"] == args.min_coverage[0])]
    if len(charts):
        print(f"\nPer chart at {max(args.sizes)}px, min coverage {args.min_coverage[0]}:")
        print(charts[["region", "skin_texel_fraction", "region_pixels", "valid_pixels_mean"]].to_string(index=False))
    print(f"\nWrote {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
