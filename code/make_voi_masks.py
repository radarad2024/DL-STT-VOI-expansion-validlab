"""Generate the R-VOI and E-VOI masks from the tumor segmentation (S-VOI).

  R-VOI  rectangular VOI: the smallest rectangular box enclosing the tumor
         segmentation, aligned with the image axes.
  S-VOI  standard VOI: the tumor segmentation itself (input, unchanged).
  E-VOI  extended VOI: the tumor segmentation extended by 1 cm into the
         peritumoral tissue. The extension is performed within each axial slice
         (in-plane), so the E-VOI covers the same slices as the segmentation.

Input : <src>/EN|T1|T2/<patient_id>/*.nii.gz   (S-VOI directory; each sequence folder holds the
        image and its own segmentation, whose file name starts with "seg")
Output: <dst_root>/R-VOI/... and <dst_root>/E-VOI/... with the same layout
        (images are copied, masks are replaced by the new VOI masks).

Usage
    python make_voi_masks.py --src ./data/train/S-VOI --dst_root ./data/train
    python make_voi_masks.py --src ./data/external/S-VOI --dst_root ./data/external
"""
import os
import shutil
import argparse

import numpy as np
import nibabel as nib
from scipy import ndimage

MARGIN_MM = 10.0   # peritumoral extension of the E-VOI


def slice_axis_of(zooms):
    """Axial slices are stacked along the axis with the largest voxel spacing."""
    return int(np.argmax(zooms))


def rectangular_voi(seg):
    """Smallest axis-aligned box enclosing the segmentation."""
    box = np.zeros_like(seg, dtype=bool)
    idx = np.argwhere(seg)
    if idx.size == 0:
        return box
    lo, hi = idx.min(axis=0), idx.max(axis=0) + 1
    box[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return box


def extended_voi(seg, zooms, margin_mm=MARGIN_MM, slice_axis=None):
    """Segmentation extended by margin_mm within each axial slice."""
    if slice_axis is None:
        slice_axis = slice_axis_of(zooms)
    in_plane = [z for a, z in enumerate(zooms) if a != slice_axis]
    seg_s = np.moveaxis(seg, slice_axis, 0)
    out = np.zeros_like(seg_s, dtype=bool)
    for k in range(seg_s.shape[0]):
        if seg_s[k].any():
            dist = ndimage.distance_transform_edt(~seg_s[k], sampling=in_plane)   # mm to the tumor
            out[k] = dist <= margin_mm
    return np.moveaxis(out, 0, slice_axis)


def process_mask(path, out_r, out_e, margin_mm, slice_axis):
    img = nib.load(path)
    seg = np.asanyarray(img.dataobj) > 0
    zooms = img.header.get_zooms()[:3]
    r_voi = rectangular_voi(seg)
    e_voi = extended_voi(seg, zooms, margin_mm, slice_axis)
    for voi, out in ((r_voi, out_r), (e_voi, out_e)):
        nib.save(nib.Nifti1Image(voi.astype(np.uint8), img.affine, img.header), out)
    return int(seg.sum()), int(r_voi.sum()), int(e_voi.sum())


def main():
    ap = argparse.ArgumentParser(description="Generate R-VOI and E-VOI masks from the tumor segmentation.")
    ap.add_argument("--src", default="./data/train/S-VOI", help="S-VOI directory (images and tumor segmentation)")
    ap.add_argument("--dst_root", default="./data/train", help="R-VOI/ and E-VOI/ are created here")
    ap.add_argument("--margin_mm", type=float, default=MARGIN_MM)
    ap.add_argument("--slice_axis", type=int, default=None,
                    help="array axis of the axial slices (default: axis with the largest voxel spacing)")
    args = ap.parse_args()

    n = 0
    for seq in sorted(os.listdir(args.src)):
        seq_dir = os.path.join(args.src, seq)
        if not os.path.isdir(seq_dir):
            continue
        for pid in sorted(os.listdir(seq_dir)):
            p = os.path.join(seq_dir, pid)
            if not os.path.isdir(p):
                continue
            dst = {v: os.path.join(args.dst_root, v, seq, pid) for v in ("R-VOI", "E-VOI")}
            for d in dst.values():
                os.makedirs(d, exist_ok=True)
            for f in sorted(os.listdir(p)):
                if not f.endswith(".nii.gz"):
                    continue
                if f.startswith("seg"):
                    s, r, e = process_mask(os.path.join(p, f), os.path.join(dst["R-VOI"], f),
                                           os.path.join(dst["E-VOI"], f), args.margin_mm, args.slice_axis)
                    print(f"{seq}/{pid}: S-VOI {s} voxels, R-VOI {r}, E-VOI {e}")
                    n += 1
                else:
                    for d in dst.values():
                        shutil.copy2(os.path.join(p, f), os.path.join(d, f))
    print(f"\n{n} masks processed.")


if __name__ == "__main__":
    main()
