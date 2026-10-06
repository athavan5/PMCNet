"""
Fix HDF5 weight file so Keras 2.2.x can load it.
Kaggle/some exports save attributes as str; this env expects bytes.
Run once: python fix_weights_attrs.py
"""
import h5py
import os

weights_dir = os.path.join(os.path.dirname(__file__), "weights")
path = os.path.join(weights_dir, "EfficientNetB0.h5")
path_fixed = os.path.join(weights_dir, "EfficientNetB0_fixed.h5")
if not os.path.isfile(path):
    print("weights/EfficientNetB0.h5 not found")
    exit(1)

# Copy full file structure, then write root attributes as bytes
with h5py.File(path, "r") as src, h5py.File(path_fixed, "w") as dst:
    for key in src.keys():
        src.copy(key, dst)
    for key in ("keras_version", "backend"):
        if key in src.attrs:
            v = src.attrs[key]
            dst.attrs[key] = v.encode("utf-8") if isinstance(v, str) else v
    if "layer_names" in src.attrs:
        dst.attrs["layer_names"] = src.attrs["layer_names"]

os.replace(path_fixed, path)
print("Done. Try training again.")
