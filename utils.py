# For downnloading datasets

from pathlib import Path
from urllib.request import urlretrieve

DSPRITES_URL = (
    "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)

def download_dsprites(dataset_dir="datasets"):
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    filepath = dataset_dir / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

    if filepath.exists():
        print(f"Already exists: {filepath}")
        return filepath

    print(f"Downloading dSprites to {filepath}...")
    urlretrieve(DSPRITES_URL, filepath)
    print("Done.")

    return filepath