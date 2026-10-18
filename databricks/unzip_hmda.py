"""Unzip the HMDA LAR archives inside the volume, where the bundle's bronze stage expects <year>_public_lar_csv.csv."""

import shutil
import time
import zipfile

RAW = "/Volumes/workspace/lending/raw"

for year in (2023, 2024, 2025):
    start = time.time()
    name = f"{year}_public_lar_csv.csv"
    with zipfile.ZipFile(f"{RAW}/zips/{year}_public_lar_csv.zip") as z:
        info = z.getinfo(name)
        with z.open(info) as src, open(f"{RAW}/{name}", "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    print(f"{name}: {info.file_size} bytes in {time.time() - start:.0f} s")
