"""
src/data_engineering/ingest_opencellid.py
──────────────────────────────────────────
Downloads real cell tower data from OpenCelliD for network topology enrichment.
Tower density per H3 cell helps distinguish coverage gaps from anomalies.

HOW TO GET DATA (free):
  1. Register at https://opencellid.org/register
  2. Get API token from profile page
  3. Set OPENCELLID_TOKEN in .env
  4. Run: python src/data_engineering/ingest_opencellid.py --country ID

Usage:
    python src/data_engineering/ingest_opencellid.py --country ID
    python src/data_engineering/ingest_opencellid.py --local data/external/cell_towers.csv
"""

import os, sys, gzip, shutil, argparse
from pathlib import Path

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

COUNTRY_MCC = {
    "ID": [510], "SG": [525], "MY": [502], "AU": [505], "GB": [234, 235],
}
BOUNDS = {
    "ID": {"lat_min": -11.0, "lat_max": 6.0, "lon_min": 95.0, "lon_max": 141.0},
}


def download(token: str, output_dir: Path) -> Path:
    url = (f"https://download.opencellid.org/ocid/downloads"
           f"?token={token}&type=full&file=cell_towers.csv.gz")
    gz_path = output_dir / "cell_towers.csv.gz"
    logger.info("Downloading OpenCelliD (~1 GB)...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(gz_path, "wb") as f:
            for chunk in r.iter_content(131072):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/total*100:.1f}%", end="")
    print()
    return gz_path


def filter_towers(gz_path: Path, output_dir: Path, country: str) -> pd.DataFrame:
    mcc_list = COUNTRY_MCC.get(country, [])
    bounds   = BOUNDS.get(country, {"lat_min": -90, "lat_max": 90,
                                     "lon_min": -180, "lon_max": 180})
    csv_path = output_dir / "cell_towers_raw.csv"
    with gzip.open(gz_path, "rb") as fin, open(csv_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)

    cols = ["radio","mcc","net","area","cell","unit","lon","lat",
            "range","samples","changeable","created","updated","averageSignal"]
    chunks = []
    for chunk in pd.read_csv(csv_path, names=cols, chunksize=500_000, low_memory=False):
        if mcc_list:
            chunk = chunk[chunk["mcc"].isin(mcc_list)]
        chunk = chunk[
            (chunk["lat"].between(bounds["lat_min"], bounds["lat_max"])) &
            (chunk["lon"].between(bounds["lon_min"], bounds["lon_max"]))
        ]
        if len(chunk):
            chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def process(df: pd.DataFrame) -> gpd.GeoDataFrame:
    df = df.rename(columns={"net":"mnc","area":"lac","cell":"cell_id",
                             "lon":"longitude","lat":"latitude",
                             "range":"range_m","averageSignal":"avg_signal"})
    df = df.dropna(subset=["latitude","longitude"])
    df = df[df["latitude"].between(-90,90) & df["longitude"].between(-180,180)]
    for col in ["created","updated"]:
        df[col] = pd.to_datetime(df[col], unit="s", errors="coerce")
    df["generation"] = df["radio"].map({"GSM":"2G","UMTS":"3G","LTE":"4G","NR":"5G"}).fillna("?")
    geom = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")


def compute_h3_density(gdf: gpd.GeoDataFrame, resolution: int = 8) -> pd.DataFrame:
    """Count towers per H3 cell."""
    try:
        import h3
        gdf = gdf.copy()
        gdf["h3_cell"] = gdf.apply(
            lambda r: h3.geo_to_h3(r["latitude"], r["longitude"], resolution), axis=1
        )
        return (gdf.groupby("h3_cell")
                .agg(tower_count=("h3_cell","count"),
                     lte_count=("generation", lambda x: (x=="4G").sum()),
                     nr_count=("generation",  lambda x: (x=="5G").sum()),
                     avg_range_m=("range_m","mean"))
                .reset_index())
    except ImportError:
        logger.warning("H3 not installed — skipping density")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Ingest OpenCelliD tower data")
    parser.add_argument("--country", default="ID")
    parser.add_argument("--token",   default=None)
    parser.add_argument("--local",   default=None)
    parser.add_argument("--output",  default="data/external")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = args.token or os.getenv("OPENCELLID_TOKEN")

    if args.local:
        cols = ["radio","mcc","net","area","cell","unit","lon","lat",
                "range","samples","changeable","created","updated","averageSignal"]
        df_raw = pd.read_csv(args.local, names=cols, low_memory=False)
        mcc_list = COUNTRY_MCC.get(args.country, [])
        if mcc_list:
            df_raw = df_raw[df_raw["mcc"].isin(mcc_list)]
    elif token:
        gz = download(token, output_dir)
        df_raw = filter_towers(gz, output_dir, args.country)
    else:
        logger.error(
            "Provide --token or --local.\n"
            "  Free token: https://opencellid.org/register\n"
            "  Set in .env: OPENCELLID_TOKEN=your_token"
        )
        sys.exit(1)

    gdf = process(df_raw)
    parquet = output_dir / f"cell_towers_{args.country}.parquet"
    gdf.drop(columns=["geometry"]).to_parquet(parquet, index=False)
    logger.success(f"Saved {len(gdf):,} towers → {parquet}")

    density = compute_h3_density(gdf)
    if len(density):
        density.to_parquet(output_dir / f"tower_density_h3_{args.country}.parquet", index=False)
        logger.success(f"H3 tower density → {output_dir}/tower_density_h3_{args.country}.parquet")

    print("\nRadio breakdown:")
    print(gdf["radio"].value_counts().to_string())


if __name__ == "__main__":
    main()
