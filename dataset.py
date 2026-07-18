from __future__ import annotations

import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DEFAULT_FILTER_COLUMNS = (
    "Mass",
    "Suspicious_Lymph_Node",
    "Nipple_Retraction",
    "Skin_Retraction",
    "Skin_Thickening",
    "Suspicious_Calcification",
    "No_Finding",
)

DEFAULT_CLS_POSITIVE_COLUMNS = (
    "Mass",
    "Suspicious_Lymph_Node",
    "Nipple_Retraction",
    "Skin_Retraction",
    "Skin_Thickening",
    "Suspicious_Calcification",
)


@dataclass
class DatasetConfig:
    root_dir: Path = field(default_factory=lambda: Path.cwd().resolve())
    data_dirname: str = "mammo"
    gcs_images_tar: str = "gs://helen-data/square_images.tar.gz"
    gcs_data_csv: str = "gs://helen-data/square_data.csv"
    gcs_splits_json: str = "gs://helen-data/dataset_splits.json"
    tar_filename: str = "square_images.tar.gz"
    csv_filename: str = "square_data.csv"
    splits_filename: str = "dataset_splits.json"
    download_from_gcs: bool = True
    extract_images: bool = True
    filter_columns: tuple[str, ...] = DEFAULT_FILTER_COLUMNS
    cls_column: str = "cls"
    cls_positive_columns: tuple[str, ...] = DEFAULT_CLS_POSITIVE_COLUMNS
    cls_positive_value: int = 1

    @property
    def data_dir(self) -> Path:
        return self.root_dir / self.data_dirname

    @property
    def raw_img_dir(self) -> Path:
        return self.data_dir / "raw" / "images"

    @property
    def raw_csv_dir(self) -> Path:
        return self.data_dir / "raw" / "csv"

    @property
    def splits_dir(self) -> Path:
        return self.root_dir / "splits"

    @property
    def csv_main(self) -> Path:
        return self.raw_csv_dir / self.csv_filename

    @property
    def tar_local(self) -> Path:
        return self.raw_img_dir / self.tar_filename

    @property
    def splits_local(self) -> Path:
        return self.splits_dir / self.splits_filename


def ensure_dataset_dirs(config: DatasetConfig) -> None:
    for directory in (config.raw_img_dir, config.raw_csv_dir, config.splits_dir):
        directory.mkdir(parents=True, exist_ok=True)


def _gcs_uri_to_https(uri: str) -> str:
    assert uri.startswith("gs://"), f"URI GCS inválida: {uri}"
    return "https://storage.googleapis.com/" + uri[len("gs://"):]


def _download_from_gcs(source: str, destination_file: Path) -> None:
    import requests  # type: ignore

    destination_file.parent.mkdir(parents=True, exist_ok=True)
    url = _gcs_uri_to_https(source)
    print(f"GET {url}")
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination_file.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)


def _has_extracted_images(config: DatasetConfig) -> bool:
    return any(
        path.is_file() and path.name != config.tar_filename
        for path in config.raw_img_dir.rglob("*")
    )


def ensure_dataset_downloaded(config: DatasetConfig) -> None:
    ensure_dataset_dirs(config)

    if not config.download_from_gcs:
        return

    if not config.tar_local.is_file() or config.tar_local.stat().st_size == 0:
        print("Downloading images...")
        _download_from_gcs(config.gcs_images_tar, config.tar_local)
        if not config.tar_local.is_file() or config.tar_local.stat().st_size == 0:
            raise FileNotFoundError(f"No se pudo descargar {config.tar_local}.")
        print("Images downloaded")

    if config.extract_images and not _has_extracted_images(config):
        print("Extracting images...")
        with tarfile.open(config.tar_local, "r:gz") as tar:
            tar.extractall(config.raw_img_dir)
        print(f"Images extracted to {config.raw_img_dir}")

    if not config.csv_main.is_file() or config.csv_main.stat().st_size == 0:
        print("Downloading CSV...")
        _download_from_gcs(config.gcs_data_csv, config.csv_main)
        print("CSV downloaded")

    if not config.splits_local.is_file() or config.splits_local.stat().st_size == 0:
        print("Downloading dataset splits...")
        _download_from_gcs(config.gcs_splits_json, config.splits_local)
        print(f"Splits downloaded to {config.splits_local}")


def load_raw_dataframe(config: DatasetConfig) -> pd.DataFrame:
    ensure_dataset_downloaded(config)
    if not config.csv_main.is_file():
        raise FileNotFoundError(f"No existe el CSV principal: {config.csv_main}")
    ds_raw = pd.read_csv(config.csv_main, low_memory=False)
    ds_raw["path"] = ds_raw.apply(
        lambda row: str(config.raw_img_dir / str(row["patient_id"]) / str(row["image_id"])),
        axis=1,
    )
    return ds_raw


def add_cls_column(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    missing_columns = [
        column for column in config.cls_positive_columns if column not in df.columns
    ]
    if missing_columns:
        raise KeyError(f"Faltan columnas para generar {config.cls_column}: {missing_columns}")

    df = df.copy()
    df[config.cls_column] = (
        df[list(config.cls_positive_columns)].eq(config.cls_positive_value).any(axis=1)
    ).astype("float32")
    return df


def filter_existing_files(df: pd.DataFrame, path_column: str = "path") -> pd.DataFrame:
    """Descarta filas cuya imagen no exista en disco (evita NotFoundError en tf.data)."""
    if df.empty or path_column not in df.columns:
        return df
    exists = df[path_column].map(os.path.exists)
    missing = int((~exists).sum())
    if missing:
        print(
            f"[dataset] {missing}/{len(df)} filas descartadas: imagen no encontrada en disco "
            "(CSV con mas entradas que el tar o extraccion incompleta)."
        )
    return df[exists].copy()


def _sample_dataframe(
    df: pd.DataFrame,
    fraction: float,
    seed: int | None,
) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("sample_fraction debe estar entre 0 y 1.")

    if df.empty or fraction == 1:
        return df.copy()

    return df.sample(frac=fraction, random_state=seed).copy()


def download_and_build_dataset(
    config: dict | None = None,
    *,
    dataset_config: DatasetConfig | None = None,
    reduced: bool | None = None,
    sample_fraction: float = 0.10,
    sample_seed: int | None = 42,
) -> dict[str, object]:
    """Descarga/arma el dataset. ``reduced`` sale de ``config["GENERAL"]["REDUCED_DATASET"]``
    salvo que se pase explicito; ``dataset_config`` es la config de I/O (GCS/paths)."""
    if reduced is None:
        reduced = bool(config["GENERAL"]["REDUCED_DATASET"]) if config is not None else False
    dataset_config = dataset_config or DatasetConfig()
    ds_raw = load_raw_dataframe(dataset_config)
    missing_columns = [column for column in dataset_config.filter_columns if column not in ds_raw.columns]
    if missing_columns:
        raise KeyError(f"Faltan columnas para el filtrado: {missing_columns}")

    ds_raw = add_cls_column(ds_raw, dataset_config)
    ds = ds_raw[ds_raw[list(dataset_config.filter_columns)].eq(1).any(axis=1)].copy()
    if reduced:
        ds = _sample_dataframe(ds, sample_fraction, sample_seed)

    return {
        "root": dataset_config.root_dir,
        "data_dir": dataset_config.data_dir,
        "raw_img_dir": dataset_config.raw_img_dir,
        "raw_csv_dir": dataset_config.raw_csv_dir,
        "csv_main": dataset_config.csv_main,
        "ds_raw": ds_raw,
        "ds": ds,
        "reduced": reduced,
        "sample_fraction": sample_fraction if reduced else 1.0,
    }
