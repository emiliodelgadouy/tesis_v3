from __future__ import annotations

import subprocess
import shutil
import sys
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
    tar_filename: str = "square_images.tar.gz"
    csv_filename: str = "square_data.csv"
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
    def csv_main(self) -> Path:
        return self.raw_csv_dir / self.csv_filename

    @property
    def tar_local(self) -> Path:
        return self.raw_img_dir / self.tar_filename


def ensure_dataset_dirs(config: DatasetConfig) -> None:
    for directory in (config.raw_img_dir, config.raw_csv_dir):
        directory.mkdir(parents=True, exist_ok=True)


def authenticate_colab_user() -> None:
    try:
        from google.colab import auth  # type: ignore
    except ImportError:
        return

    auth.authenticate_user()


def _run_command(command: list[str]) -> None:
    print("+", " ".join(command))
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    result.check_returncode()


def _resolve_command(name: str) -> str:
    executable = shutil.which(name)
    if executable:
        return executable

    local_executable = Path(sys.executable).with_name(name)
    if local_executable.is_file():
        return str(local_executable)

    return name


def _download_from_gcs(source: str, destination_file: Path) -> None:
    destination_file.parent.mkdir(parents=True, exist_ok=True)

    commands = [
        [_resolve_command("gcloud"), "storage", "cp", source, str(destination_file)],
        [_resolve_command("gsutil"), "-m", "cp", source, str(destination_file)],
    ]

    last_error = None

    for command in commands:
        try:
            _run_command(command)
            if destination_file.is_file() and destination_file.stat().st_size > 0:
                return
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            last_error = error

    raise FileNotFoundError(
        f"No se pudo descargar {source} hacia {destination_file}. "
        f"Último error: {last_error}"
    )


def _has_extracted_images(config: DatasetConfig) -> bool:
    return any(
        path.is_file() and path.name != config.tar_filename
        for path in config.raw_img_dir.rglob("*")
    )


def ensure_dataset_downloaded(config: DatasetConfig) -> None:
    ensure_dataset_dirs(config)

    if not config.download_from_gcs:
        return

    authenticate_colab_user()

    if not config.tar_local.is_file() or config.tar_local.stat().st_size == 0:
        print("Downloading images...")
        _download_from_gcs(
            config.gcs_images_tar,
            config.tar_local)
        if not config.tar_local.is_file() or config.tar_local.stat().st_size == 0:
            raise FileNotFoundError(
                f"No se pudo descargar {config.tar_local}. "
                "Instalá crcmod y revisá credenciales de GCS."
            )
        print("Images downloaded")

    if config.extract_images and not _has_extracted_images(config):
        print("Extracting images...")
        with tarfile.open(config.tar_local, "r:gz") as tar:
            tar.extractall(config.raw_img_dir)
        print(f"Images extracted to {config.raw_img_dir}")

    if not config.csv_main.is_file() or config.csv_main.stat().st_size == 0:
        print("Downloading CSV...")
        _download_from_gcs(
            config.gcs_data_csv,
            config.csv_main)
        print("CSV downloaded")


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


def load_filtered_dataframe(config: DatasetConfig) -> pd.DataFrame:
    ds_raw = load_raw_dataframe(config)
    missing_columns = [column for column in config.filter_columns if column not in ds_raw.columns]
    if missing_columns:
        raise KeyError(f"Faltan columnas para el filtrado: {missing_columns}")

    return ds_raw[ds_raw[list(config.filter_columns)].eq(1).any(axis=1)].copy()


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


def build_dataset(
    config: DatasetConfig | None = None,
    reduced: bool = False,
    sample_fraction: float = 0.10,
    sample_seed: int | None = 42,
) -> dict[str, object]:
    config = config or DatasetConfig()
    ds_raw = load_raw_dataframe(config)
    missing_columns = [column for column in config.filter_columns if column not in ds_raw.columns]
    if missing_columns:
        raise KeyError(f"Faltan columnas para el filtrado: {missing_columns}")

    ds_raw = add_cls_column(ds_raw, config)
    ds = ds_raw[ds_raw[list(config.filter_columns)].eq(1).any(axis=1)].copy()
    if reduced:
        ds = _sample_dataframe(ds, sample_fraction, sample_seed)

    return {
        "root": config.root_dir,
        "data_dir": config.data_dir,
        "raw_img_dir": config.raw_img_dir,
        "raw_csv_dir": config.raw_csv_dir,
        "csv_main": config.csv_main,
        "ds_raw": ds_raw,
        "ds": ds,
        "reduced": reduced,
        "sample_fraction": sample_fraction if reduced else 1.0,
    }
