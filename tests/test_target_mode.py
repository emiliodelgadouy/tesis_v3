import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.target_ab import (
    apply_patient_split_manifest,
    assert_disjoint_patients,
    build_patient_split_manifest,
    load_patient_split_manifest,
    save_patient_split_manifest,
    undersample_training_negatives,
)
from src.target_preflight import validate_ab_preflight
from src.targets import (
    resolve_target_mode,
    source_frame_name,
    target_masks,
    validate_birads_labels,
)


def _binary_table(frame: pd.DataFrame, target_mode: str) -> pd.DataFrame:
    """Replica el contrato de etiquetado/dedup sin importar TensorFlow."""
    positive_mask, negative_mask = target_masks(frame, target_mode)
    keys = ["patient_id", "image_id"]
    positive = frame.loc[positive_mask].drop_duplicates(keys)
    negative = frame.loc[negative_mask].drop_duplicates(keys)
    negative = (
        negative.merge(positive[keys], on=keys, how="left", indicator=True)
        .query("_merge == 'left_only'")
        .drop(columns="_merge")
    )
    positive = positive.copy()
    negative = negative.copy()
    positive["cls"] = 1.0
    negative["cls"] = 0.0
    return pd.concat([positive, negative], ignore_index=True)


def _rows():
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4", "p5"],
            "image_id": ["i1", "i2", "i3", "i4", "i5"],
            "breast_birads": [
                "BI-RADS 1",
                "BI-RADS 2",
                "BI-RADS 3",
                "BI-RADS 4",
                "BI-RADS 5",
            ],
            "Mass": [0, 0, 1, 0, 1],
            "cls": [0, 1, 1, 0, 1],
        }
    )


class TargetModeTest(unittest.TestCase):
    def test_birads_mapping_one_through_five(self):
        frame = _rows()
        positive, negative = target_masks(frame, "birads_recall")
        self.assertEqual(positive.tolist(), [False, False, True, True, True])
        self.assertEqual(negative.tolist(), [True, True, False, False, False])

    def test_rejects_null_invalid_and_inconsistent_birads(self):
        frame = _rows()
        frame.loc[0, "breast_birads"] = None
        with self.assertRaisesRegex(ValueError, "nulls=1"):
            validate_birads_labels(frame)

        frame = _rows()
        frame.loc[0, "breast_birads"] = "BI-RADS 0"
        with self.assertRaisesRegex(ValueError, "BI-RADS 0"):
            validate_birads_labels(frame)

        frame = pd.concat([_rows().iloc[[0]], _rows().iloc[[0]]], ignore_index=True)
        frame.loc[1, "breast_birads"] = "BI-RADS 2"
        with self.assertRaisesRegex(ValueError, "inconsistente"):
            validate_birads_labels(frame)

    def test_legacy_mass_and_full_masks_do_not_change(self):
        frame = _rows()
        mass_positive, mass_negative = target_masks(frame, "mass")
        full_positive, full_negative = target_masks(frame, "full")
        self.assertEqual(mass_positive.tolist(), [False, False, True, False, True])
        self.assertEqual(mass_negative.tolist(), [True, False, False, True, False])
        self.assertEqual(full_positive.tolist(), [False, True, True, False, True])
        self.assertEqual(full_negative.tolist(), [True, False, False, True, False])

    def test_target_mode_precedes_legacy_alias(self):
        config = {
            "GENERAL": {
                "TARGET_MODE": "birads-recall",
                "POSITIVE_MODE": "mass",
            }
        }
        self.assertEqual(resolve_target_mode(config), "birads_recall")
        self.assertEqual(
            resolve_target_mode({"GENERAL": {"POSITIVE_MODE": "full"}}), "full"
        )

    def test_source_frame_matches_legacy_and_birads_universes(self):
        self.assertEqual(source_frame_name("mass"), "ds")
        self.assertEqual(source_frame_name("full"), "ds")
        self.assertEqual(source_frame_name("birads_recall"), "ds_raw")

    def test_build_dataset_deduplicates_and_matches_legacy_filters(self):
        frame = _rows()
        duplicate = frame.iloc[[2]].copy()
        duplicate["breast_birads"] = "BI-RADS 3"
        frame = pd.concat([frame, duplicate], ignore_index=True)
        mass = _binary_table(frame, "mass")
        full = _binary_table(frame, "full")
        birads = _binary_table(frame, "birads_recall")
        self.assertEqual(len(mass), 4)
        self.assertEqual(sorted(mass.loc[mass["cls"].eq(1), "image_id"]), ["i3", "i5"])
        self.assertEqual(sorted(mass.loc[mass["cls"].eq(0), "image_id"]), ["i1", "i4"])
        self.assertEqual(len(full), 5)
        self.assertEqual(len(birads), 5)
        self.assertEqual(int(birads["cls"].sum()), 3)
        self.assertEqual(birads["image_id"].nunique(), 5)

        consistent_dup = pd.concat([_rows().iloc[[0]], _rows().iloc[[0]]], ignore_index=True)
        validate_birads_labels(consistent_dup)


class PatientManifestTest(unittest.TestCase):
    def _raw(self):
        return pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
                "split": [
                    "training",
                    "training",
                    "training",
                    "training",
                    "test",
                    "test",
                ],
            }
        )

    def _canonical(self):
        return {
            "meta": {"positive_mode": "mass"},
            "train": [{"patient_id": "p1", "image_id": "i1"}],
            "val": [{"patient_id": "p2", "image_id": "i2"}],
            "test": [{"patient_id": "p5", "image_id": "i5"}],
        }

    def test_manifest_preserves_and_extends_patient_assignments(self):
        first = build_patient_split_manifest(
            self._raw(), self._canonical(), validation_ratio=0.5, seed=42
        )
        second = build_patient_split_manifest(
            self._raw(), self._canonical(), validation_ratio=0.5, seed=42
        )
        self.assertEqual(first, second)
        assignments = {
            row["patient_id"]: row["split"] for row in first["patients"]
        }
        self.assertEqual(assignments["p1"], "train")
        self.assertEqual(assignments["p2"], "val")
        self.assertEqual(assignments["p5"], "test")
        self.assertEqual(assignments["p6"], "test")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            save_patient_split_manifest(first, path)
            self.assertEqual(load_patient_split_manifest(path), first)
            tampered = json.loads(path.read_text())
            tampered["patients"][0]["split"] = "test"
            path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, "Hash inválido"):
                load_patient_split_manifest(path)

    def test_apply_manifest_and_undersampling_are_patient_safe(self):
        manifest = build_patient_split_manifest(
            self._raw(), self._canonical(), validation_ratio=0.5, seed=42
        )
        rows = []
        for patient in self._raw()["patient_id"]:
            rows.append(
                {
                    "patient_id": patient,
                    "image_id": f"{patient}_1",
                    "cls": float(patient in {"p1", "p2", "p5"}),
                }
            )
        train, val, test = apply_patient_split_manifest(pd.DataFrame(rows), manifest)
        assert_disjoint_patients(train, val, test)

        train_many = pd.DataFrame(
            {
                "patient_id": [f"p{i}" for i in range(10)],
                "cls": [1.0, 1.0] + [0.0] * 8,
            }
        )
        sampled = undersample_training_negatives(
            train_many, max_negatives_per_positive=3, seed=42
        )
        self.assertEqual(int(sampled["cls"].eq(1).sum()), 2)
        self.assertEqual(int(sampled["cls"].eq(0).sum()), 6)


class PreflightTest(unittest.TestCase):
    def _labeled(self, directory: Path):
        rows = []
        patients = (
            ("p1", 1, "BI-RADS 4"),
            ("p2", 0, "BI-RADS 1"),
            ("p3", 1, "BI-RADS 5"),
            ("p4", 0, "BI-RADS 2"),
            ("p5", 1, "BI-RADS 3"),
            ("p6", 0, "BI-RADS 2"),
        )
        for index, (patient, label, birads) in enumerate(patients, start=1):
            path = directory / f"{patient}.png"
            path.write_bytes(b"png")
            rows.append(
                {
                    "patient_id": patient,
                    "image_id": f"i{index}",
                    "path": str(path),
                    "cls": float(label),
                    "breast_birads": birads,
                    "view": "CC" if label else "MLO",
                    "breast_density": "DENSITY A",
                    "Mass": int(label),
                    "No_Finding": int(not label),
                }
            )
        return pd.DataFrame(rows)

    def test_preflight_exports_counts_and_fails_on_leakage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            frame = self._labeled(directory)
            raw = frame.copy()
            train_natural = frame.iloc[[0, 1]].copy()
            train = train_natural.copy()
            val = frame.iloc[[2, 3]].copy()
            test = frame.iloc[[4, 5]].copy()
            output = directory / "ok"
            result = validate_ab_preflight(
                raw_df=raw,
                train_natural=train_natural,
                train_effective=train,
                val=val,
                test=test,
                output_dir=output,
                target_mode="mass",
                require_files=True,
            )
            self.assertEqual(result["birads_invalid_or_null"], 0)
            self.assertIn("natural_prevalence", result)
            self.assertTrue((output / "preflight_counts.csv").is_file())
            self.assertTrue((output / "preflight_distributions.csv").is_file())

            leaked_val = pd.concat([val, train.iloc[[0]]], ignore_index=True)
            with self.assertRaisesRegex(ValueError, "Leakage"):
                validate_ab_preflight(
                    raw_df=raw,
                    train_natural=train_natural,
                    train_effective=train,
                    val=leaked_val,
                    test=test,
                    output_dir=directory / "leak",
                    target_mode="mass",
                    require_files=False,
                )

            empty_test = test.iloc[0:0]
            with self.assertRaisesRegex(ValueError, "vacío"):
                validate_ab_preflight(
                    raw_df=raw,
                    train_natural=train_natural,
                    train_effective=train,
                    val=val,
                    test=empty_test,
                    output_dir=directory / "empty",
                    target_mode="mass",
                    require_files=False,
                )

            uniclase = test.copy()
            uniclase["cls"] = 0.0
            with self.assertRaisesRegex(ValueError, "ambas clases"):
                validate_ab_preflight(
                    raw_df=raw,
                    train_natural=train_natural,
                    train_effective=train,
                    val=val,
                    test=uniclase,
                    output_dir=directory / "uni",
                    target_mode="mass",
                    require_files=False,
                )

            missing = test.copy()
            missing.loc[missing.index[0], "path"] = str(directory / "missing.png")
            with self.assertRaises(FileNotFoundError):
                validate_ab_preflight(
                    raw_df=raw,
                    train_natural=train_natural,
                    train_effective=train,
                    val=val,
                    test=missing,
                    output_dir=directory / "files",
                    target_mode="mass",
                    require_files=True,
                )

            raw_invalid = raw.copy()
            raw_invalid.loc[raw_invalid.index[0], "breast_birads"] = "BI-RADS 0"
            with self.assertRaisesRegex(ValueError, "BI-RADS 0"):
                validate_ab_preflight(
                    raw_df=raw_invalid,
                    train_natural=train_natural,
                    train_effective=train,
                    val=val,
                    test=test,
                    output_dir=directory / "birads",
                    target_mode="birads_recall",
                    require_files=False,
                )


if __name__ == "__main__":
    unittest.main()
