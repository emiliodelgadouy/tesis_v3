"""Drivers de experimentos para los notebooks de entrenamiento.

Contiene la logica de una corrida completa (``run_training_experiment``),
extraida de los notebooks para que las celdas queden en una sola llamada.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.backbones import get_backbone, resolve_backbone
from src.comet_logging import (
    log_dataset_class_counts,
    log_deterministic_input_samples,
    log_keras_eval_metrics,
    log_test_results,
    log_training_timing_summary,
    start_training_experiment,
)
from src.dataset_provider import build_dataset_provider
from src.modes import is_mil_mode, normalize_mode
from src.model_builder import ModelBuilder
from src.utils import (
    TrainingTimer,
    apply_probability_threshold,
    dispose_model_builder,
    expand_grid_patch_table,
    logit_initial_bias,
    predict_probs_and_labels,
    release_gpu_memory,
    resample_train_for_patch,
    resolve_steps_per_execution,
    run_training_stage,
    threshold_recall_target,
    threshold_youden_j,
)


def _bootstrap_auc_intervals(
    y_true,
    y_prob,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    """Intervalos bootstrap estratificados para ROC-AUC y PR-AUC."""
    if samples <= 0:
        return {}
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    if len(positive) == 0 or len(negative) == 0:
        return {}
    rng = np.random.default_rng(seed)
    roc_values = np.empty(samples, dtype=np.float64)
    pr_values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        roc_values[index] = roc_auc_score(y_true[sampled], y_prob[sampled])
        pr_values[index] = average_precision_score(y_true[sampled], y_prob[sampled])
    return {
        "test_roc_auc_ci_low": float(np.quantile(roc_values, 0.025)),
        "test_roc_auc_ci_high": float(np.quantile(roc_values, 0.975)),
        "test_pr_auc_ci_low": float(np.quantile(pr_values, 0.025)),
        "test_pr_auc_ci_high": float(np.quantile(pr_values, 0.975)),
    }


def _export_predictions(
    exp_name: str,
    split_name: str,
    table,
    y_true,
    y_prob,
) -> Path:
    """Guarda predicciones alineadas para comparaciones pareadas sin reentrenar."""
    export_dir = Path("exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        column
        for column in (
            "patient_id",
            "image_id",
            "path",
            "view",
            "laterality",
            "cls",
            "_bag_cls",
            "_patch_tile_index",
            "_roi_tile_overlap",
        )
        if column in table.columns
    ]
    predictions = table[columns].reset_index(drop=True).copy()
    if len(predictions) != len(y_true):
        raise RuntimeError(
            f"No se pueden exportar predicciones de {exp_name}: "
            f"tabla={len(predictions)} vs predicciones={len(y_true)}"
        )
    predictions["y_true"] = np.asarray(y_true, dtype=np.int64)
    predictions["y_prob"] = np.asarray(y_prob, dtype=np.float64)
    path = export_dir / f"{exp_name}_{split_name}_predictions.csv"
    predictions.to_csv(path, index=False)
    return path


def run_training_experiment(
    config,
    mode,
    backbone_name,
    train_df,
    val_df,
    test_df,
    *,
    pretrained_builder=None,
    return_builder=False,
    return_summary=False,
    experiment_suffix=None,
    dispose_pretrained_builder=True,
    branch_checkpoints=None,
):
    mode = normalize_mode(mode)
    general = config["GENERAL"]
    training = config["TRAINING"]
    mil = config["MIL"]
    full_cfg = config.get("FULL") or {}
    patch_cfg = config.get("PATCH") or {}
    patch_hardneg_cfg = config.get("PATCH_HARDNEG") or {}
    patch_alltiles_cfg = config.get("PATCH_ALLTILES") or {}
    multibranch_cfg = config.get("MULTIBRANCH") or {}

    # Parametros derivados de la config (antes se pasaban sueltos por argumento):
    bag_grid = full_cfg.get("BAG_GRID", (3, 3))
    bag_canvas_mode = full_cfg.get("BAG_CANVAS_MODE", "resize")
    bag_keras_tiling = mil["BAG_KERAS_TILING"]
    attention_dim = mil["ATTENTION_DIM"]
    attention_gated = mil["ATTENTION_GATED"]
    patch_resize_to_bag_canvas = patch_cfg.get("RESIZE_TO_BAG_CANVAS", True)
    # patch_hardneg puede alinear su muestreo; patch_alltiles fuerza el tile exacto
    # mediante ``patch_tile_index_column`` mas abajo.
    patch_align_to_bag_grid = (
        patch_hardneg_cfg.get("ALIGN_TO_BAG_GRID", False)
        if mode == "patch_hardneg"
        else False
    )
    patch_tile_index_column = "_patch_tile_index" if mode == "patch_alltiles" else None
    # FULL = misma escala que el canvas ABMIL: BAG_GRID * tamaño nativo del backbone.
    # FULL["INPUT_SIZE"] queda como override opcional (p.ej. pruebas puntuales).
    if mode in ("full", "multibranch"):
        full_override = full_cfg.get("INPUT_SIZE")
        if full_override is not None:
            input_size = tuple(full_override)
        else:
            native_h, native_w = get_backbone(backbone_name).input_size
            input_size = (bag_grid[0] * native_h, bag_grid[1] * native_w)
    else:
        input_size = None

    experiment = model = builder = backbone = dataset_provider = train_ds = val_ds = (
        ds_test
    ) = result_builder = summary = pretrain_best = None
    tile_backbone = tile_preprocess_input = tile_size = None
    source_full_builder = source_abmil_builder = None
    transfer_diagnostics = {}
    release_gpu_memory(clear_keras_session=pretrained_builder is None)

    exp_name = f"{mode}_{backbone_name}"
    if experiment_suffix:
        exp_name = f"{exp_name}_{experiment_suffix}"
    is_mil_run = is_mil_mode(mode)
    is_multibranch_run = mode == "multibranch"
    if branch_checkpoints is not None and not is_multibranch_run:
        raise ValueError("branch_checkpoints solo se admite en modo multibranch")
    if is_multibranch_run:
        batch_size = int(multibranch_cfg.get("BATCH_SIZE", 16))
    else:
        batch_size = mil["BATCH_SIZE"] if is_mil_run else general["BATCH_SIZE"]
    # Default: cache on en simple/patch; off en full/abmil (canvases grandes, >100 GB posibles).
    cache_dataset = general.get(
        "CACHE_DATASET",
        not is_mil_run and mode not in ("full", "multibranch"),
    )

    # Modos PATCH: se remuestrea train (pos + hard/random neg) segun el ratio de la config,
    # y de ese remuestreo salen focal_alpha e initial_bias. El resto usa el train tal cual.
    if mode == "patch_alltiles":
        train_df = expand_grid_patch_table(config, train_df, training=True)
        val_df = expand_grid_patch_table(config, val_df, training=False)
        test_df = expand_grid_patch_table(config, test_df, training=False)
        n_pos = int((train_df["cls"] == 1).sum())
        n_neg = int((train_df["cls"] == 0).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"patch_alltiles necesita ambas clases de tile; recibido pos={n_pos}, neg={n_neg}"
            )
        focal_alpha = n_neg / len(train_df)
        bias = logit_initial_bias(n_pos, n_neg)
    elif mode in ("patch", "patch_hardneg"):
        patch_ratio = (
            config["PATCH_HARDNEG"]
            if mode == "patch_hardneg"
            else config["PATCH"]
        )
        train_df, focal_alpha, bias = resample_train_for_patch(
            train_df, patch_ratio, general["RANDOM_SEED"]
        )
    else:
        if len(train_df) == 0:
            raise ValueError("train_df esta vacio; no se puede calcular focal_alpha/bias")
        n_pos = int((train_df["cls"] == 1).sum())
        n_neg = int((train_df["cls"] == 0).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"train_df necesita ambas clases; recibido pos={n_pos}, neg={n_neg}"
            )
        focal_alpha = n_neg / len(train_df)
        bias = logit_initial_bias(n_pos, n_neg)

    try:
        if pretrained_builder is not None:
            input_size = pretrained_builder.IMG_SIZE
            backbone = pretrained_builder.backbone
            preprocess_input = pretrained_builder.preprocess_input
        else:
            backbone, preprocess_input, input_size = resolve_backbone(
                backbone_name, input_size=input_size
            )
            if is_multibranch_run:
                tile_backbone, tile_preprocess_input, tile_size = resolve_backbone(
                    backbone_name
                )

        if branch_checkpoints is not None:
            required = {"full", "abmil"}
            missing = required.difference(branch_checkpoints)
            if missing:
                raise ValueError(
                    f"Faltan checkpoints para el transfer multibranch: {sorted(missing)}"
                )
            checkpoint_paths = {
                name: Path(branch_checkpoints[name]) for name in sorted(required)
            }
            missing_files = [
                str(path) for path in checkpoint_paths.values() if not path.is_file()
            ]
            if missing_files:
                raise FileNotFoundError(
                    "No se encontraron checkpoints de ramas: " + ", ".join(missing_files)
                )

            source_full_backbone, source_full_preprocess, source_full_size = (
                resolve_backbone(backbone_name, input_size=input_size)
            )
            source_full_builder = ModelBuilder(
                config,
                source_full_size,
                source_full_backbone,
                source_full_preprocess,
                mode="full",
                initial_bias=bias,
                focal_alpha=focal_alpha,
                lateralized_inputs=True,
                steps_per_execution=1,
            )
            source_full_builder.build()
            source_full_builder.model.load_weights(str(checkpoint_paths["full"]))

            source_abmil_backbone, source_abmil_preprocess, source_abmil_size = (
                resolve_backbone(backbone_name)
            )
            source_abmil_builder = ModelBuilder(
                config,
                source_abmil_size,
                source_abmil_backbone,
                source_abmil_preprocess,
                mode="abmil",
                initial_bias=bias,
                focal_alpha=focal_alpha,
                lateralized_inputs=True,
                steps_per_execution=1,
            )
            source_abmil_builder.build()
            source_abmil_builder.model.load_weights(str(checkpoint_paths["abmil"]))

        dataset_provider = build_dataset_provider(
            config,
            input_size,
            batch_size,
            lateralize=True,
            mode=mode,
            patch_align_to_bag_grid=patch_align_to_bag_grid,
            patch_tile_index_column=patch_tile_index_column,
            cache_dataset=cache_dataset,
        )
        train_ds, val_ds, ds_test = dataset_provider.build_splits(
            train_df, val_df, test_df
        )

        run_config = {
            **general,
            **training,
            "BACKBONE_ARCHITECTURE": backbone_name,
            "MODE": mode,
            "INPUT_SIZE": list(input_size),
            "TRAIN_ROWS": len(train_df),
            "BATCH_SIZE": batch_size,
            "CACHE_DATASET": cache_dataset,
            "JIT_COMPILE": True,
        }
        if mode in ("patch", "patch_hardneg", "patch_alltiles"):
            run_config["PATCH_ALIGN_TO_BAG_GRID"] = patch_align_to_bag_grid
            run_config["PATCH_RESIZE_TO_BAG_CANVAS"] = patch_resize_to_bag_canvas
            run_config["PATCH_EVAL_USES_ROI_ORACLE"] = mode != "patch_alltiles"
            if mode == "patch_alltiles":
                run_config["PATCH_EXACT_BAG_GRID_TILES"] = True
                run_config["PATCH_ALLTILES_EXHAUSTIVE_EVAL"] = True
                run_config["PATCH_ALLTILES_MIN_ROI_TILE_OVERLAP"] = patch_alltiles_cfg.get(
                    "MIN_ROI_TILE_OVERLAP", 0.0
                )
                run_config["PATCH_ALLTILES_HARD_NEG_RATIO"] = patch_alltiles_cfg.get(
                    "HARD_NEGATIVE_TO_POSITIVE_RATIO", 4.0
                )
                run_config["PATCH_ALLTILES_RANDOM_NEG_RATIO"] = patch_alltiles_cfg.get(
                    "RANDOM_NEGATIVE_TO_POSITIVE_RATIO", 4.0
                )
            if mode == "patch_alltiles" or patch_align_to_bag_grid or patch_resize_to_bag_canvas:
                run_config["BAG_GRID"] = list(bag_grid)
                run_config["BAG_CANVAS_MODE"] = bag_canvas_mode
        if is_mil_run:
            run_config["BAG_GRID"] = list(bag_grid)
            run_config["BAG_KERAS_TILING"] = bag_keras_tiling
            run_config["BAG_CANVAS_MODE"] = bag_canvas_mode
            run_config["BAG_CANVAS_SIZE"] = [
                bag_grid[0] * input_size[0],
                bag_grid[1] * input_size[1],
            ]
            run_config["BAG_INSTANCES"] = bag_grid[0] * bag_grid[1]
            run_config["ATTENTION_DIM"] = attention_dim
            run_config["ATTENTION_GATED"] = attention_gated
            run_config["BAG_SIZE"] = bag_grid[0] * bag_grid[1]
            if mode in ("abmil_patch_hardneg_guided", "abmil_patch_alltiles_gated"):
                run_config["GUIDED_ATTENTION_SOURCE"] = "patch_instance_logits"
                run_config["GUIDED_ATTENTION_TEMPERATURE"] = mil.get("GUIDED_ATTENTION_TEMPERATURE", 1.0)
                run_config["GUIDED_ATTENTION_STRENGTH"] = mil.get("GUIDED_ATTENTION_STRENGTH", 1.0)
            if mode == "abmil_patch_alltiles_gated":
                run_config["GUIDED_ATTENTION_INITIAL_GATE"] = mil.get(
                    "GUIDED_ATTENTION_INITIAL_GATE", 0.05
                )
            if mode == "abmil_patch_alltiles_score":
                run_config["PATCH_SCORE_USAGE"] = "concatenated_instance_feature"
        if is_multibranch_run:
            run_config["BAG_GRID"] = list(bag_grid)
            run_config["BAG_CANVAS_MODE"] = bag_canvas_mode
            run_config["TILE_SIZE"] = list(tile_size)
            run_config["BAG_INSTANCES"] = bag_grid[0] * bag_grid[1]
            run_config["ATTENTION_DIM"] = attention_dim
            run_config["ATTENTION_GATED"] = attention_gated
            run_config["FUSION_DIM"] = int(multibranch_cfg.get("FUSION_DIM", 128))
            run_config["BRANCHES"] = ["full", "abmil"]
            if branch_checkpoints is not None:
                run_config["BRANCH_INITIALIZATION"] = "standalone_checkpoints"
                run_config["FULL_SOURCE_CHECKPOINT"] = str(
                    checkpoint_paths["full"]
                )
                run_config["ABMIL_SOURCE_CHECKPOINT"] = str(
                    checkpoint_paths["abmil"]
                )
        if pretrained_builder is not None:
            pretrain_best = pretrained_builder.load_best_global_checkpoint()
            pretraining_mode = getattr(
                pretrained_builder, "pretraining_mode", pretrained_builder.model_name
            )
            run_config["PRETRAINED_FROM"] = f"{pretraining_mode}_{backbone_name}"
            run_config["PRETRAINED_BEST_VAL_METRIC"] = pretrain_best["value"]
            run_config["PRETRAINED_FROZEN_STAGE1"] = ["backbone", "dense"]
            if mode in (
                "abmil_patch_hardneg_guided",
                "abmil_patch_alltiles_gated",
                "abmil_patch_alltiles_score",
            ):
                run_config["PRETRAINED_FROZEN_STAGE1"].append("instance_output")
            run_config["BAG_OUTPUT_INITIALIZATION"] = "new"

        steps_per_execution = resolve_steps_per_execution(
            len(train_df),
            batch_size,
            max_steps=32,
        )
        run_config["STEPS_PER_EXECUTION"] = steps_per_execution

        builder_kwargs = {}
        if is_multibranch_run:
            builder_kwargs.update(
                tile_backbone=tile_backbone,
                tile_preprocess_input=tile_preprocess_input,
                tile_size=tile_size,
                pretrained_full_builder=source_full_builder,
                pretrained_abmil_builder=source_abmil_builder,
            )
        builder = ModelBuilder(
            config,
            input_size,
            backbone,
            preprocess_input,
            mode=mode,
            initial_bias=bias,
            focal_alpha=focal_alpha,
            pretrained_builder=pretrained_builder,
            checkpoint_prefix=exp_name,
            lateralized_inputs=True,
            steps_per_execution=steps_per_execution,
            **builder_kwargs,
        )
        model = builder.build()
        if hasattr(model, "branch_transfer_diagnostics"):
            transfer_diagnostics = model.branch_transfer_diagnostics()
        dispose_model_builder(source_full_builder)
        dispose_model_builder(source_abmil_builder)
        source_full_builder = source_abmil_builder = None

        experiment = start_training_experiment(
            config, experiment_name=exp_name, run_config=run_config, model=model.model
        )
        if transfer_diagnostics:
            experiment.log_metrics(transfer_diagnostics)
        log_dataset_class_counts(experiment, train=train_df, val=val_df, test=test_df)
        log_deterministic_input_samples(
            config, experiment, train_ds, experiment_name=exp_name, split_name="train"
        )

        training_timer = TrainingTimer()
        training_timer.start_training()
        epoch_offset = 0

        history_frozen, _, _ = run_training_stage(
            config,
            model,
            train_ds,
            val_ds,
            stage=1,
            training_timer=training_timer,
            epoch_offset=epoch_offset,
            experiment=experiment,
        )
        epoch_offset += len(history_frozen.history.get("loss", []))

        history_partial, _, _ = run_training_stage(
            config,
            model,
            train_ds,
            val_ds,
            stage=2,
            training_timer=training_timer,
            epoch_offset=epoch_offset,
            experiment=experiment,
            setup_fn=lambda: model.make_backbone_partially_trainable(
                trainable_fraction=training["BACKBONE_TRAINABLE_FRACTION"],
                learning_rate=1e-4,
            ),
        )
        epoch_offset += len(history_partial.history.get("loss", []))

        run_training_stage(
            config,
            model,
            train_ds,
            val_ds,
            stage=3,
            training_timer=training_timer,
            epoch_offset=epoch_offset,
            experiment=experiment,
            setup_fn=lambda: model.make_backbone_trainable(
                trainable=True, learning_rate=1e-5
            ),
        )
        log_training_timing_summary(experiment, training_timer)
        best_global_checkpoint = model.load_best_global_checkpoint()
        guide_gate_final = None
        if hasattr(model, "guide_gate_value"):
            guide_gate_final = float(model.guide_gate_value())
            experiment.log_metric("guide_gate_final", guide_gate_final)
        branch_diagnostics = dict(transfer_diagnostics)
        if hasattr(model, "branch_fusion_weight_norms"):
            fusion_diagnostics = model.branch_fusion_weight_norms()
            branch_diagnostics.update(fusion_diagnostics)
            experiment.log_metrics(fusion_diagnostics)

        log_keras_eval_metrics(
            experiment,
            model,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=ds_test,
        )

        # Una sola pasada por split (labels + probabilidades juntas) en vez de
        # recorrer el pipeline tf.data dos veces (labels_from_tf_dataset + predict).
        y_train_true, y_train_prob = predict_probs_and_labels(
            model, train_ds.ordered()
        )
        y_val_true, y_val_prob = predict_probs_and_labels(model, val_ds)
        y_test_true, y_test_prob = predict_probs_and_labels(model, ds_test)

        evaluation_cfg = config.get("EVALUATION") or {}
        prediction_paths = {}
        if evaluation_cfg.get("EXPORT_PREDICTIONS", False):
            prediction_paths["val"] = _export_predictions(
                exp_name,
                "val",
                val_df,
                y_val_true,
                y_val_prob,
            )
            prediction_paths["test"] = _export_predictions(
                exp_name,
                "test",
                test_df,
                y_test_true,
                y_test_prob,
            )
        bootstrap_intervals = _bootstrap_auc_intervals(
            y_test_true,
            y_test_prob,
            samples=int(evaluation_cfg.get("BOOTSTRAP_SAMPLES", 0)),
            seed=int(general["RANDOM_SEED"]),
        )
        if bootstrap_intervals:
            experiment.log_metrics(bootstrap_intervals)

        thr_youden = threshold_youden_j(y_val_true, y_val_prob)
        thr_recall90 = threshold_recall_target(
            y_val_true, y_val_prob, target_recall=0.90
        )

        y_train_pred_default = apply_probability_threshold(
            y_train_prob, general["PROBABILITY_THRESHOLD"]
        )
        y_train_pred_youden = apply_probability_threshold(y_train_prob, thr_youden)
        y_val_pred_default = apply_probability_threshold(
            y_val_prob, general["PROBABILITY_THRESHOLD"]
        )
        y_val_pred_youden = apply_probability_threshold(y_val_prob, thr_youden)
        y_pred_default = apply_probability_threshold(
            y_test_prob, general["PROBABILITY_THRESHOLD"]
        )
        y_pred_youden = apply_probability_threshold(y_test_prob, thr_youden)

        comet_url = log_test_results(
            config,
            experiment,
            backbone_name=exp_name,
            y_test_true=y_test_true,
            y_test_prob=y_test_prob,
            y_pred_default=y_pred_default,
            y_pred_youden=y_pred_youden,
            y_train_true=y_train_true,
            y_train_prob=y_train_prob,
            y_train_pred_default=y_train_pred_default,
            y_train_pred_youden=y_train_pred_youden,
            y_val_true=y_val_true,
            y_val_prob=y_val_prob,
            y_val_pred_default=y_val_pred_default,
            y_val_pred_youden=y_val_pred_youden,
            thr_youden=thr_youden,
            thr_recall90=thr_recall90,
            best_val_metric=best_global_checkpoint["value"],
            final_weights_path=best_global_checkpoint["path"],
            show_plots=False,
        )
        experiment = (
            None  # ya cerrado por log_test_results; evita doble end() en finally
        )

        if return_summary:
            from sklearn.metrics import average_precision_score, roc_auc_score

            summary = {
                "experiment": exp_name,
                "mode": mode,
                "backbone": backbone_name,
                "input_size": list(input_size),
                "val_best_metric_name": str(best_global_checkpoint["monitor"]),
                "val_best_metric": float(best_global_checkpoint["value"]),
                "final_weights_file": str(best_global_checkpoint["path"]),
                "test_roc_auc": float(roc_auc_score(y_test_true, y_test_prob)),
                "test_pr_auc": float(average_precision_score(y_test_true, y_test_prob)),
                "thr_youden": float(thr_youden),
                "comet_url": comet_url,
                **bootstrap_intervals,
                **branch_diagnostics,
            }
            if str(best_global_checkpoint["monitor"]) == "val_auc":
                summary["val_best_auc"] = float(best_global_checkpoint["value"])
            if prediction_paths:
                summary["val_predictions_file"] = str(prediction_paths["val"])
                summary["test_predictions_file"] = str(
                    prediction_paths["test"]
                )
                # Alias legacy usado por notebooks anteriores.
                summary["predictions_file"] = str(prediction_paths["test"])
            if guide_gate_final is not None:
                summary["guide_gate_final"] = guide_gate_final
            if pretrain_best is not None:
                summary["pretrained_checkpoint"] = str(pretrain_best["path"])
                summary["pretrained_best_metric"] = float(pretrain_best["value"])

        if return_builder:
            model.pretraining_mode = mode
            result_builder = model
    except Exception as exc:
        import traceback

        print(f"\n[FALLO] {backbone_name}/{mode}: {exc}")
        traceback.print_exc()
        if experiment is not None:
            try:
                experiment.end()
            except Exception:
                pass
        raise
    finally:
        keep_builder = return_builder and result_builder is not None
        keep_pretrained_builder = pretrained_builder is not None and not dispose_pretrained_builder
        dispose_model_builder(source_full_builder)
        dispose_model_builder(source_abmil_builder)
        del dataset_provider, train_ds, val_ds, ds_test

        if keep_builder or keep_pretrained_builder:
            if not keep_builder:
                del model, builder, backbone
            release_gpu_memory(clear_keras_session=False)
        else:
            del model, builder, backbone
            dispose_model_builder(pretrained_builder)
            release_gpu_memory(clear_keras_session=True)

    if return_summary:
        return summary
    return result_builder
