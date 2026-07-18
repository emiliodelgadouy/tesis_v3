"""Drivers de experimentos para los notebooks de entrenamiento.

Contiene la logica de una corrida completa (``run_training_experiment``),
extraida de los notebooks para que las celdas queden en una sola llamada.
"""

from __future__ import annotations

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
from src.modes import normalize_mode
from src.model_builder import ModelBuilder
from src.utils import (
    TrainingTimer,
    apply_probability_threshold,
    dispose_model_builder,
    logit_initial_bias,
    predict_probs_and_labels,
    release_gpu_memory,
    resample_train_for_patch,
    resolve_steps_per_execution,
    run_training_stage,
    threshold_recall_target,
    threshold_youden_j,
)


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
):
    mode = normalize_mode(mode)
    general = config["GENERAL"]
    training = config["TRAINING"]
    mil = config["MIL"]
    full_cfg = config.get("FULL") or {}
    patch_cfg = config.get("PATCH") or {}
    patch_hardneg_cfg = config.get("PATCH_HARDNEG") or {}

    # Parametros derivados de la config (antes se pasaban sueltos por argumento):
    bag_grid = full_cfg.get("BAG_GRID", (3, 3))
    bag_canvas_mode = full_cfg.get("BAG_CANVAS_MODE", "resize")
    bag_keras_tiling = mil["BAG_KERAS_TILING"]
    attention_dim = mil["ATTENTION_DIM"]
    attention_gated = mil["ATTENTION_GATED"]
    patch_resize_to_bag_canvas = patch_cfg.get("RESIZE_TO_BAG_CANVAS", True)
    # Solo patch_hardneg puede alinearse a la grilla del bag.
    patch_align_to_bag_grid = (
        patch_hardneg_cfg.get("ALIGN_TO_BAG_GRID", False)
        if mode == "patch_hardneg"
        else False
    )
    # FULL = misma escala que el canvas ABMIL: BAG_GRID * tamaño nativo del backbone.
    # FULL["INPUT_SIZE"] queda como override opcional (p.ej. pruebas puntuales).
    if mode == "full":
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
    ) = result_builder = None
    release_gpu_memory(clear_keras_session=pretrained_builder is None)

    exp_name = f"{mode}_{backbone_name}"
    is_mil_run = mode in ("abmil", "abmil_patch_hardneg")
    batch_size = mil["BATCH_SIZE"] if is_mil_run else general["BATCH_SIZE"]
    # Default: cache on en simple/patch; off en full/abmil (canvases grandes, >100 GB posibles).
    cache_dataset = general.get(
        "CACHE_DATASET",
        mode not in ("abmil", "abmil_patch_hardneg", "full"),
    )

    # Modos PATCH: se remuestrea train (pos + hard/random neg) segun el ratio de la config,
    # y de ese remuestreo salen focal_alpha e initial_bias. El resto usa el train tal cual.
    if mode in ("patch", "patch_hardneg"):
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

        dataset_provider = build_dataset_provider(
            config,
            input_size,
            batch_size,
            lateralize=True,
            mode=mode,
            patch_align_to_bag_grid=patch_align_to_bag_grid,
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
        if mode in ("patch", "patch_hardneg"):
            run_config["PATCH_ALIGN_TO_BAG_GRID"] = patch_align_to_bag_grid
            run_config["PATCH_RESIZE_TO_BAG_CANVAS"] = patch_resize_to_bag_canvas
            run_config["PATCH_EVAL_USES_ROI_ORACLE"] = True
            if patch_align_to_bag_grid or patch_resize_to_bag_canvas:
                run_config["BAG_GRID"] = list(bag_grid)
                run_config["BAG_CANVAS_MODE"] = bag_canvas_mode
        if mode in ("abmil", "abmil_patch_hardneg"):
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
        if pretrained_builder is not None:
            pretrain_best = pretrained_builder.load_best_global_checkpoint()
            pretraining_mode = getattr(
                pretrained_builder, "pretraining_mode", pretrained_builder.model_name
            )
            run_config["PRETRAINED_FROM"] = f"{pretraining_mode}_{backbone_name}"
            run_config["PRETRAINED_BEST_VAL_METRIC"] = pretrain_best["value"]
            run_config["PRETRAINED_FROZEN_STAGE1"] = ["backbone", "dense", "output"]

        steps_per_execution = resolve_steps_per_execution(
            len(train_df),
            batch_size,
            max_steps=32,
        )
        run_config["STEPS_PER_EXECUTION"] = steps_per_execution

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
        )
        model = builder.build()

        experiment = start_training_experiment(
            config, experiment_name=exp_name, run_config=run_config, model=model.model
        )
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

        log_test_results(
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
        del dataset_provider, train_ds, val_ds, ds_test

        if keep_builder:
            release_gpu_memory(clear_keras_session=False)
        else:
            del model, builder, backbone
            dispose_model_builder(pretrained_builder)
            release_gpu_memory(clear_keras_session=True)

    return result_builder
