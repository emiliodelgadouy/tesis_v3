"""Fachada de imports para los notebooks de entrenamiento.

Reexporta en un unico lugar todo lo que los notebooks (abmil, full, multirun)
usan de `src`, para que la celda de imports quede en una linea:

    from src.notebook_api import *

Llamar a ``configure_notebook()`` antes de importar este modulo (TensorFlow se
carga al importar la fachada). Comet ML se importa aqui por si la fachada se
usa sin pasar por ``configure_notebook()``.
"""

from __future__ import annotations

import comet_ml  # noqa: F401 - antes de TensorFlow

import numpy as np
import pandas as pd
import tensorflow as tf

from src.backbones import resolve_backbone
from src.comet_logging import (
    CometEpochLogger,
    log_dataset_class_counts,
    log_deterministic_input_samples,
    log_keras_eval_metrics,
    log_stage_timing,
    log_test_results,
    log_training_timing_summary,
    login_comet,
    start_training_experiment,
)
from src.dataset import download_and_build_dataset
from src.dataset_provider import as_tf_dataset, build_dataset_provider, hard_negatives_from_positives
from src.experiment import run_training_experiment
from src.imports import set_random_seeds
from src.modes import normalize_mode
from src.model_builder import ModelBuilder
from src.utils import (
    TrainingTimer,
    apply_probability_threshold,
    build_positive_negative_dataset,
    deduplicate_images,
    dispose_model_builder,
    get_dataset_splits,
    logit_initial_bias,
    predict_probs_and_labels,
    prepare_dataset,
    release_gpu_memory,
    resample_train_for_patch,
    load_dataset_splits,
    resolve_steps_per_execution,
    run_training_stage,
    save_dataset_splits,
    stratified_train_val_test_split,
    threshold_best_f1,
    threshold_recall_target,
    threshold_youden_j,
    undersample_negatives,
    warmup,
)

__all__ = [
    # third-party
    "np",
    "pd",
    "tf",
    # backbones
    "resolve_backbone",
    # comet_logging
    "CometEpochLogger",
    "log_dataset_class_counts",
    "log_deterministic_input_samples",
    "log_keras_eval_metrics",
    "log_stage_timing",
    "log_test_results",
    "log_training_timing_summary",
    "login_comet",
    "start_training_experiment",
    # dataset / dataset_provider
    "download_and_build_dataset",
    "as_tf_dataset",
    "build_dataset_provider",
    "hard_negatives_from_positives",
    # experiment
    "run_training_experiment",
    # imports
    "set_random_seeds",
    "normalize_mode",
    # model_builder
    "ModelBuilder",
    # utils
    "TrainingTimer",
    "apply_probability_threshold",
    "build_positive_negative_dataset",
    "deduplicate_images",
    "dispose_model_builder",
    "get_dataset_splits",
    "logit_initial_bias",
    "predict_probs_and_labels",
    "prepare_dataset",
    "release_gpu_memory",
    "resample_train_for_patch",
    "load_dataset_splits",
    "resolve_steps_per_execution",
    "run_training_stage",
    "save_dataset_splits",
    "stratified_train_val_test_split",
    "threshold_best_f1",
    "threshold_recall_target",
    "threshold_youden_j",
    "undersample_negatives",
    "warmup",
]
