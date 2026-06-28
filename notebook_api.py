"""Fachada de imports para los notebooks de entrenamiento.

Reexporta en un unico lugar todo lo que los notebooks (abmil, abmil_multiple,
clam, full) usan de `src`, para que la celda de imports quede en una linea:

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

from src.backbone_provider import (
    format_all_architecture_descriptions,
    get_backbone_description,
    resolve_backbone,
)
from src.comet_logging import (
    CometEpochLogger,
    log_dataset_class_counts,
    log_deterministic_input_samples,
    log_keras_eval_metrics,
    log_stage_timing,
    log_test_results,
    log_training_timing_summary,
    login,
    start_training_experiment,
)
from src.dataset import build_dataset
from src.dataset_provider import as_tf_dataset, build_dataset_provider
from src.imports import get_device, set_seed
from src.model_builder import ModelBuilder
from src.plots import plot_history
from src.utils import (
    TrainingTimer,
    apply_probability_threshold,
    binary_report,
    deduplicate_images,
    labels_from_tf_dataset,
    logit_initial_bias,
    plot_binary_confusion_matrix,
    predict_probabilities,
    predict_probabilities_tta,
    run_training_stage,
    stratified_split,
    threshold_best_f1,
    threshold_max_recall,
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
    # backbone_provider
    "format_all_architecture_descriptions",
    "get_backbone_description",
    "resolve_backbone",
    # comet_logging
    "CometEpochLogger",
    "log_dataset_class_counts",
    "log_deterministic_input_samples",
    "log_keras_eval_metrics",
    "log_stage_timing",
    "log_test_results",
    "log_training_timing_summary",
    "login",
    "start_training_experiment",
    # dataset / dataset_provider
    "build_dataset",
    "as_tf_dataset",
    "build_dataset_provider",
    # imports
    "get_device",
    "set_seed",
    # model_builder
    "ModelBuilder",
    # plots
    "plot_history",
    # utils
    "TrainingTimer",
    "apply_probability_threshold",
    "binary_report",
    "deduplicate_images",
    "labels_from_tf_dataset",
    "logit_initial_bias",
    "plot_binary_confusion_matrix",
    "predict_probabilities",
    "predict_probabilities_tta",
    "run_training_stage",
    "stratified_split",
    "threshold_best_f1",
    "threshold_max_recall",
    "threshold_recall_target",
    "threshold_youden_j",
    "undersample_negatives",
    "warmup",
]
