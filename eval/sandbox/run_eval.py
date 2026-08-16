"""
eval/sandbox/run_eval.py

Corre la evaluación completa (split + linear probe + MLP probe + bootstrap
+ fairness) para cualquier combinación de modelo × tarea.

Uso:
    cd eval/sandbox

    # Una combinación específica
    python run_eval.py --task breast_molsub --model virchow2

    # Todas las tareas con un modelo
    python run_eval.py --model virchow2 --all-tasks

    # Todo
    python run_eval.py --all

    # MLP probe (default)
    python run_eval.py --all

    # Ambos probes
    python run_eval.py --all --probe both

    # Solo linear probe
    python run_eval.py --all --probe linear

    # Listar qué hay disponible
    python run_eval.py --list
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.config.datasets import TASKS, MODELS
from eval.evaluation.splitter import (
    split_train_val_test,
    cross_validate_small_dataset,
    should_use_cv,
    choose_eval_method,
    eval_method_reason,
)
from eval.evaluation.metrics import (
    compute_metrics, compute_confusion_matrix, compute_error_distribution,
    compute_auc_ba_diagnostics, k_consistency_check,
)
from eval.evaluation.bootstrap import bootstrap_metrics, bootstrap_fairness_gaps
from eval.evaluation.fairness import fairness_report, compute_fairness_summary, compute_error_analysis_by_subgroup, bin_age
from eval.evaluation.cross_dataset import cross_dataset_evaluate
from eval.models.linear_probe import stack_embeddings, fit_linear_probe
from eval.models.mlp_probe import fit_mlp_probe

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
N_BOOTSTRAP = 1000
MIN_GROUP_SIZE = 10
N_AUC_PERMUTATIONS = 1000


def _add_auc_ba_diagnostics(
    metrics: dict,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: list,
) -> dict:
    """Attach final-run AUC diagnostics to a persisted metric dictionary."""
    metrics["auc_ba_diagnostics"] = compute_auc_ba_diagnostics(
        y_true=y_true,
        y_proba=y_proba,
        balanced_accuracy=metrics.get("balanced_accuracy"),
        labels=labels,
        n_perm=N_AUC_PERMUTATIONS,
        seed=SEED,
    )
    return metrics


def _save_predictions_npz(
    out_dir: Path,
    stem: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list,
    uids: np.ndarray | None = None,
    extra: dict | None = None,
) -> Path:
    """
    Persiste las predicciones del test set para diagnóstico post-hoc
    (p. ej. AUC pairwise por clase). Formato: pred_{stem}.npz

    Contenido: y_true, y_pred, y_proba, class_names (mapeo columna->clase),
    uids (pacientes en el mismo orden) y metadatos del combo.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "npz" / f"pred_{stem}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "y_proba": np.asarray(y_proba, dtype=np.float64),
        "class_names": np.asarray(class_names, dtype=object),
        "uids": np.asarray(uids, dtype=object) if uids is not None else np.asarray([], dtype=object),
    }
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)
    logger.info("Predicciones guardadas: %s", path)
    return path


def _bootstrap_fairness_gaps_for_dims(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    dims: list[str],
) -> dict:
    """
    Bootstrap IC 95% de las brechas de fairness (ΔAUC, avg_gap, TPR/FPR,
    AUC-ES) para cada dimensión sensible presente en `dims`.

    Reusa `bootstrap_fairness_gaps` para que los IC reflejen la variación
    muestral de las brechas, no solo de las métricas globales.
    """
    result = {}
    for dim in dims:
        if dim == "age" and "age" in df.columns:
            gdf = df.copy()
            gdf["age_group"] = bin_age(gdf["age"])
            group_col = "age_group"
        elif dim == "sex" and "sex" in df.columns:
            gdf = df
            group_col = "sex"
        else:
            continue
        if gdf[group_col].notna().sum() < 2:
            continue
        result[dim] = bootstrap_fairness_gaps(
            gdf, y_true, y_pred, y_proba,
            group_col=group_col, min_group_size=MIN_GROUP_SIZE,
            n_bootstrap=N_BOOTSTRAP, seed=SEED,
        )
    return result


def find_pkl(task_key: str, model_name: str, search_dirs: list[str]) -> Path | None:
    """Busca el .pkl generado por run_prepare.py."""
    name = f"final_df_{model_name}_{task_key}.pkl"
    for d in search_dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def evaluate(task_key: str, model_name: str, pkl_path: Path, output_dir: str, probe_filter: str = "both", save_npz: bool = True) -> list[dict]:
    """
    Pipeline completo de evaluación para un (task, model).
    Evalúa cada dataset por separado y guarda un JSON por cada uno.
    Retorna una lista de resultados (uno por dataset).
    """
    task = TASKS[task_key]
    task_name = task["task_name"]
    evaluate_merged = task.get("evaluate_merged", False)

    logger.info("=" * 60)
    logger.info("EVALUANDO: task=%s | model=%s", task_name, model_name)
    logger.info("PKL: %s", pkl_path)
    logger.info("=" * 60)

    # ── 1. Cargar datos ──────────────────────────────────────────────
    with open(pkl_path, "rb") as f:
        final_df = pickle.load(f)
    logger.info("Pacientes cargados: %d", len(final_df))

    # ── 2. Obtener datasets individuales ─────────────────────────────
    if "dataset" in final_df.columns and not evaluate_merged:
        dataset_names = sorted(final_df["dataset"].unique())
    else:
        dataset_names = ["all"]
    logger.info("Datasets encontrados: %s", dataset_names)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ds_name in dataset_names:
        logger.info("--- Dataset: %s ---", ds_name)

        if ds_name == "all":
            ds_df = final_df
        else:
            ds_df = final_df[final_df["dataset"] == ds_name].reset_index(drop=True)

        if len(ds_df) < 30:
            logger.warning("Dataset %s tiene solo %d pacientes, saltando", ds_name, len(ds_df))
            continue

        if ds_df["label"].nunique() < 2:
            logger.warning("Dataset %s tiene una sola clase, saltando evaluación por-dataset", ds_name)
            continue

        # ── 3. Split ─────────────────────────────────────────────────
        # Decisión centralizada (DATASET_EVAL_CONFIG): CV 5-fold si la
        # cohorte lo requiere (fairness por subgrupo sobre el 100% de
        # pacientes), split fijo en caso contrario.
        use_cv = should_use_cv(ds_name, len(ds_df))
        method_reason = eval_method_reason(ds_name, len(ds_df))
        use_mlp_early_stopping = (method_reason == "small_n")
        if use_cv:
            logger.info("[%s] Usando 5-fold CV (motivo: %s)", ds_name, method_reason)
            folds = cross_validate_small_dataset(ds_df, n_splits=5, seed=SEED)
            # Para el resumen, tomamos el primer fold
            first_fold = folds[0]
            train_df, val_df, test_df = first_fold["train"], first_fold["val"], first_fold["test"]
        else:
            logger.info("[%s] Usando split fijo 70/15/15 (motivo: %s)", ds_name, method_reason)
            splits = split_train_val_test(ds_df, seed=SEED)
            train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

        # ── 4. Encode labels ─────────────────────────────────────────
        # Fit on train only so classifier & metrics use the same class set.
        # Para CV, fit sobre el dataset COMPLETO: los folds pueden tener
        # clases distintas y transform() fallaría con clases no vistas.
        label_encoder = LabelEncoder()
        label_encoder.fit(ds_df["label"] if use_cv else train_df["label"])

        X_train = stack_embeddings(train_df)
        y_train = label_encoder.transform(train_df["label"])

        # Filter val/test to classes present in train
        val_mask = val_df["label"].isin(label_encoder.classes_)
        val_df = val_df[val_mask].reset_index(drop=True)
        X_val = stack_embeddings(val_df)
        y_val = label_encoder.transform(val_df["label"])

        test_mask = test_df["label"].isin(label_encoder.classes_)
        test_df = test_df[test_mask].reset_index(drop=True)
        X_test = stack_embeddings(test_df)
        y_test = label_encoder.transform(test_df["label"])

        logger.info("[%s] Clases: %s", ds_name, list(label_encoder.classes_))
        logger.info("[%s] X_train=%s X_val=%s X_test=%s", ds_name, X_train.shape, X_val.shape, X_test.shape)

        # ── 5. Entrenar probes ───────────────────────────────────────
        probes_to_run = []
        if use_cv:
            # 5-fold CV: entrenar y predecir por fold, combinar predicciones
            cv_results = _run_cv_evaluation(
                folds, label_encoder, probe_filter,
                early_stopping=use_mlp_early_stopping,
            )
            # Evaluación normal para el primer fold (val metrics)
            cv_probe_results = {}
            for probe_name in probes_to_run_cv(probe_filter):
                first_clf = cv_results[probe_name]["classifiers"][0]
                y_val_pred = first_clf.predict(X_val)
                y_val_proba = _align_proba_to_full_classes(first_clf, first_clf.predict_proba(X_val), len(label_encoder.classes_))
                val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba, labels=list(range(len(label_encoder.classes_))))

                # Combinar predicciones de todos los folds para test
                n_folds = len(cv_results[probe_name]["test_dfs"])
                all_y_pred = np.concatenate([cv_results[probe_name]["predictions"][f] for f in range(n_folds)])
                all_y_test = np.concatenate([cv_results[probe_name]["true_labels"][f] for f in range(n_folds)])
                all_y_proba = np.vstack([cv_results[probe_name]["probabilities"][f] for f in range(n_folds)])
                all_test_df = pd.concat([cv_results[probe_name]["test_dfs"][f] for f in range(n_folds)], ignore_index=True)

                test_metrics = compute_metrics(all_y_test, all_y_pred, all_y_proba, labels=list(range(len(label_encoder.classes_))))
                test_metrics = _add_auc_ba_diagnostics(
                    test_metrics, all_y_test, all_y_proba,
                    labels=list(range(len(label_encoder.classes_))),
                )
                if save_npz:
                    _save_predictions_npz(
                        out,
                        f"{model_name}_{task_key}_{ds_name}_{probe_name}",
                        all_y_test, all_y_pred, all_y_proba,
                        class_names=list(label_encoder.classes_),
                        uids=all_test_df["uid"].to_numpy() if "uid" in all_test_df.columns else None,
                        extra={
                            "task": task_name, "model": model_name, "dataset": ds_name,
                            "probe": probe_name, "cv_used": True, "seed": SEED,
                            "eval_method_id": "cv5",
                            "n_test": int(len(all_y_test)),
                        },
                    )
                confusion = compute_confusion_matrix(all_y_test, all_y_pred, labels=list(range(len(label_encoder.classes_))))
                error_dist = compute_error_distribution(all_y_test, all_y_pred, labels=list(range(len(label_encoder.classes_))))

                bootstrap_result = bootstrap_metrics(
                    all_y_test, all_y_pred, all_y_proba,
                    n_bootstrap=N_BOOTSTRAP, seed=SEED,
                )

                fairness_result = fairness_report(
                    all_test_df, all_y_test, all_y_pred, all_y_proba,
                    min_group_size=MIN_GROUP_SIZE, seed=SEED,
                )

                fairness_summary = {}
                for dim, subgroup_df in fairness_result.items():
                    auc_col = "auc" if "auc" in test_metrics else "auc_ovr_macro"
                    global_auc = test_metrics.get(auc_col)
                    summary = compute_fairness_summary(subgroup_df, global_auc, lambda_param=1.0)
                    fairness_summary[dim] = summary

                fairness_gaps_bs = _bootstrap_fairness_gaps_for_dims(
                    all_test_df, all_y_test, all_y_pred, all_y_proba,
                    dims=list(fairness_summary.keys()),
                )

                error_by_subgroup = {}
                for dim, subgroup_df in fairness_result.items():
                    if dim == "age" and "age_group" in all_test_df.columns:
                        group_col = "age_group"
                    elif dim == "sex" and "sex" in all_test_df.columns:
                        group_col = "sex"
                    else:
                        continue
                    error_analysis = compute_error_analysis_by_subgroup(
                        all_test_df, all_y_test, all_y_pred, group_col,
                        labels=list(range(len(label_encoder.classes_))),
                        min_group_size=MIN_GROUP_SIZE,
                    )
                    error_by_subgroup[dim] = error_analysis

                cv_probe_results[probe_name] = {
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "confusion_matrix": confusion,
                    "error_distribution": error_dist,
                    "bootstrap": bootstrap_result,
                    "fairness": fairness_result,
                    "fairness_summary": fairness_summary,
                    "fairness_gaps_bootstrap": fairness_gaps_bs,
                    "error_by_subgroup": error_by_subgroup,
                    "cv_fold_k_checks": cv_results[probe_name]["fold_k_checks"],
                }

            results = cv_probe_results
            labels = list(range(len(label_encoder.classes_)))
            n_train_cv = len(train_df)  # approximate from first fold
            first_probe = probes_to_run_cv(probe_filter)[0]
            n_test_cv = sum(len(cv_results[first_probe]["test_dfs"][f]) for f in range(len(cv_results[first_probe]["test_dfs"])))
        else:
            if probe_filter in ("linear", "both"):
                clf_linear = fit_linear_probe(X_train, y_train, seed=SEED)
                probes_to_run.append(("linear", clf_linear))
            if probe_filter in ("mlp", "both"):
                clf_mlp = fit_mlp_probe(
                    X_train, y_train, seed=SEED,
                    early_stopping=use_mlp_early_stopping,
                )
                probes_to_run.append(("mlp", clf_mlp))
            logger.info("[%s] Probes entrenados: %s", ds_name, [p[0] for p in probes_to_run])

            # ── 6. Evaluar en val y test ─────────────────────────────────
            labels = list(range(len(label_encoder.classes_)))
            results = {}
            for probe_name, clf in probes_to_run:
                y_val_pred = clf.predict(X_val)
                y_val_proba = clf.predict_proba(X_val)
                val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba, labels=labels)

                y_test_pred = clf.predict(X_test)
                y_test_proba = clf.predict_proba(X_test)
                test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba, labels=labels)
                test_metrics = _add_auc_ba_diagnostics(
                    test_metrics, y_test, y_test_proba, labels=labels,
                )

                if save_npz:
                    _save_predictions_npz(
                        out,
                        f"{model_name}_{task_key}_{ds_name}_{probe_name}",
                        y_test, y_test_pred, y_test_proba,
                        class_names=list(label_encoder.classes_),
                        uids=test_df["uid"].to_numpy() if "uid" in test_df.columns else None,
                        extra={
                            "task": task_name, "model": model_name, "dataset": ds_name,
                            "probe": probe_name, "cv_used": False, "seed": SEED,
                            "eval_method_id": "fixed_split",
                            "n_test": int(len(y_test)),
                        },
                    )

                # Confusion matrix and error distribution
                confusion = compute_confusion_matrix(y_test, y_test_pred, labels=labels)
                error_dist = compute_error_distribution(y_test, y_test_pred, labels=labels)

                logger.info("[%s][%s] VAL: %s", ds_name, probe_name, val_metrics)
                logger.info("[%s][%s] TEST: %s", ds_name, probe_name, test_metrics)

                # ── 7. Bootstrap ─────────────────────────────────────────
                bootstrap_result = bootstrap_metrics(
                    y_test, y_test_pred, y_test_proba,
                    n_bootstrap=N_BOOTSTRAP, seed=SEED,
                )
                logger.info("[%s][%s] Bootstrap completado", ds_name, probe_name)

                # ── 8. Fairness ──────────────────────────────────────────
                fairness_result = fairness_report(
                    test_df, y_test, y_test_pred, y_test_proba,
                    min_group_size=MIN_GROUP_SIZE,
                    seed=SEED,
                )
                logger.info("[%s][%s] Fairness completado", ds_name, probe_name)

                # ── 9. Fairness Summary ──────────────────────────────────
                fairness_summary = {}
                for dim, subgroup_df in fairness_result.items():
                    auc_col = "auc" if "auc" in test_metrics else "auc_ovr_macro"
                    global_auc = test_metrics.get(auc_col)
                    summary = compute_fairness_summary(subgroup_df, global_auc, lambda_param=1.0)
                    fairness_summary[dim] = summary
                logger.info("[%s][%s] Fairness summary completado", ds_name, probe_name)

                # ── 9b. Bootstrap de brechas de fairness (IC 95%) ─────────
                fairness_gaps_bs = _bootstrap_fairness_gaps_for_dims(
                    test_df, y_test, y_test_pred, y_test_proba,
                    dims=list(fairness_summary.keys()),
                )
                logger.info("[%s][%s] Bootstrap fairness gaps completado", ds_name, probe_name)

                # ── 10. Error Analysis by Subgroup ───────────────────────
                error_by_subgroup = {}
                for dim, subgroup_df in fairness_result.items():
                    if dim == "age" and "age_group" in test_df.columns:
                        group_col = "age_group"
                    elif dim == "sex" and "sex" in test_df.columns:
                        group_col = "sex"
                    else:
                        continue

                    error_analysis = compute_error_analysis_by_subgroup(
                        test_df, y_test, y_test_pred, group_col, labels=labels, min_group_size=MIN_GROUP_SIZE
                    )
                    error_by_subgroup[dim] = error_analysis
                logger.info("[%s][%s] Error analysis completado", ds_name, probe_name)

                results[probe_name] = {
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "confusion_matrix": confusion,
                    "error_distribution": error_dist,
                    "bootstrap": bootstrap_result,
                    "fairness": fairness_result,
                    "fairness_summary": fairness_summary,
                    "fairness_gaps_bootstrap": fairness_gaps_bs,
                    "error_by_subgroup": error_by_subgroup,
                }

        # ── 11. Guardar por dataset ──────────────────────────────────
        n_test_final = len(test_df) if not use_cv else n_test_cv
        summary = {
            "task": task_name,
            "model": model_name,
            "dataset": ds_name,
            "seed": SEED,
            "cv_used": use_cv,
            "eval_method_id": "cv5" if use_cv else "fixed_split",
            "eval_method_reason": method_reason,
            "mlp_early_stopping": use_mlp_early_stopping,
            "n_total": len(ds_df),
            "n_train": len(train_df) if not use_cv else n_train_cv,
            "n_val": len(val_df) if not use_cv else None,
            "n_test": n_test_final,
            "classes": list(label_encoder.classes_),
            "probes": {},
        }

        for probe_name, r in results.items():
            probe_summary = {
                "val_metrics": r["val_metrics"],
                "test_metrics": r["test_metrics"],
                "confusion_matrix": r["confusion_matrix"],
                "error_distribution": r["error_distribution"],
                "bootstrap": {k: v for k, v in r["bootstrap"].items() if not k.startswith("_")},
                "fairness_summary": r["fairness_summary"],
                "fairness_gaps_bootstrap": r.get("fairness_gaps_bootstrap", {}),
                "error_by_subgroup": r["error_by_subgroup"],
                "cv_fold_k_checks": r.get("cv_fold_k_checks", []),
            }
            # Fairness DataFrames → CSV
            for dim, df in r["fairness"].items():
                csv_path = out / f"fairness_{model_name}_{task_key}_{ds_name}_{probe_name}_{dim}.csv"
                df.to_csv(csv_path, index=False)
                probe_summary[f"fairness_{dim}"] = df.to_dict(orient="records")

            summary["probes"][probe_name] = probe_summary

        # JSON por dataset
        json_path = out / f"eval_{model_name}_{task_key}_{ds_name}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Guardado: %s", json_path)

        # Print resumen
        print(f"\n{'='*60}")
        print(f"RESULTADOS: {task_name} + {model_name} + {ds_name}" + (" [5-fold CV]" if use_cv else ""))
        print(f"{'='*60}")
        for probe_name, r in results.items():
            t = r["test_metrics"]
            auc_val = t.get('auc', t.get('auc_ovr_macro', 0)) or 0
            ba_val = t.get('balanced_accuracy', 0)
            sens_str = ", ".join(f"{k}={v:.3f}" for k, v in t.get("per_class_sensitivity", {}).items())
            print(f"  {probe_name:<8} n_test={n_test_final}  acc={t['accuracy']:.3f}  bal_acc={ba_val:.3f}  f1={t['f1_macro']:.3f}  auc={auc_val:.3f}")
            if sens_str:
                print(f"           sensitivity: {sens_str}")

            # Confusion matrix
            cm = r["confusion_matrix"]["confusion_matrix"]
            cm_labels = r["confusion_matrix"]["labels"]
            print(f"           Confusion matrix ({len(cm_labels)}x{len(cm_labels)}):")
            for i, row in enumerate(cm):
                print(f"             true={cm_labels[i]}: {row}")

            # Error distribution
            ed = r["error_distribution"]
            print(f"           Errors: {ed['total_errors']}/{ed['total_samples']} ({ed['error_rate']:.1%})")
            if ed["most_confused"]:
                true_cls, pred_cls, count = ed["most_confused"]
                print(f"           Most confused: {true_cls} → {pred_cls} ({count} times)")

            # Fairness
            for dim, fs in r["fairness_summary"].items():
                delta_auc = fs.get("delta_auc", {}).get("delta_auc")
                avg_gap = fs.get("avg_gap", {}).get("avg_gap")
                auc_es = fs.get("auc_es")
                if delta_auc is not None:
                    print(f"           {dim}: ΔAUC={delta_auc:.3f}  AvgGap={avg_gap:.3f}  AUC_ES={auc_es:.3f}")
        print()

        all_results.append(summary)

    return all_results


def cross_dataset_evaluate_task(
    task_key: str,
    model_name: str,
    pkl_path: Path,
    output_dir: str,
    probe_filter: str = "both",
    save_npz: bool = True,
) -> list[dict]:
    """
    Cross-dataset OOD evaluation: train on each dataset, test on every other dataset.
    Saves one JSON per (train_dataset, test_dataset, probe) combination.
    """
    from eval.evaluation.cross_dataset import compute_ood_degradation

    task = TASKS[task_key]
    task_name = task["task_name"]

    logger.info("=" * 60)
    logger.info("CROSS-DATASET: task=%s | model=%s", task_name, model_name)
    logger.info("PKL: %s", pkl_path)
    logger.info("=" * 60)

    with open(pkl_path, "rb") as f:
        final_df = pickle.load(f)

    if "dataset" not in final_df.columns:
        logger.warning("No 'dataset' column — skipping cross-dataset")
        return []

    dataset_names = sorted(final_df["dataset"].unique())
    if len(dataset_names) < 2:
        logger.warning("Need >= 2 datasets for cross-dataset, found %d", len(dataset_names))
        return []

    logger.info("Datasets: %s (%d pairs)", dataset_names, len(dataset_names) * (len(dataset_names) - 1))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    in_distribution_results = []

    all_cross = []
    for train_ds in dataset_names:
        train_df = final_df[final_df["dataset"] == train_ds].reset_index(drop=True)
        label_encoder = LabelEncoder()
        label_encoder.fit(train_df["label"])

        for test_ds in dataset_names:
            if train_ds == test_ds:
                continue

            test_df = final_df[final_df["dataset"] == test_ds].reset_index(drop=True)
            test_mask = test_df["label"].isin(label_encoder.classes_)
            test_df = test_df[test_mask].reset_index(drop=True)
            if len(train_df) < 30 or len(test_df) < 10:
                logger.warning("Skipping %s->%s (train=%d, test=%d)", train_ds, test_ds, len(train_df), len(test_df))
                continue

            # Saltar pares con una sola clase en train o test (ej: organ
            # evaluado por-dataset, donde cada dataset tiene 1 clase)
            if train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2:
                logger.warning(
                    "Skipping %s->%s (train_classes=%d, test_classes=%d)",
                    train_ds, test_ds,
                    train_df["label"].nunique(), test_df["label"].nunique(),
                )
                continue

            logger.info("Cross: %s -> %s (train=%d, test=%d)", train_ds, test_ds, len(train_df), len(test_df))

            probes_to_run = []
            if probe_filter in ("linear", "both"):
                probes_to_run.append("linear")
            if probe_filter in ("mlp", "both"):
                probes_to_run.append("mlp")

            for probe_name in probes_to_run:
                try:
                    result = cross_dataset_evaluate(
                        train_df=train_df,
                        test_df=test_df,
                        model_class=None,
                        model_params={},
                        label_encoder=label_encoder,
                        sensitive_attrs=["sex", "age"],
                        min_group_size=MIN_GROUP_SIZE,
                        seed=SEED,
                    )
                except TypeError:
                    result = _cross_evaluate_manual(
                        train_df, test_df, label_encoder, probe_name,
                    )

                pred_info = result.get("predictions")

                if pred_info is not None:
                    ood_labels = list(range(len(label_encoder.classes_)))
                    result["metrics"] = _add_auc_ba_diagnostics(
                        result.get("metrics", {}),
                        pred_info["y_true"],
                        pred_info["y_proba"],
                        labels=ood_labels,
                    )

                probe_result = {
                    "task": task_name,
                    "model": model_name,
                    "train_dataset": train_ds,
                    "test_dataset": test_ds,
                    "probe": probe_name,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "eval_method_ood": "full_data_model",  # modelo origen 100%-data
                    "eval_method_id": choose_eval_method(train_ds, len(train_df)),  # método del origen
                    "metrics": result.get("metrics", {}),
                    "confusion_matrix": result.get("confusion_matrix", {}),
                    "error_distribution": result.get("error_distribution", {}),
                    "bootstrap": {k: v for k, v in result.get("bootstrap", {}).items() if not k.startswith("_")},
                    "fairness_summary": {},
                    "fairness_gaps_bootstrap": result.get("fairness_gaps_bootstrap", {}),
                }

                # Persistir predicciones OOD del modelo 100%-data (bootstraps
                # de ΔBA/ΔF1 en el ensamblado de generalización).
                pred_info = result.get("predictions")
                if pred_info is not None and save_npz:
                    _save_predictions_npz(
                        out,
                        f"cross_{model_name}_{task_key}_{train_ds}_to_{test_ds}_{probe_name}",
                        pred_info["y_true"], pred_info["y_pred"], pred_info["y_proba"],
                        class_names=list(label_encoder.classes_),
                        uids=pred_info.get("uids"),
                        extra={
                            "task": task_name, "model": model_name,
                            "train_dataset": train_ds, "test_dataset": test_ds,
                            "probe": probe_name, "ood": True, "seed": SEED,
                            "eval_method_ood": "full_data_model",
                            "n_test": int(len(pred_info["y_true"])),
                        },
                    )

                for dim, sd in result.get("fairness_summary", {}).items():
                    probe_result["fairness_summary"][dim] = sd

                for dim, df in result.get("fairness", {}).items():
                    csv_path = out / f"cross_{model_name}_{task_key}_{train_ds}_to_{test_ds}_{probe_name}_{dim}.csv"
                    if hasattr(df, "to_csv"):
                        df.to_csv(csv_path, index=False)

                json_path = out / f"cross_{model_name}_{task_key}_{train_ds}_to_{test_ds}_{probe_name}.json"
                with open(json_path, "w") as f:
                    json.dump(probe_result, f, indent=2, default=str)
                logger.info("Guardado: %s", json_path)

                t = probe_result["metrics"]
                auc_val = t.get("auc", t.get("auc_ovr_macro", 0)) or 0
                ba_val = t.get("balanced_accuracy", 0)
                print(f"  {train_ds} -> {test_ds} [{probe_name}]  acc={t.get('accuracy',0):.3f}  bal_acc={ba_val:.3f}  f1={t.get('f1_macro',0):.3f}  auc={auc_val:.3f}")

                all_cross.append(probe_result)

    return all_cross


def _cross_evaluate_manual(train_df, test_df, label_encoder, probe_name):
    """Fallback: manual cross-dataset evaluation using probe functions."""
    from eval.evaluation.bootstrap import bootstrap_metrics
    from eval.evaluation.metrics import compute_confusion_matrix, compute_error_distribution
    from eval.evaluation.fairness import fairness_report, compute_fairness_summary

    X_train = stack_embeddings(train_df)
    y_train = label_encoder.transform(train_df["label"])

    test_mask = test_df["label"].isin(label_encoder.classes_)
    test_df = test_df[test_mask].reset_index(drop=True)
    X_test = stack_embeddings(test_df)
    y_test = label_encoder.transform(test_df["label"])

    if probe_name == "linear":
        clf = fit_linear_probe(X_train, y_train, seed=SEED)
    else:
        # OOD is not classified by eval_method_reason here: keep early stopping off.
        clf = fit_mlp_probe(X_train, y_train, seed=SEED, early_stopping=False)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    test_metrics = compute_metrics(y_test, y_pred, y_proba, labels=list(range(len(label_encoder.classes_))))
    test_metrics = _add_auc_ba_diagnostics(
        test_metrics, y_test, y_proba,
        labels=list(range(len(label_encoder.classes_))),
    )
    labels = list(range(len(label_encoder.classes_)))
    confusion = compute_confusion_matrix(y_test, y_pred, labels=labels)
    error_dist = compute_error_distribution(y_test, y_pred, labels=labels)

    bootstrap_result = bootstrap_metrics(y_test, y_pred, y_proba, n_bootstrap=N_BOOTSTRAP, seed=SEED)

    fairness_result = fairness_report(test_df, y_test, y_pred, y_proba, min_group_size=MIN_GROUP_SIZE, seed=SEED)

    fairness_summary = {}
    for dim, subgroup_df in fairness_result.items():
        auc_col = "auc" if "auc" in test_metrics else "auc_ovr_macro"
        global_auc = test_metrics.get(auc_col)
        summary = compute_fairness_summary(subgroup_df, global_auc, lambda_param=1.0)
        fairness_summary[dim] = summary

    fairness_gaps_bs = _bootstrap_fairness_gaps_for_dims(
        test_df, y_test, y_pred, y_proba,
        dims=list(fairness_summary.keys()),
    )

    return {
        "metrics": test_metrics,
        "confusion_matrix": confusion,
        "error_distribution": error_dist,
        "bootstrap": bootstrap_result,
        "fairness": fairness_result,
        "fairness_summary": fairness_summary,
        "fairness_gaps_bootstrap": fairness_gaps_bs,
        "predictions": {
            "y_true": y_test,
            "y_pred": y_pred,
            "y_proba": np.asarray(y_proba, dtype=np.float64),
            "uids": test_df["uid"].to_numpy() if "uid" in test_df.columns else None,
        },
    }


def probes_to_run_cv(probe_filter: str) -> list[str]:
    """Helper: lista de nombres de probes a ejecutar en CV."""
    probes = []
    if probe_filter in ("linear", "both"):
        probes.append("linear")
    if probe_filter in ("mlp", "both"):
        probes.append("mlp")
    return probes


def _align_proba_to_full_classes(
    clf, y_proba: np.ndarray, n_full_classes: int,
) -> np.ndarray:
    """
    Alinea predict_proba() al conjunto de clases completo.

    En CV 5-fold, cada fold entrena el clasificador con las clases
    presentes en SU train: si una clase rara no aparece en un fold, el
    proba de ese fold tiene menos columnas. Se rellena con 0 las columnas
    faltantes para poder apilar los folds con np.vstack.
    """
    if y_proba.shape[1] == n_full_classes:
        return y_proba
    aligned = np.zeros((y_proba.shape[0], n_full_classes))
    clf_classes = np.asarray(clf.classes_, dtype=int)
    aligned[:, clf_classes] = y_proba
    return aligned


def _run_cv_evaluation(
    folds: dict[int, dict],
    label_encoder,
    probe_filter: str,
    early_stopping: bool = False,
) -> dict:
    """
    Ejecuta entrenamiento+predicción para cada fold de CV.
    Retorna dict con resultados por probe, incluyendo clasificadores
    entrenados y predicciones combinables.
    """
    from eval.models.linear_probe import fit_linear_probe
    from eval.models.mlp_probe import fit_mlp_probe

    n_full_classes = len(label_encoder.classes_)
    probes = probes_to_run_cv(probe_filter)
    cv_results = {
        p: {
            "classifiers": [], "predictions": [], "probabilities": [],
            "true_labels": [], "test_dfs": [], "fold_k_checks": [],
        }
        for p in probes
    }

    for fold_idx in sorted(folds.keys()):
        fold = folds[fold_idx]
        train_df = fold["train"]
        test_df = fold["test"]

        X_train = stack_embeddings(train_df)
        y_train = label_encoder.transform(train_df["label"])

        test_mask = test_df["label"].isin(label_encoder.classes_)
        test_df = test_df[test_mask].reset_index(drop=True)
        X_test = stack_embeddings(test_df)
        y_test = label_encoder.transform(test_df["label"])
        fold_k_check = k_consistency_check(
            y_test, list(range(n_full_classes)),
        )

        for probe_name in probes:
            if probe_name == "linear":
                clf = fit_linear_probe(X_train, y_train, seed=SEED)
            else:
                clf = fit_mlp_probe(
                    X_train, y_train, seed=SEED,
                    early_stopping=early_stopping,
                )

            y_pred = clf.predict(X_test)
            y_proba = _align_proba_to_full_classes(clf, clf.predict_proba(X_test), n_full_classes)

            cv_results[probe_name]["classifiers"].append(clf)
            cv_results[probe_name]["predictions"].append(y_pred)
            cv_results[probe_name]["probabilities"].append(y_proba)
            cv_results[probe_name]["true_labels"].append(y_test)
            cv_results[probe_name]["test_dfs"].append(test_df)
            cv_results[probe_name]["fold_k_checks"].append({
                "fold": int(fold_idx),
                **fold_k_check,
            })

    return cv_results


def main():
    parser = argparse.ArgumentParser(description="Evaluar modelo en tarea específica")
    parser.add_argument("--task", choices=list(TASKS.keys()), help="Task key")
    parser.add_argument("--model", choices=MODELS, help="Model name")
    parser.add_argument("--all-tasks", action="store_true", help="Todas las tareas con el modelo")
    parser.add_argument("--all", action="store_true", help="Todas las combinaciones")
    parser.add_argument("--list", action="store_true", help="Listar combinaciones")
    parser.add_argument("--pkl-dir", default="pkl_final", help="Directorio donde están los .pkl")
    parser.add_argument("--output-dir", default="results", help="Directorio de salida")
    parser.add_argument("--probe", choices=["linear", "mlp", "both"], default="mlp",
                        help="Tipo de clasificador (default: mlp)")
    parser.add_argument("--cross-dataset", action="store_true", help="Cross-dataset OOD evaluation")
    parser.add_argument("--no-npz", action="store_true",
                        help="No guardar pred_*.npz (predicciones para diagnóstico post-hoc)")
    args = parser.parse_args()

    if args.list:
        print("\nTareas disponibles:")
        for key, task in TASKS.items():
            datasets = [s.name for s in task["datasets"]]
            print(f"  {key:<20} task={task['task_name']:<25} datasets={datasets}")
        print(f"\nModelos: {MODELS}")
        print(f"\nTotal: {len(TASKS) * len(MODELS)} combinaciones")
        return

    if args.all:
        combos = [(t, m) for t in TASKS for m in MODELS]
    elif args.all_tasks and args.model:
        combos = [(t, args.model) for t in TASKS]
    elif args.task and args.model:
        combos = [(args.task, args.model)]
    else:
        parser.print_help()
        return

    # Buscar .pkl para cada combinación
    search_dirs = [args.pkl_dir, "."]
    missing = []
    ready = []
    for task_key, model in combos:
        pkl = find_pkl(task_key, model, search_dirs)
        if pkl:
            ready.append((task_key, model, pkl))
        else:
            missing.append((task_key, model))

    if missing:
        print(f"\nFaltan {len(missing)} .pkl (correr run_prepare.py primero):")
        for t, m in missing:
            print(f"  • {t} + {m}")

    if not ready:
        print("\nNo hay .pkl disponibles para evaluar.")
        print("Corré primero: python run_prepare.py --all")
        return

    print(f"\n{len(ready)} combinaciones para evaluar:\n")
    for task_key, model, pkl in ready:
        print(f"  • {task_key} + {model} → {pkl.name}")
    print()

    all_results = []
    for task_key, model, pkl in ready:
        try:
            results = evaluate(task_key, model, pkl, args.output_dir, probe_filter=args.probe, save_npz=not args.no_npz)
            for r in results:
                ds_name = r.get("dataset", "unknown")
                all_results.append((task_key, model, ds_name, "OK"))
        except Exception as e:
            logger.error("FALLÓ %s + %s: %s", task_key, model, e, exc_info=True)
            all_results.append((task_key, model, "-", f"ERROR: {e}"))

    # Cross-dataset OOD evaluation
    if args.cross_dataset:
        print("\n" + "=" * 60)
        print("CROSS-DATASET OOD EVALUATION")
        print("=" * 60)
        for task_key, model, pkl in ready:
            try:
                cross_results = cross_dataset_evaluate_task(
                    task_key, model, pkl, args.output_dir, probe_filter=args.probe, save_npz=not args.no_npz,
                )
                for cr in cross_results:
                    train_ds = cr.get("train_dataset", "?")
                    test_ds = cr.get("test_dataset", "?")
                    all_results.append((task_key, model, f"{train_ds}->{test_ds}", "OK (cross)"))
            except Exception as e:
                logger.error("FALLÓ cross-dataset %s + %s: %s", task_key, model, e, exc_info=True)
                all_results.append((task_key, model, "cross", f"ERROR: {e}"))

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    for task_key, model, ds_name, status in all_results:
        print(f"  {task_key:<20} {model:<15} {ds_name:<35} {status}")


if __name__ == "__main__":
    main()
