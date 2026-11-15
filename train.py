from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import (
    generate_features,
    get_feature_columns,
    read_table,
    save_feature_columns,
)
from model_wrappers import TorchMLPRegressor


RANDOM_STATE = 2026
TARGETS = ["future_5_return", "future_60_return"]
RIDGE_BLEND_GRID = [0.0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60]
REVERSION_BLEND_GRID = [0.0, 0.02, 0.05, 0.10, 0.15, 0.25]
DIRECTION_THRESHOLD_QUANTILES = np.linspace(0.05, 0.95, 37)


def apply_reproducibility_seeds(seed: int = RANDOM_STATE) -> None:
    """固定 Python / NumPy 随机源，并限制 BLAS 线程，便于远程严格复现（LightGBM 请保持 --lgbm-n-jobs 1）。"""
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    random.seed(seed)
    np.random.seed(seed)
    # 注：若需固定 dict/set 哈希顺序，请在启动 Python 前设置环境变量 PYTHONHASHSEED=0


def resolve_framework(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except Exception:
        return "sklearn"


def dataset_number(path: Path) -> int:
    match = re.search(r"dataset(\d+)", path.name)
    return int(match.group(1)) if match else 0


def thin_by_time(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    idx = np.linspace(0, len(df) - 1, max_rows, dtype=np.int64)
    return df.iloc[idx].copy()


def load_training_frame(input_path: str, create_targets: bool, max_train_rows: int) -> pd.DataFrame:
    path = Path(input_path)
    if not path.is_dir():
        raw = read_table(path)
        df = generate_features(raw, create_targets=create_targets)
        return thin_by_time(df, max_train_rows)

    files = sorted(path.glob("*_ohlcv.npy"), key=lambda p: (dataset_number(p), p.name))
    if not files:
        raw = read_table(path)
        df = generate_features(raw, create_targets=create_targets)
        return thin_by_time(df, max_train_rows)

    per_file_limit = 0 if max_train_rows <= 0 else max(1_000, max_train_rows // len(files))
    frames = []
    for file in files:
        print(f"读取并生成特征: {file.name}")
        one_raw = read_table(file)
        one_df = generate_features(one_raw, create_targets=create_targets)
        one_df = thin_by_time(one_df, per_file_limit)
        frames.append(one_df)
        print(f"  保留样本: {len(one_df):,}")

    df = pd.concat(frames, ignore_index=True)
    if max_train_rows > 0:
        df = thin_by_time(df.sort_values(["timestamp", "asset_id"]).reset_index(drop=True), max_train_rows)
    return df


def make_model(
    framework: str,
    torch_epochs: int = 30,
    *,
    lgbm_n_jobs: int = 1,
):
    if framework == "pytorch":
        return TorchMLPRegressor(epochs=torch_epochs, random_state=RANDOM_STATE)
    if framework == "sklearn":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=63,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )

    try:
        from lightgbm import LGBMRegressor  # type: ignore[reportMissingImports]

        # n_jobs=1 + force_col_wise：避免多线程直方图累加顺序导致的非确定性（严格复现用）。
        return LGBMRegressor(
            objective="regression",
            learning_rate=0.03,
            num_leaves=63,
            max_depth=6,
            n_estimators=800,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=lgbm_n_jobs,
            verbose=-1,
            force_col_wise=True,
        )
    except Exception as exc:
        if framework == "lightgbm":
            raise ImportError("指定了 --framework lightgbm，但当前环境无法导入 LightGBM，请先 pip install lightgbm") from exc
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=63,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )


def fill_numeric_features(
    train_x: pd.DataFrame,
    valid_x: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, float]]:
    train_clean = train_x.copy()
    for col in train_clean.columns:
        values = train_clean[col].to_numpy(copy=False)
        if np.issubdtype(values.dtype, np.floating):
            finite_mask = np.isfinite(values)
            if not finite_mask.all():
                train_clean[col] = train_clean[col].where(finite_mask, np.nan)
    fill_values = train_clean.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0).to_dict()
    train_filled = train_clean.fillna(fill_values).astype("float32")
    valid_filled = None
    if valid_x is not None:
        valid_clean = valid_x.copy()
        for col in valid_clean.columns:
            values = valid_clean[col].to_numpy(copy=False)
            if np.issubdtype(values.dtype, np.floating):
                finite_mask = np.isfinite(values)
                if not finite_mask.all():
                    valid_clean[col] = valid_clean[col].where(finite_mask, np.nan)
        valid_filled = valid_clean.fillna(fill_values).astype("float32")
    return train_filled, valid_filled, fill_values


def direction_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y = y_true.to_numpy()
    mask = y != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(y[mask]) == np.sign(np.asarray(y_pred)[mask])))


def rank_ic(y_true: pd.Series, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float("nan")
    corr = pd.Series(y_true).rank().corr(pd.Series(y_pred, index=y_true.index).rank())
    return float(corr) if corr is not None else float("nan")


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "direction_accuracy": direction_accuracy(y_true, y_pred),
        "rank_ic": rank_ic(y_true, y_pred),
    }


def reversion_feature_for_target(target: str) -> str:
    return "ret_5" if target == "future_5_return" else "ret_60"


def make_reversion_prediction(x: pd.DataFrame, target: str) -> np.ndarray:
    col = reversion_feature_for_target(target)
    if col not in x.columns:
        return np.zeros(len(x), dtype=np.float32)
    return -x[col].fillna(0).to_numpy(dtype=np.float32)


def blend_predictions(
    main_pred: np.ndarray,
    ridge_pred: np.ndarray,
    reversion_pred: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    return (
        weights.get("main", 1.0) * main_pred
        + weights.get("ridge", 0.0) * ridge_pred
        + weights.get("reversion", 0.0) * reversion_pred
    )


def apply_direction_threshold(y_pred: np.ndarray, threshold: float) -> np.ndarray:
    """Shift the prediction zero point without changing rank order."""
    return np.asarray(y_pred) - float(threshold)


def choose_blend_weights(fold_predictions: list[dict]) -> dict[str, float]:
    best_weights = {"main": 1.0, "ridge": 0.0, "reversion": 0.0}
    best_score = float("-inf")
    best_mae = float("inf")

    for ridge_weight in RIDGE_BLEND_GRID:
        for reversion_weight in REVERSION_BLEND_GRID:
            if ridge_weight + reversion_weight > 0.85:
                continue
            weights = {
                "main": 1.0 - ridge_weight - reversion_weight,
                "ridge": ridge_weight,
                "reversion": reversion_weight,
            }
            fold_scores = []
            fold_mae = []
            for item in fold_predictions:
                pred = blend_predictions(
                    item["main_pred"],
                    item["ridge_pred"],
                    item["reversion_pred"],
                    weights,
                )
                metrics = evaluate_predictions(item["y_valid"], pred)
                if np.isfinite(metrics["rank_ic"]):
                    fold_scores.append(metrics["rank_ic"])
                fold_mae.append(metrics["mae"])
            score = float(np.mean(fold_scores)) if fold_scores else float("-inf")
            mae = float(np.mean(fold_mae)) if fold_mae else float("inf")
            if score > best_score or (np.isclose(score, best_score) and mae < best_mae):
                best_score = score
                best_mae = mae
                best_weights = weights

    return best_weights


def choose_direction_threshold(fold_predictions: list[dict], weights: dict[str, float]) -> float:
    all_pred = []
    for item in fold_predictions:
        pred = blend_predictions(
            item["main_pred"],
            item["ridge_pred"],
            item["reversion_pred"],
            weights,
        )
        all_pred.append(np.asarray(pred))
    if not all_pred:
        return 0.0

    pred_values = np.concatenate(all_pred)
    finite_pred = pred_values[np.isfinite(pred_values)]
    if len(finite_pred) == 0:
        return 0.0

    pred_span = float(np.nanmax(finite_pred) - np.nanmin(finite_pred))
    eps = max(pred_span * 1e-6, 1e-12)
    candidates = np.unique(
        np.concatenate(
            (
                [0.0, float(np.nanmin(finite_pred) - eps), float(np.nanmax(finite_pred) + eps)],
                np.quantile(finite_pred, DIRECTION_THRESHOLD_QUANTILES),
            )
        )
    )
    best_threshold = 0.0
    best_direction = float("-inf")
    best_mae = float("inf")

    for threshold in candidates:
        fold_direction = []
        fold_mae = []
        for item in fold_predictions:
            pred = blend_predictions(
                item["main_pred"],
                item["ridge_pred"],
                item["reversion_pred"],
                weights,
            )
            calibrated = apply_direction_threshold(pred, float(threshold))
            metrics = evaluate_predictions(item["y_valid"], calibrated)
            fold_direction.append(metrics["direction_accuracy"])
            fold_mae.append(metrics["mae"])

        direction = float(np.mean(fold_direction)) if fold_direction else float("-inf")
        mae = float(np.mean(fold_mae)) if fold_mae else float("inf")
        if direction > best_direction or (np.isclose(direction, best_direction) and mae < best_mae):
            best_direction = direction
            best_mae = mae
            best_threshold = float(threshold)

    return best_threshold


def walk_forward_splits(df: pd.DataFrame, n_splits: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    ordered = df.sort_values("timestamp").reset_index()
    n = len(ordered)
    splits = []
    boundaries = [(0.6, 0.7), (0.7, 0.8), (0.8, 0.9)]
    for train_end_frac, valid_end_frac in boundaries[:n_splits]:
        train_end = int(n * train_end_frac)
        valid_end = int(n * valid_end_frac)
        if train_end <= 0 or valid_end <= train_end:
            continue
        train_idx = ordered.iloc[:train_end]["index"].to_numpy()
        valid_idx = ordered.iloc[train_end:valid_end]["index"].to_numpy()
        splits.append((train_idx, valid_idx))
    return splits


def fit_one_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    model_dir: Path,
    framework: str,
    torch_epochs: int,
    lgbm_n_jobs: int,
) -> dict:
    target_df = df.dropna(subset=[target]).copy()
    if target_df.empty:
        raise ValueError(f"{target} 没有可训练样本，请检查标签列或使用 --create-targets")

    fold_predictions = []
    for fold, (train_idx, valid_idx) in enumerate(walk_forward_splits(target_df), start=1):
        train_part = target_df.loc[train_idx]
        valid_part = target_df.loc[valid_idx]

        x_train, x_valid, fill_values = fill_numeric_features(train_part[feature_cols], valid_part[feature_cols])
        y_train = train_part[target]
        y_valid = valid_part[target]
        weights = train_part.get("sample_weight", pd.Series(1.0, index=train_part.index))

        model = make_model(framework, torch_epochs=torch_epochs, lgbm_n_jobs=lgbm_n_jobs)
        model.fit(x_train, y_train, sample_weight=weights)
        main_pred = model.predict(x_valid)

        ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        ridge_model.fit(x_train, y_train, ridge__sample_weight=weights)
        ridge_pred = ridge_model.predict(x_valid)
        reversion_pred = make_reversion_prediction(x_valid, target)
        fold_predictions.append(
            {
                "fold": fold,
                "valid_part": valid_part,
                "y_valid": y_valid,
                "main_pred": main_pred,
                "ridge_pred": ridge_pred,
                "reversion_pred": reversion_pred,
            }
        )

    blend_weight = choose_blend_weights(fold_predictions)
    direction_threshold = choose_direction_threshold(fold_predictions, blend_weight)

    cv_rows = []
    for item in fold_predictions:
        fold = item["fold"]
        valid_part = item["valid_part"]
        y_valid = item["y_valid"]
        pred = blend_predictions(item["main_pred"], item["ridge_pred"], item["reversion_pred"], blend_weight)
        pred = apply_direction_threshold(pred, direction_threshold)
        cv_rows.append({"target": target, "fold": fold, "group": "all", **evaluate_predictions(y_valid, pred)})
        for group_name, mask in [
            ("normal", valid_part["extreme_state"].fillna(0).eq(0)),
            ("extreme", valid_part["extreme_state"].fillna(0).eq(1)),
        ]:
            if mask.sum() > 0:
                group_metrics = evaluate_predictions(y_valid[mask], pred[mask.to_numpy()])
                cv_rows.append({"target": target, "fold": fold, "group": group_name, **group_metrics})

    x_full, _unused, fill_values = fill_numeric_features(target_df[feature_cols])
    y_full = target_df[target]
    full_weight = target_df.get("sample_weight", pd.Series(1.0, index=target_df.index))

    final_model = make_model(framework, torch_epochs=torch_epochs, lgbm_n_jobs=lgbm_n_jobs)
    final_model.fit(x_full, y_full, sample_weight=full_weight)

    ridge = None
    if blend_weight.get("ridge", 0.0) > 0:
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        ridge.fit(x_full, y_full, ridge__sample_weight=full_weight)

    model_path = model_dir / f"model_{target}.joblib"
    joblib.dump(
        {
            "target": target,
            "framework": framework,
            "random_state": RANDOM_STATE,
            "lgbm_n_jobs": lgbm_n_jobs if framework == "lightgbm" else None,
            "feature_cols": feature_cols,
            "fill_values": fill_values,
            "model": final_model,
            "ridge_model": ridge,
            "blend_weight": blend_weight,
            "reversion_feature": reversion_feature_for_target(target),
            "direction_threshold": direction_threshold,
        },
        model_path,
    )

    importance_path = None
    if hasattr(final_model, "feature_importances_"):
        importance = (
            pd.DataFrame({"feature": feature_cols, "importance": final_model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        importance_path = model_dir / f"feature_importance_{target}.csv"
        importance.to_csv(importance_path, index=False)

    return {
        "target": target,
        "model_path": str(model_path),
        "importance_path": str(importance_path) if importance_path else None,
        "blend_weight": blend_weight,
        "direction_threshold": direction_threshold,
        "cv_rows": cv_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 C 题未来收益率预测模型")
    parser.add_argument("--input", required=True, help="训练数据路径，支持 train_dataset 目录、npy、csv、parquet、xlsx")
    parser.add_argument("--model-dir", default="models", help="模型输出目录")
    parser.add_argument("--create-targets", action="store_true", help="没有标签列时，用未来收盘价生成标签")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=1_500_000,
        help="最多保留的训练行数；默认 150 万以避免内存不足，设为 0 表示全量训练",
    )
    parser.add_argument(
        "--framework",
        choices=["auto", "lightgbm", "sklearn", "pytorch"],
        default="auto",
        help="主模型框架：auto 优先 LightGBM，不可用则回退 sklearn",
    )
    parser.add_argument("--torch-epochs", type=int, default=30, help="使用 --framework pytorch 时的训练轮数")
    parser.add_argument(
        "--lgbm-n-jobs",
        type=int,
        default=1,
        help="LightGBM 线程数：默认 1 以保证严格可复现；设为 -1 可加速，但并行可能导致结果随运行略有差异",
    )
    args = parser.parse_args()

    apply_reproducibility_seeds(RANDOM_STATE)
    resolved_fw = resolve_framework(args.framework)
    if args.framework == "lightgbm" and resolved_fw != "lightgbm":
        print("错误: 未安装 LightGBM，无法使用 --framework lightgbm", file=sys.stderr)
        sys.exit(1)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_frame(args.input, args.create_targets, args.max_train_rows)
    missing_targets = [target for target in TARGETS if target not in df.columns]
    if missing_targets:
        raise ValueError(f"缺少标签列 {missing_targets}。若要由 close 生成标签，请加 --create-targets")

    feature_cols = get_feature_columns(df)
    save_feature_columns(feature_cols, model_dir / "feature_columns.json")

    summary: dict = {
        "feature_count": len(feature_cols),
        "targets": [],
        "reproducibility": {
            "random_state": RANDOM_STATE,
            "framework_resolved": resolved_fw,
            "lgbm_n_jobs": args.lgbm_n_jobs if resolved_fw == "lightgbm" else None,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }
    try:
        import sklearn

        summary["reproducibility"]["sklearn"] = sklearn.__version__
    except Exception:
        pass
    if resolved_fw == "lightgbm":
        try:
            import lightgbm as lgb

            summary["reproducibility"]["lightgbm"] = lgb.__version__
        except Exception:
            pass

    all_cv_rows = []
    for target in TARGETS:
        result = fit_one_target(
            df,
            feature_cols,
            target,
            model_dir,
            framework=resolved_fw,
            torch_epochs=args.torch_epochs,
            lgbm_n_jobs=args.lgbm_n_jobs,
        )
        summary["targets"].append({k: v for k, v in result.items() if k != "cv_rows"})
        all_cv_rows.extend(result["cv_rows"])

    cv_df = pd.DataFrame(all_cv_rows)
    cv_path = model_dir / "walk_forward_metrics.csv"
    cv_df.to_csv(cv_path, index=False)
    summary["metrics_path"] = str(cv_path)
    (model_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"训练完成，模型目录: {model_dir}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"验证指标: {cv_path}")
    print(cv_df.groupby(["target", "group"])[["mse", "mae", "direction_accuracy", "rank_ic"]].mean())


if __name__ == "__main__":
    main()
