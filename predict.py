from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from feature_engineering import generate_features, read_table


TARGETS = ["future_5_return", "future_60_return"]
PREDICT_COLS = {
    "future_5_return": "pred_5_return",
    "future_60_return": "pred_60_return",
}


def prepare_features(df: pd.DataFrame, feature_cols: list[str], fill_values: dict[str, float]) -> pd.DataFrame:
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    x = df[feature_cols].copy()
    for col in x.columns:
        values = x[col].to_numpy(copy=False)
        if np.issubdtype(values.dtype, np.floating):
            finite_mask = np.isfinite(values)
            if not finite_mask.all():
                x[col] = x[col].where(finite_mask, np.nan)
    return x.fillna(fill_values).fillna(0).astype("float32")


def predict_one(df: pd.DataFrame, bundle_path: Path) -> np.ndarray:
    bundle = joblib.load(bundle_path)
    feature_cols = bundle["feature_cols"]
    x = prepare_features(df, feature_cols, bundle["fill_values"])

    main_pred = bundle["model"].predict(x)
    ridge_model = bundle.get("ridge_model")
    weights = bundle.get("blend_weight", {"main": 1.0, "ridge": 0.0})
    pred = weights.get("main", 1.0) * main_pred
    if ridge_model is not None and weights.get("ridge", 0.0) != 0:
        pred = pred + weights.get("ridge", 0.0) * ridge_model.predict(x)

    reversion_weight = weights.get("reversion", 0.0)
    reversion_feature = bundle.get("reversion_feature")
    if reversion_weight != 0 and reversion_feature in x.columns:
        pred = pred - reversion_weight * x[reversion_feature].to_numpy()
    pred = pred - float(bundle.get("direction_threshold", 0.0))
    return pred


def build_output(raw: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=feature_df.index)
    for col in ["row_id", "id", "asset_id", "timestamp"]:
        if col in feature_df.columns:
            output[col] = feature_df[col]
        elif col in raw.columns and len(raw) == len(feature_df):
            output[col] = raw[col].to_numpy()

    if output.empty:
        output["row_id"] = np.arange(len(feature_df))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="使用已训练模型预测未来 5/60 分钟收益率")
    parser.add_argument("--input", required=True, help="测试数据路径，支持 csv/parquet/xlsx")
    parser.add_argument("--model-dir", default="models", help="训练脚本输出的模型目录")
    parser.add_argument("--output", default="submission.csv", help="预测结果输出路径")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    raw = read_table(args.input)
    feature_df = generate_features(raw, create_targets=False)
    output = build_output(raw, feature_df)

    for target in TARGETS:
        model_path = model_dir / f"model_{target}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"未找到模型文件: {model_path}")
        output[PREDICT_COLS[target]] = predict_one(feature_df.copy(), model_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"预测完成: {output_path}, shape={output.shape}")


if __name__ == "__main__":
    main()
