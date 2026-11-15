from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12

ASSET_CANDIDATES = ["asset_id", "symbol", "code", "ticker", "id", "标的", "资产", "证券代码"]
TIME_CANDIDATES = ["timestamp", "datetime", "date", "time", "trade_time", "交易时间", "时间", "日期"]
OPEN_CANDIDATES = ["open", "Open", "OPEN", "开盘价", "开盘"]
HIGH_CANDIDATES = ["high", "High", "HIGH", "最高价", "最高"]
LOW_CANDIDATES = ["low", "Low", "LOW", "最低价", "最低"]
CLOSE_CANDIDATES = ["close", "Close", "CLOSE", "收盘价", "收盘"]
VOLUME_CANDIDATES = ["volume", "Volume", "VOL", "vol", "成交量", "交易量"]
TARGET_5_CANDIDATES = [
    "future_5_return",
    "target_5",
    "y_5",
    "return_5",
    "未来5分钟收益率",
    "未来_5_分钟收益率",
]
TARGET_60_CANDIDATES = [
    "future_60_return",
    "target_60",
    "y_60",
    "return_60",
    "未来60分钟收益率",
    "未来_60_分钟收益率",
]
NPY_COLUMNS_WITH_TARGETS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "future_5_return",
    "future_60_return",
]
NPY_COLUMNS_WITHOUT_TARGETS = ["timestamp", "open", "high", "low", "close", "volume"]


def find_column(columns: Iterable[str], candidates: Iterable[str], required: bool = True) -> str | None:
    """Return the first matching column using exact and case-insensitive names."""
    columns = list(columns)
    lower_map = {str(col).lower(): col for col in columns}
    for name in candidates:
        if name in columns:
            return name
        if str(name).lower() in lower_map:
            return lower_map[str(name).lower()]
    if required:
        raise ValueError(f"无法在数据中找到列，候选列名为: {list(candidates)}")
    return None


def infer_schema(df: pd.DataFrame) -> dict[str, str | None]:
    """Infer canonical OHLCV, asset, time and target columns."""
    schema = {
        "asset": find_column(df.columns, ASSET_CANDIDATES, required=False),
        "timestamp": find_column(df.columns, TIME_CANDIDATES, required=False),
        "open": find_column(df.columns, OPEN_CANDIDATES),
        "high": find_column(df.columns, HIGH_CANDIDATES),
        "low": find_column(df.columns, LOW_CANDIDATES),
        "close": find_column(df.columns, CLOSE_CANDIDATES),
        "volume": find_column(df.columns, VOLUME_CANDIDATES),
        "target_5": find_column(df.columns, TARGET_5_CANDIDATES, required=False),
        "target_60": find_column(df.columns, TARGET_60_CANDIDATES, required=False),
    }
    return schema


def _dataset_number(path: Path) -> int:
    match = re.search(r"dataset(\d+)", path.name)
    return int(match.group(1)) if match else 0


def _natural_npy_key(path: Path) -> tuple[int, str]:
    return (_dataset_number(path), path.name)


def _load_extreme_state_from_intervals(timestamps: np.ndarray, intervals_path: Path) -> np.ndarray:
    state = np.zeros(len(timestamps), dtype=np.int8)
    if not intervals_path.exists():
        return state

    intervals = np.load(intervals_path, allow_pickle=False)
    if intervals.size == 0:
        return state
    intervals = np.asarray(intervals).reshape(-1, 2)
    positions = pd.Series(np.arange(len(timestamps)), index=timestamps)

    for start, end in intervals:
        start = int(start)
        end = int(end)
        if start in positions.index and end in positions.index:
            start_pos = int(positions.loc[start])
            end_pos = int(positions.loc[end])
            state[start_pos : end_pos + 1] = 1
        else:
            state[(timestamps >= start) & (timestamps <= end)] = 1
    return state


def _read_npy_ohlcv(path: Path) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2 or arr.shape[1] not in {6, 8}:
        raise ValueError(
            f"{path} 的 shape 应为 (n, 6) 或 (n, 8)，实际为 {arr.shape}。"
            "列顺序需为 timestamp, open, high, low, close, volume[, future_5_return, future_60_return]"
        )

    columns = NPY_COLUMNS_WITH_TARGETS if arr.shape[1] == 8 else NPY_COLUMNS_WITHOUT_TARGETS
    df = pd.DataFrame(arr, columns=columns)
    asset_num = _dataset_number(path)
    df["asset_id"] = f"dataset{asset_num}"

    intervals_path = path.with_name(path.name.replace("_ohlcv.npy", "_extreme_intervals.npy"))
    df["provided_extreme_state"] = _load_extreme_state_from_intervals(
        df["timestamp"].astype(np.int64).to_numpy(),
        intervals_path,
    )
    return df


def read_npy_dataset_dir(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*_ohlcv.npy"), key=_natural_npy_key)
    if not files:
        raise FileNotFoundError(f"{path} 中没有找到 *_ohlcv.npy 文件")

    frames = [_read_npy_ohlcv(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        return read_npy_dataset_dir(path)

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _read_npy_ohlcv(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"暂不支持的数据格式: {path.suffix}，支持目录、.npy、.csv、.parquet、.xlsx")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"输出路径建议使用 .csv 或 .parquet: {path}")


def normalize_ohlcv_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    schema = infer_schema(df)
    rename_map = {
        schema["open"]: "open",
        schema["high"]: "high",
        schema["low"]: "low",
        schema["close"]: "close",
        schema["volume"]: "volume",
    }
    if schema["asset"] is not None:
        rename_map[schema["asset"]] = "asset_id"
    if schema["timestamp"] is not None:
        rename_map[schema["timestamp"]] = "timestamp"
    if schema["target_5"] is not None:
        rename_map[schema["target_5"]] = "future_5_return"
    if schema["target_60"] is not None:
        rename_map[schema["target_60"]] = "future_60_return"

    normalized = df.rename(columns=rename_map).copy()
    if "asset_id" not in normalized.columns:
        normalized["asset_id"] = "asset_0"
    if "timestamp" not in normalized.columns:
        normalized["timestamp"] = np.arange(len(normalized))

    for col in ["open", "high", "low", "close", "volume"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    normalized["missing_flag"] = normalized[["open", "high", "low", "close", "volume"]].isna().any(axis=1).astype(int)
    return normalized, schema


def _sort_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["asset_id", "timestamp"], keep="last")
    df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    price_cols = ["open", "high", "low", "close"]
    df[price_cols] = df.groupby("asset_id", group_keys=False)[price_cols].ffill()
    df[price_cols] = df.groupby("asset_id", group_keys=False)[price_cols].bfill()
    df["volume"] = df["volume"].fillna(0)
    return df


def _asset_numeric_code(asset_id: pd.Series) -> pd.Series:
    """Create a deterministic numeric asset code for dataset-style identifiers."""
    labels = asset_id.astype(str)
    extracted = labels.str.extract(r"dataset(\d+)", expand=False)
    codes = pd.to_numeric(extracted, errors="coerce")
    if codes.notna().all():
        return codes.astype("float32")

    categories = sorted(labels.dropna().unique())
    mapping = {label: idx for idx, label in enumerate(categories)}
    return labels.map(mapping).fillna(-1).astype("float32")


def _group_rolling_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=max(5, window // 3)).corr(y)


def _rolling_corr_by_asset(df: pd.DataFrame, window: int) -> pd.Series:
    corr = pd.Series(np.nan, index=df.index, dtype="float64")
    for _asset_id, part in df.groupby("asset_id", sort=False):
        corr.loc[part.index] = _group_rolling_corr(part["ret_1"], part["volume"], window).to_numpy()
    return corr


def generate_features(df: pd.DataFrame, create_targets: bool = False) -> pd.DataFrame:
    """Generate leakage-safe rolling-window features.

    All rolling statistics use current and historical rows only inside each asset.
    When create_targets=True, labels are generated from future close prices and
    must only be used for training or validation, never as prediction features.
    """
    df, _schema = normalize_ohlcv_schema(df)
    df = _sort_and_clean(df)
    df["asset_code"] = _asset_numeric_code(df["asset_id"])
    group = df.groupby("asset_id", group_keys=False)

    prev_close = group["close"].shift(1)
    df["ret_1"] = np.log((df["close"] + EPS) / (prev_close + EPS)).replace([np.inf, -np.inf], np.nan)

    for window in [3, 5, 10, 20, 30, 60, 120]:
        shifted = group["close"].shift(window)
        df[f"ret_{window}"] = np.log((df["close"] + EPS) / (shifted + EPS)).replace([np.inf, -np.inf], np.nan)

    for window in [5, 10, 20, 60]:
        ma = group["close"].transform(lambda s: s.rolling(window, min_periods=max(2, window // 3)).mean())
        df[f"ma_{window}"] = ma
        df[f"ma_dev_{window}"] = df["close"] / (ma + EPS) - 1

    df["ma_ratio_5_20"] = df["ma_5"] / (df["ma_20"] + EPS) - 1
    df["ma_ratio_20_60"] = df["ma_20"] / (df["ma_60"] + EPS) - 1

    for window in [5, 15, 30, 60, 120]:
        df[f"vol_{window}"] = group["ret_1"].transform(
            lambda s: s.rolling(window, min_periods=max(3, window // 3)).std()
        )

    df["vol_ratio_15_60"] = df["vol_15"] / (df["vol_60"] + EPS)
    df["vol_ratio_30_120"] = df["vol_30"] / (df["vol_120"] + EPS)

    df["range"] = (df["high"] - df["low"]) / (df["close"].abs() + EPS)
    true_range = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["true_range"] = true_range / (df["close"].abs() + EPS)
    for window in [5, 15, 60]:
        df[f"atr_{window}"] = group["true_range"].transform(
            lambda s: s.rolling(window, min_periods=max(3, window // 3)).mean()
        )

    for window in [5, 20, 60, 120]:
        df[f"volume_mean_{window}"] = group["volume"].transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 3)).mean()
        )
        df[f"volume_std_{window}"] = group["volume"].transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 3)).std()
        )

    df["volume_ratio_5_60"] = df["volume_mean_5"] / (df["volume_mean_60"] + EPS)
    df["volume_z_60"] = (df["volume"] - df["volume_mean_60"]) / (df["volume_std_60"] + EPS)
    df["impact"] = df["ret_1"].abs() / np.sqrt(df["volume"].clip(lower=0) + 1)

    df["corr_ret_volume_20"] = _rolling_corr_by_asset(df, 20)
    df["corr_ret_volume_60"] = _rolling_corr_by_asset(df, 60)

    df["body"] = (df["close"] - df["open"]) / (df["open"].abs() + EPS)
    df["upper_shadow"] = (df["high"] - np.maximum(df["open"], df["close"])) / (df["open"].abs() + EPS)
    df["lower_shadow"] = (np.minimum(df["open"], df["close"]) - df["low"]) / (df["open"].abs() + EPS)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / ((df["high"] - df["low"]).abs() + EPS)

    df["extreme_ret"] = (df["ret_1"].abs() > 3 * df["vol_60"].fillna(np.inf)).astype(int)
    df["vol_15_q90_120"] = group["vol_15"].transform(
        lambda s: s.rolling(120, min_periods=40).quantile(0.9)
    )
    df["extreme_vol"] = (df["vol_15"] > df["vol_15_q90_120"]).astype(int)
    df["extreme_volume"] = (df["volume_z_60"] > 2).astype(int)
    extreme_cols = ["extreme_ret", "extreme_vol", "extreme_volume"]
    if "provided_extreme_state" in df.columns:
        extreme_cols.append("provided_extreme_state")
    df["extreme_state"] = df[extreme_cols].max(axis=1)
    df["sample_weight"] = 1.0 + 2.0 * df["extreme_state"]

    if df["timestamp"].nunique() > 1 and df["asset_id"].nunique() > 1:
        df["cross_mean_ret"] = df.groupby("timestamp")["ret_1"].transform("mean")
        df["relative_ret"] = df["ret_1"] - df["cross_mean_ret"]
        df["rank_ret"] = df.groupby("timestamp")["ret_1"].rank(pct=True)
        df["rank_volume"] = df.groupby("timestamp")["volume_z_60"].rank(pct=True)
        df["rank_volatility"] = df.groupby("timestamp")["vol_15"].rank(pct=True)
    else:
        df["cross_mean_ret"] = df["ret_1"]
        df["relative_ret"] = 0.0
        df["rank_ret"] = 0.5
        df["rank_volume"] = 0.5
        df["rank_volatility"] = 0.5

    if create_targets:
        for horizon in [5, 60]:
            future_close = group["close"].shift(-horizon)
            df[f"future_{horizon}_return"] = np.log((future_close + EPS) / (df["close"] + EPS))

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    no_clip_cols = {
        "timestamp",
        "future_5_return",
        "future_60_return",
        "sample_weight",
        "missing_flag",
        "provided_extreme_state",
        "extreme_ret",
        "extreme_vol",
        "extreme_volume",
        "extreme_state",
        "asset_code",
    }
    for col in numeric_cols:
        values = df[col].to_numpy(copy=False)
        if np.issubdtype(values.dtype, np.floating):
            finite_mask = np.isfinite(values)
            if not finite_mask.all():
                df[col] = df[col].where(finite_mask, np.nan)
        if col in no_clip_cols:
            if col != "timestamp" and pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype("float32")
            continue
        low, high = df[col].quantile([0.001, 0.999])
        if pd.notna(low) and pd.notna(high) and low < high:
            df[col] = df[col].clip(low, high)
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "asset_id",
        "timestamp",
        "provided_extreme_state",
        "future_5_return",
        "future_60_return",
        "sample_weight",
    }
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def save_feature_columns(feature_cols: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")


def load_feature_columns(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 C 题 Rolling Window 特征")
    parser.add_argument("--input", required=True, help="原始训练或测试数据路径，支持 train_dataset 目录、npy、csv、parquet、xlsx")
    parser.add_argument("--output", required=True, help="特征输出路径，建议 csv 或 parquet")
    parser.add_argument("--create-targets", action="store_true", help="当原始训练集没有标签时，用未来收盘价生成标签")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.is_dir():
        files = sorted(input_path.glob("*_ohlcv.npy"), key=_natural_npy_key)
        if not files:
            raise FileNotFoundError(f"{input_path} 中没有找到 *_ohlcv.npy 文件")
        output_dir = output_path if not output_path.suffix else output_path.with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            raw = read_table(file)
            features = generate_features(raw, create_targets=args.create_targets)
            one_output = output_dir / file.name.replace("_ohlcv.npy", "_features.parquet")
            write_table(features, one_output)
            print(f"已生成特征: {one_output}, shape={features.shape}")
        return

    raw = read_table(input_path)
    features = generate_features(raw, create_targets=args.create_targets)
    write_table(features, output_path)
    print(f"已生成特征: {output_path}, shape={features.shape}")


if __name__ == "__main__":
    main()
