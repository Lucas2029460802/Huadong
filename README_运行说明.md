# C题收益率预测程序运行说明

- `feature_engineering.py`：数据清洗、Rolling Window 特征、极端行情标记、样本权重。
- `train.py`：训练未来 5 分钟和未来 60 分钟收益率模型，并做 Walk-forward 验证。
- `predict.py`：加载模型并生成预测结果。
- `requirements.txt`：Python 依赖。

## 1. 安装依赖

在 PowerShell 中进入项目目录：

```powershell
cd d:\huadong
```

建议创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
pip install -r requirements.txt
```

如果 `lightgbm` 安装失败，训练脚本会自动回退到 `sklearn` 的梯度提升模型，但建议优先安装成功。

如果要使用 PyTorch 模型，额外安装：

```powershell
pip install torch
```

## 2. 生成特征文件

检查特征表运行：

```powershell
python feature_engineering.py --input train_dataset --output outputs\train_features
```

注意：完整 30 个数据集较大，程序会逐个 `dataset` 输出独立的 `parquet` 文件到 `outputs\train_features` 目录。如果只是想快速检查单个标的，可运行：

```powershell
python feature_engineering.py --input train_dataset\dataset0_train_ohlcv.npy --output outputs\dataset0_features.parquet
```

## 4. 训练模型

使用 LightGBM：

```powershell
python train.py --input train_dataset --model-dir models
```

训练完成后会生成：

- `models\model_future_5_return.joblib`
- `models\model_future_60_return.joblib`
- `models\feature_columns.json`
- `models\walk_forward_metrics.csv`
- `models\training_summary.json`
- `models\feature_importance_*.csv`

其中 `walk_forward_metrics.csv` 用于总体、普通行情、极端行情分组结果分析。

## 5. 生成预测结果

```powershell
python predict.py --input test_dataset --model-dir models --output submission.csv
```

输出文件默认包含：

- `asset_id`
- `timestamp`
- `pred_5_return`
- `pred_60_return`


## 6. 复现赛题要求

**随机种子 2026**、`LightGBM` **单线程**（`--lgbm-n-jobs 1`）、`force_col_wise`、并在进程内限制常见 BLAS 线程数为 1。`training_summary.json` 中会记录 `random_state`、实际框架版本等信息。


```powershell
$env:PYTHONHASHSEED=0
```

```powershell
python train.py --input train_dataset --model-dir models --framework lightgbm --lgbm-n-jobs -1
```
