# OAF 代码包

从本次完整交付包提取的训练、评价及绘图代码。 原目录关系保留，训练实现没有改写。

## 目录与入口

| 路径 | 用途 |
|---|---|
| `project/primary_benchmark/` | 六模型、A–E 分解、融合对照及相关分析 |
| `project/nested_validation/` | 三种子完整嵌套验证与源院区校准 |
| `project/START.py` | 默认复核已保存结果；顶部 `MODE` 可切换训练流程 |
| `project/TRAIN_ALL.py` | 完整嵌套重训入口 |
| `project/supplemental_experiments_package/` | 模拟与补充压力测试 |
| `project/exploratory_gate_redesign/` | 探索性改造试验，区别于论文主方法 |
| `plotting/START.py` | 生成当前图2–5及补充图 |

## 使用方法

建议使用原复现环境 **Python 3.12**。各流程依赖分别保存在对应目录的 `requirements.txt`，不要混用环境。

先从之前完整包恢复加工数据、冻结划分和保存结果：

```bash
python RESTORE_DATA.py "路径/OAF_Full_Delivery_20261005.zip"
```

安装主实验依赖并运行：

```bash
python -m pip install -r project/primary_benchmark/requirements.txt
python project/primary_benchmark/START.py --processed project/primary_benchmark/processed_private
```

运行完整嵌套重训：

```bash
python project/TRAIN_ALL.py
```

仅重新绘图：

```bash
python -m pip install -r plotting/requirements.txt
python plotting/START.py
```

仅复核已有结果：

```bash
python project/START.py
```

**仅解压代码包不能直接得到结果，需要先恢复数据。** `RESTORE_DATA.py` 只复制输入和保存结果，不训练模型。重训需时较长；本次打包没有重新拟合模型。原入口可能自动安装缺失依赖。

未使用完整包时，需自行提供相同格式的 `patients.csv`、`events.csv`、`landmarks.csv`、`xuhui_folds.csv`，以及代码依赖的 `reference_folds.csv`；复核与绘图还需要对应预测/汇总文件。不要用其他队列的划分替换论文冻结划分。

本次检查：所有 Python 文件通过语法解析，压缩包完整性通过；未逐一重跑训练流程。
