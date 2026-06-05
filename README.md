# AI 从业者年薪预测 — C 语言实现

本项目使用 C 语言实现一个简单的回归模型，基于 `ai_jobs_market_2025_2026.csv` 数据，对 AI 岗位从业者的年薪 `annual_salary_usd` 进行预测。

数据来源：
- Kaggle 数据集：https://www.kaggle.com/datasets/alitaqishah/ai-jobs-market-2025-2026-salaries

## 项目目的

本项目目标是：

1. 读取 CSV 数据文件
2. 仅保留以下字段作为训练数据：
   - `years_of_experience`
   - `experience_level`
   - `education_required`
   - `job_category`
   - `remote_work`
   - `company_size`
   - `country`
   - `industry`
   - `annual_salary_usd`
3. 清洗数据：检查并剔除缺失值、无效值与重复行
4. 对分类特征做 one-hot 编码，对数值特征做标准化
5. 实现带 L2 正则的线性回归（Ridge）模型
6. 评估模型性能：MAE、RMSE、R²
7. 支持自定义特征输入并预测年薪

## 目录结构

- `1.c` — 项目主程序，包含：
  - CSV 读取
  - 数据清洗
  - 特征编码
  - 训练
  - 评估
  - 主函数
- `ai_jobs_market_2025_2026.csv` — 原始数据文件
- `README.md` — 本说明文档

## 使用前提

你需要在 Windows 下安装 GCC 或 MinGW 编译器。

如果尚未安装，请先安装：
- MinGW
- 或者其他支持 `gcc` 的 C 编译工具链

## 运行步骤

1. 将 `1.c` 和 `ai_jobs_market_2025_2026.csv` 放在同一目录下。
2. 打开命令行终端（如 CMD 或 PowerShell）。
3. 切换到项目目录：

```bash
cd /d d:\ALL_kinds_apps\GitRepository\AI_job_salary_dictator
```

4. 编译程序：

```bash
gcc -o salary_predictor 1.c -lm
```

5. 运行程序：

```bash
salary_predictor
```

## 程序输出内容说明

程序运行后会输出：

- 读取有效数据样本数量
- 丢弃的缺失值、无效数据和重复行数量
- 各分类特征的 one-hot 编码类别映射
- 训练过程中的 MSE 损失
- 训练集和测试集上的评估指标：
  - MAE（平均绝对误差）
  - RMSE（均方根误差）
  - R²（决定系数）
- 模型前几个权重示例
- 交互式输入步骤，用于预测自定义特征的年薪

## 自定义预测

程序结束时会进入自定义预测环节，提示你输入：

1. `years_of_experience`
2. `experience_level`（从程序输出的编号中选择）
3. `education_required`
4. `job_category`
5. `remote_work`
6. `company_size`
7. `country`
8. `industry`

输入完成后，程序会输出预测的 `annual_salary_usd`。

## 问题排查

- 如果出现 `无法打开文件`，请确认 `ai_jobs_market_2025_2026.csv` 与程序在同一目录。
- 如果出现编译错误，请检查你的 GCC 是否已正确安装，并确认命令行已切换到项目目录。
- 如果运行后评估指标较差，说明当前数据的非线性关系较强，线性模型可能不足，可以考虑更复杂的模型。

## 说明

本项目适合学习数据预处理与经典回归流程的入门演示。它把所有代码保存在一个单文件 `1.c` 中，便于直接编译和运行。