# AI 从业者年薪预测 — C 语言实现

基于 Ridge 线性回归（L2 正则化），使用 7 个特征预测 AI 岗位的 `annual_salary_usd`。

数据来源：[Kaggle — AI Jobs Market 2025-2026 Salaries](https://www.kaggle.com/datasets/alitaqishah/ai-jobs-market-2025-2026-salaries)

---

## 快速开始

### 1. 环境要求

- **GCC**（MinGW / MSYS2 / WSL 均可）
- 项目目录下同时存在 `1.c` 和 `ai_jobs_market_2025_2026.csv`

### 2. 编译

```bash
gcc -Wall -Wextra -std=c11 -o salary_predictor 1.c -lm
```

### 3. 运行

**交互模式**（手动输入特征）：

```bash
./salary_predictor
```

**非交互模式**（管道传入 7 个特征值，跳过手动输入）：

```bash
echo "7 2 1 0 2 0 1" | ./salary_predictor
```

---

## 程序输出解读

程序运行后依次输出以下内容：

### ① 数据加载报告

```
读取到 1499 条有效样本。
丢弃 0 条缺失值样本。
丢弃 0 条无效数值样本。
丢弃 1 条重复样本。
```

### ② 特征编码映射

每个分类特征的类别列表及其 one-hot 索引编号。**经验等级和学历要求已按从低到高排序**：

```
experience_level categories (4):
  [ 0] Entry (0-2 yrs) (385)
  [ 1] Mid (3-5 yrs) (370)
  [ 2] Senior (6-9 yrs) (363)
  [ 3] Lead (10+ yrs) (381)

education_required categories (5):
  [ 0] Bootcamp/Self-taught (297)
  [ 1] Associate's (296)
  [ 2] Bachelor's (311)
  [ 3] Master's (316)
  [ 4] PhD (279)
```

括号内数字是该类别在数据集中的样本数。**记住这里的索引号，交互预测时需要用到。**

### ③ 训练过程

```
Epoch 1/2000: MSE=42689220583.82
Epoch 200/2000: MSE=1821253566.24
...
Epoch 2000/2000: MSE=1084151775.73
```

MSE 持续下降说明模型在稳定收敛。

### ④ 模型评估

```
训练集评估: MAE=25823.57, RMSE=32925.96, R²=0.7555
测试集评估: MAE=26404.25, RMSE=34453.33, R²=0.7239
```

| 指标 | 含义 | 判断标准 |
|---|---|---|
| MAE | 平均绝对误差（美元） | 越小越好 |
| RMSE | 均方根误差（美元） | 越小越好，对离群值敏感 |
| R² | 决定系数 | 越接近 1 越好，>0.7 为可用 |

训练集与测试集指标接近 → 无过拟合；R² ≈ 0.72 → 模型能解释约 72% 的年薪方差。

---

## 交互预测：输入什么数据？

程序最后会要求你依次输入 **7 个特征值**（1 个数值 + 6 个分类索引）。**你需要根据前面打印的类别列表，输入对应的索引号。**

### 完整输入示例

假设你要预测这样一个 AI 从业者的年薪：

| 特征 | 你的选择 | 对应索引 |
|---|---|---|
| 工作年限 | 7 年 | 直接输入 `7` |
| 经验等级 | Senior (6-9 yrs) | `2` |
| 学历 | Associate's | `1` |
| 岗位类别 | AI Engineering | `0` |
| 公司规模 | Big Tech (FAANG+) | `2` |
| 国家 | USA | `0` |
| 行业 | Technology | `1` |

**在终端中依次输入：**

```
7
2
1
0
2
0
1
```

**程序输出：**

```
预测年薪 annual_salary_usd = 240875.91 USD
```

### 全部类别索引速查

<details>
<summary><b>experience_level（经验等级）— 从低到高排列</b></summary>

| 索引 | 类别 | 样本数 |
|---|---|---|
| 0 | Entry (0-2 yrs) | 385 |
| 1 | Mid (3-5 yrs) | 370 |
| 2 | Senior (6-9 yrs) | 363 |
| 3 | Lead (10+ yrs) | 381 |

</details>

<details>
<summary><b>education_required（学历要求）— 从低到高排列</b></summary>

| 索引 | 类别 | 样本数 |
|---|---|---|
| 0 | Bootcamp/Self-taught | 297 |
| 1 | Associate's | 296 |
| 2 | Bachelor's | 311 |
| 3 | Master's | 316 |
| 4 | PhD | 279 |

</details>

<details>
<summary><b>job_category（岗位类别）</b></summary>

| 索引 | 类别 | 样本数 |
|---|---|---|
| 0 | AI Engineering | 735 |
| 1 | Data Engineering | 51 |
| 2 | Product | 70 |
| 3 | Security | 50 |
| 4 | Architecture | 52 |
| 5 | ML Operations | 51 |
| 6 | Business | 62 |
| 7 | Robotics | 74 |
| 8 | Data Science | 127 |
| 9 | Governance | 122 |
| 10 | Infrastructure | 55 |
| 11 | Research | 50 |

</details>

<details>
<summary><b>company_size（公司规模）— 从小到大排列</b></summary>

| 索引 | 类别 | 样本数 |
|---|---|---|
| 0 | Startup (1-50) | 292 |
| 1 | SME (51-500) | 300 |
| 2 | Mid-size (501-5000) | 312 |
| 3 | Enterprise (5000+) | 296 |
| 4 | Big Tech (FAANG+) | 299 |

</details>

<details>
<summary><b>country（国家）</b></summary>

| 索引 | 类别 | 样本数 |
|---|---|---|
| 0 | USA | 514 |
| 1 | UK | 90 |
| 2 | Singapore | 71 |
| 3 | Global | 82 |
| 4 | India | 57 |
| 5 | Japan | 76 |
| 6 | China | 87 |
| 7 | Canada | 85 |
| 8 | UAE | 62 |
| 9 | Netherlands | 74 |
| 10 | France | 66 |
| 11 | Australia | 78 |
| 12 | Germany | 81 |
| 13 | Switzerland | 76 |

</details>

<details>
<summary><b>industry（行业）</b></summary>

| 索引 | 类别 | 样本数 |
|---|---|---|
| 0 | Finance | 131 |
| 1 | Technology | 106 |
| 2 | Automotive | 138 |
| 3 | Government | 136 |
| 4 | Manufacturing | 114 |
| 5 | Education | 120 |
| 6 | Retail | 131 |
| 7 | Consulting | 126 |
| 8 | Research | 109 |
| 9 | Healthcare | 138 |
| 10 | Media | 120 |
| 11 | Energy | 130 |

</details>

---

## 使用的特征（共 7 个）

| 特征 | 类型 | 处理方式 |
|---|---|---|
| `years_of_experience` | 数值 | Z-score 标准化 |
| `experience_level` | 分类 (4 类) | One-hot 编码，从低到高 |
| `education_required` | 分类 (5 类) | One-hot 编码，从低到高 |
| `job_category` | 分类 (12 类) | One-hot 编码 |
| `company_size` | 分类 (5 类) | One-hot 编码 |
| `country` | 分类 (14 类) | One-hot 编码 |
| `industry` | 分类 (12 类) | One-hot 编码 |

总计：1 个数值特征 + 52 个 one-hot 特征 = **53 维特征向量**。

> **注意：** `remote_work`（办公模式）已从特征中移除，不作为训练输入。

## 未使用的字段（避免数据泄漏）

以下字段在训练中被排除：

`job_id`, `job_title`, `salary_min_usd`, `salary_max_usd`, `salary_tier`, `ai_salary_premium_pct`, `demand_score`, `demand_growth_yoy_pct`, `benefits_score_10`, `posting_year`, `posting_month`, `is_senior`, `is_remote_friendly`, `is_llm_role`, `city`, `required_skills`, `remote_work`

---

## 模型参数

| 参数 | 值 | 说明 |
|---|---|---|
| 算法 | Ridge 线性回归 | L2 正则化线性模型 |
| 优化器 | 批量梯度下降 | 全量数据计算梯度 |
| 训练轮数 | 2000 | 已稳定收敛 |
| 学习率 | 0.01 | 梯度下降步长 |
| L2 正则化系数 λ | 0.1 | 防止权重过大 |
| 训练/测试划分 | 8:2 | 随机打乱后划分 |
| 随机种子 | 123456 | 可复现结果 |
| 低频类别阈值 | 2 | 出现次数 <2 的类别合并为 "Other" |

---

## 项目文件

| 文件 | 用途 |
|---|---|
| `1.c` | 完整源码（CSV 读取 → 清洗 → 编码 → 训练 → 评估 → 预测） |
| `ai_jobs_market_2025_2026.csv` | 原始数据（1500 条 AI 岗位记录） |
| `README.md` | 本说明文档 |

---

## 常见问题

**Q: 提示 "无法打开文件"？**
A: 确认 `ai_jobs_market_2025_2026.csv` 与可执行文件在同一目录下。

**Q: 编译报错？**
A: 确认 GCC 已安装（`gcc --version`），且命令行当前目录正确。

**Q: R² 不够高？**
A: 线性模型只能捕捉线性关系。数据中存在非线性模式时，可以尝试多项式特征、决策树、神经网络等更复杂的模型。

**Q: 如何批量预测多条数据？**
A: 目前程序只支持逐条交互预测。如需批量预测，可以修改 `main()` 函数，去掉 `predict_custom_example` 调用，改为循环读取文件中的特征行。

