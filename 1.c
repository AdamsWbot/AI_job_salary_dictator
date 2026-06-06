#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stddef.h>
#include <math.h>

// Windows 下设置控制台为 UTF-8，避免中文乱码
#ifdef _WIN32
#include <windows.h>
#endif

// 最大文本行长度，用于读取 CSV 行
#define MAX_LINE 8192
// CSV 最多字段数，用于拆分行
#define MAX_FIELDS 64
// 支持的最大样本数
#define MAX_ROWS 10000
// 支持的最大类别数量，用于统计 one-hot 特征
#define MAX_CAT_VALUES 512
// 允许的最大总特征数（包括数值特征和所有 one-hot 特征）
#define MAX_FEATURES 1024
// 低频类别阈值，小于该数量的类别会被归并为 "Other"
#define RARE_CATEGORY_THRESHOLD 2

// 用于保存单个分类值的统计信息
typedef struct {
    char *value;        // 类别名称，例如 "USA" 或 "Senior (6-9 yrs)"
    int count;          // 该类别在训练集中出现的次数
    int keep;           // 是否保留该类别（低频类别会被归并为 Other）
    int mapped_index;   // one-hot 编码后对应的特征索引
} Category;

// 保存一条数据记录的结构
typedef struct {
    double years_of_experience;
    double annual_salary_usd;
    char *experience_level;
    char *education_required;
    char *job_category;
    char *company_size;
    char *country;
    char *industry;
} Row;

// 复制字符串，类似于 strdup，但更兼容标准 C
static char *strdup_c(const char *text) {
    if (!text) return NULL;
    size_t len = strlen(text);
    char *copy = malloc(len + 1);
    if (!copy) return NULL;
    memcpy(copy, text, len + 1);
    return copy;
}

// 去掉字符串前后空白字符，并返回处理后的指针
static char *trim(char *s) {
    if (!s) return s;
    char *start = s;
    while (*start && isspace((unsigned char)*start)) start++;
    char *end = start + strlen(start);
    while (end > start && isspace((unsigned char)*(end - 1))) end--;
    *end = '\0';
    if (start != s) memmove(s, start, end - start + 1);
    return s;
}

// 解析一行 CSV 内容，将每个字段拆分到 fields 数组中。
// 支持双引号包含的字段和字段内的双引号转义。
static int parse_csv_line(const char *line, char **fields, int max_fields) {
    int field = 0;
    int in_quotes = 0;
    char buf[MAX_LINE];
    int b = 0;

    for (int i = 0; line[i] != '\0' && field < max_fields; ++i) {
        char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (line[i + 1] == '"') {
                    buf[b++] = '"';
                    i++;
                } else {
                    in_quotes = 0;
                }
            } else {
                buf[b++] = c;
            }
        } else {
            if (c == '"') {
                in_quotes = 1;
            } else if (c == ',') {
                buf[b] = '\0';
                trim(buf);
                fields[field++] = strdup_c(buf);
                b = 0;
            } else {
                buf[b++] = c;
            }
        }
        if (b >= MAX_LINE - 1) break;
    }
    if (field < max_fields) {
        buf[b] = '\0';
        trim(buf);
        fields[field++] = strdup_c(buf);
    }
    return field;
}

// 便携的忽略大小写字符串比较（避免 strcasecmp 的平台兼容性问题）
static int stricmp_portable(const char *a, const char *b) {
    if (!a || !b) return (a == b) ? 0 : (a ? 1 : -1);
    while (*a && *b) {
        int ca = tolower((unsigned char)*a);
        int cb = tolower((unsigned char)*b);
        if (ca != cb) return ca - cb;
        a++; b++;
    }
    return tolower((unsigned char)*a) - tolower((unsigned char)*b);
}

// 判断字符串是否为空值，例如空字符串、空格、NA、N/A、null
static int is_missing_value(const char *s) {
    if (!s) return 1;
    while (*s && isspace((unsigned char)*s)) s++;
    if (*s == '\0') return 1;
    if (stricmp_portable(s, "NA") == 0 || stricmp_portable(s, "N/A") == 0 || stricmp_portable(s, "null") == 0) return 1;
    return 0;
}

// 在类别数组中查找给定分类值的索引
static int category_lookup(Category *cats, int count, const char *value) {
    for (int i = 0; i < count; ++i) {
        if (strcmp(cats[i].value, value) == 0) return i;
    }
    return -1;
}

// 将给定分类值插入到类别数组中，或如果已存在则增加计数
static int category_add(Category *cats, int *count, const char *value) {
    int idx = category_lookup(cats, *count, value);
    if (idx >= 0) {
        cats[idx].count += 1;
        return idx;
    }
    if (*count >= MAX_CAT_VALUES) return -1;
    cats[*count].value = strdup_c(value);
    cats[*count].count = 1;
    cats[*count].keep = 0;
    cats[*count].mapped_index = -1;
    return (*count)++;
}

// 在类别数组中查找 "Other" 类别的索引（如果存在）
static int find_other_index(Category *cats, int count) {
    for (int i = 0; i < count; ++i) {
        if (strcmp(cats[i].value, "Other") == 0) return i;
    }
    return -1;
}

// ── 类别排序：按语义从低到高排列 ──────────────────────────────────

// 返回 experience_level 的排序权重（从低到高）
static int exp_level_order(const char *value) {
    if (strstr(value, "Entry"))     return 0;
    if (strstr(value, "Mid"))       return 1;
    if (strstr(value, "Senior"))    return 2;
    if (strstr(value, "Lead"))      return 3;
    return 99; // 未知值排到最后
}

// 返回 education_required 的排序权重（从低到高）
static int education_order(const char *value) {
    if (strstr(value, "Bootcamp") || strstr(value, "Self-taught")) return 0;
    if (strstr(value, "Associate")) return 1;
    if (strstr(value, "Bachelor"))  return 2;
    if (strstr(value, "Master"))    return 3;
    if (strstr(value, "PhD"))       return 4;
    return 99;
}

// 通用类别排序比较函数：根据 compare_func 的返回值升序排列
static int category_cmp(const void *a, const void *b, int (*order_func)(const char*)) {
    const Category *ca = (const Category *)a;
    const Category *cb = (const Category *)b;
    int oa = order_func(ca->value);
    int ob = order_func(cb->value);
    if (oa != ob) return oa - ob;
    return strcmp(ca->value, cb->value);
}

static int cmp_exp(const void *a, const void *b) { return category_cmp(a, b, exp_level_order); }
static int cmp_edu(const void *a, const void *b) { return category_cmp(a, b, education_order); }

// 根据类别频率决定是否保留类别，并为保留类别分配 one-hot 索引。
// 低于 min_count 的类别会被标记为不保留，并在后续映射到 "Other"。
static void build_category_mapping(Category *cats, int *count, int min_count) {
    int other_index = find_other_index(cats, *count);
    int rare_found = 0;

    for (int i = 0; i < *count; ++i) {
        if (cats[i].count < min_count && strcmp(cats[i].value, "Other") != 0) {
            rare_found = 1;
            cats[i].keep = 0;
        } else {
            cats[i].keep = 1;
        }
    }

    if (rare_found && other_index < 0) {
        int idx = category_add(cats, count, "Other");
        if (idx >= 0) {
            cats[idx].keep = 1;
            cats[idx].count = 0;
            other_index = idx;
        }
    }

    int mapped = 0;
    for (int i = 0; i < *count; ++i) {
        if (cats[i].keep) {
            cats[i].mapped_index = mapped++;
        } else {
            cats[i].mapped_index = -1;
        }
    }
}

// 将分类值映射到 one-hot 特征索引，如果该类别被归并为 Other，则返回 Other 的索引。
static int map_category_value(Category *cats, int count, const char *value) {
    int idx = category_lookup(cats, count, value);
    if (idx >= 0 && cats[idx].keep) {
        return cats[idx].mapped_index;
    }
    int other_idx = find_other_index(cats, count);
    if (other_idx >= 0) return cats[other_idx].mapped_index;
    if (count > 0) return 0;
    return -1;
}

// 释放存储在行结构中的动态字符串内存
static void free_rows(Row *rows, int row_count) {
    for (int i = 0; i < row_count; ++i) {
        free(rows[i].experience_level);
        free(rows[i].education_required);
        free(rows[i].job_category);
        free(rows[i].company_size);
        free(rows[i].country);
        free(rows[i].industry);
    }
}

// 释放类别数组中保存的字符串
static void free_categories(Category *cats, int count) {
    for (int i = 0; i < count; ++i) {
        free(cats[i].value);
    }
}

// 安全地将字符串转换为 double，错误输入返回 0
static int safe_parse_double(const char *text, double *out) {
    if (is_missing_value(text)) return 0;
    char *endptr = NULL;
    double value = strtod(text, &endptr);
    if (endptr == text || *endptr != '\0') return 0;
    *out = value;
    return 1;
}

// 检查两个样本是否在保留字段上完全相同，用于过滤重复行
static int rows_are_duplicate(const Row *a, const Row *b) {
    if (a->years_of_experience != b->years_of_experience) return 0;
    if (a->annual_salary_usd != b->annual_salary_usd) return 0;
    if (strcmp(a->experience_level, b->experience_level) != 0) return 0;
    if (strcmp(a->education_required, b->education_required) != 0) return 0;
    if (strcmp(a->job_category, b->job_category) != 0) return 0;
    if (strcmp(a->company_size, b->company_size) != 0) return 0;
    if (strcmp(a->country, b->country) != 0) return 0;
    if (strcmp(a->industry, b->industry) != 0) return 0;
    return 1;
}

// 随机打乱索引数组，用于划分训练集和测试集
static void shuffle_indices(int *indices, int n) {
    for (int i = n - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

// 计算给定索引集合的均值和标准差，用于特征标准化
static void compute_normalization(const double *values, const int *indices, int n, double *mean_out, double *std_out) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += values[indices[i]];
    }
    double mean = sum / n;
    double variance = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = values[indices[i]] - mean;
        variance += diff * diff;
    }
    variance /= (n > 1 ? (n - 1) : 1);
    double std = sqrt(variance);
    if (std < 1e-8) std = 1.0;
    *mean_out = mean;
    *std_out = std;
}

// 构建特征矩阵 X：每一行对应一个样本，第一列是标准化后的 years_of_experience，后面依次是分类特征的 one-hot 编码。
static void build_feature_matrix(double *X, const double *years, int n, double mean, double std,
                                 int exp_idx[], int exp_count,
                                 int edu_idx[], int edu_count,
                                 int job_idx[], int job_count,
                                 int company_idx[], int company_count,
                                 int country_idx[], int country_count,
                                 int industry_idx[], int industry_count,
                                 int total_features) {
    (void)industry_count; /* 最后一组类别数，无需偏移 */
    for (int i = 0; i < n; ++i) {
        double *row = X + ((ptrdiff_t)i * total_features);
        for (int j = 0; j < total_features; ++j) row[j] = 0.0;
        row[0] = (years[i] - mean) / std;
        int offset = 1;
        if (exp_idx[i] >= 0) row[offset + exp_idx[i]] = 1.0;
        offset += exp_count;
        if (edu_idx[i] >= 0) row[offset + edu_idx[i]] = 1.0;
        offset += edu_count;
        if (job_idx[i] >= 0) row[offset + job_idx[i]] = 1.0;
        offset += job_count;
        if (company_idx[i] >= 0) row[offset + company_idx[i]] = 1.0;
        offset += company_count;
        if (country_idx[i] >= 0) row[offset + country_idx[i]] = 1.0;
        offset += country_count;
        if (industry_idx[i] >= 0) row[offset + industry_idx[i]] = 1.0;
    }
}

// 训练 Ridge 线性回归模型。
// 使用批量梯度下降更新参数，并在权重上添加 L2 正则化，降低过拟合风险。
static void ridge_train(const double *X, const double *y, int n, int dim, double *weights,
                        int epochs, double learning_rate, double lambda) {
    for (int j = 0; j <= dim; ++j) weights[j] = 0.0;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // 计算每个参数的梯度
        double bias_grad = 0.0;
        double *weight_grads = calloc(dim, sizeof(double));
        if (!weight_grads) {
            fprintf(stderr, "内存分配失败\n");
            exit(EXIT_FAILURE);
        }
        double mse = 0.0;

        for (int i = 0; i < n; ++i) {
            const double *xrow = X + ((ptrdiff_t)i * dim);
            double prediction = weights[0];
            for (int j = 0; j < dim; ++j) {
                prediction += weights[j + 1] * xrow[j];
            }
            double error = prediction - y[i];
            mse += error * error;
            bias_grad += 2.0 * error / n;
            for (int j = 0; j < dim; ++j) {
                weight_grads[j] += 2.0 * error * xrow[j] / n;
            }
        }

        // 更新偏置项（截距），不对其使用正则化
        weights[0] -= learning_rate * bias_grad;
        for (int j = 0; j < dim; ++j) {
            // 正则化梯度：L2 正则化项会让权重不至于变得太大
            double reg_grad = 2.0 * lambda * weights[j + 1] / n;
            weights[j + 1] -= learning_rate * (weight_grads[j] + reg_grad);
        }

        if (epoch % 200 == 0 || epoch == 1 || epoch == epochs) {
            printf("Epoch %d/%d: MSE=%.2f\n", epoch, epochs, mse / n);
        }
        free(weight_grads);
    }
}

// 使用模型权重对一条样本进行预测，返回预测的年薪值。
static double predict_one(const double *xrow, int dim, const double *weights) {
    double pred = weights[0];
    for (int j = 0; j < dim; ++j) {
        pred += weights[j + 1] * xrow[j];
    }
    return pred;
}

// 评估模型在选定样本上的表现，计算 MAE、RMSE 和 R²。
static void evaluate_model(const double *X, const double *y, const int *indices, int n, int dim, const double *weights,
                           double *mae, double *rmse, double *r2) {
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double sum_target = 0.0;
    for (int i = 0; i < n; ++i) sum_target += y[indices[i]];
    double mean_target = sum_target / n;
    double total_variance = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = y[indices[i]] - mean_target;
        total_variance += diff * diff;
    }
    double residual_sum = 0.0;

    for (int i = 0; i < n; ++i) {
        const double *xrow = X + ((ptrdiff_t)indices[i] * dim);
        double prediction = predict_one(xrow, dim, weights);
        double error = prediction - y[indices[i]];
        sum_abs += fabs(error);
        sum_sq += error * error;
        residual_sum += error * error;
    }
    *mae = sum_abs / n;
    *rmse = sqrt(sum_sq / n);
    // 计算 R²：1 - SSE/SST。R² 越接近 1，模型解释能力越强。
    *r2 = (total_variance > 1e-12) ? (1.0 - residual_sum / total_variance) : 0.0;
}

// 打印类别名称和对应的 one-hot 索引，方便用户按照编号选择类别。
static void print_categories(const char *name, Category *cats, int count) {
    printf("%s categories (%d):\n", name, count);
    for (int i = 0; i < count; ++i) {
        if (cats[i].keep) {
            printf("  [%2d] %s (%d)\n", cats[i].mapped_index, cats[i].value, cats[i].count);
        }
    }
}

// 从用户输入中读取一个整数索引，并确保它在合法范围内。
static int ask_for_index(const char *prompt, int max_index) {
    int index = -1;
    while (index < 0 || index > max_index) {
        printf("%s (0-%d): ", prompt, max_index);
        if (scanf("%d", &index) != 1) {
            while (getchar() != '\n');
            printf("输入无效，请输入整数。\n");
            index = -1;
            continue;
        }
        if (index < 0 || index > max_index) {
            printf("索引超出范围，请重新输入。\n");
        }
    }
    return index;
}

// 允许用户手动输入样本特征，并使用训练好的模型进行预测。
static void predict_custom_example(const double *weights, int dim,
                                   Category *exp_cats, int exp_count,
                                   Category *edu_cats, int edu_count,
                                   Category *job_cats, int job_count,
                                   Category *company_cats, int company_count,
                                   Category *country_cats, int country_count,
                                   Category *industry_cats, int industry_count,
                                   double year_mean, double year_std) {
    printf("\n===== 人工输入预测示例 =====\n");
    printf("请根据提示输入特征值来预测 annual_salary_usd。\n");

    double years;
    printf("years_of_experience: ");
    if (scanf("%lf", &years) != 1) {
        printf("输入错误，跳过自定义预测。\n");
        while (getchar() != '\n');
        return;
    }

    printf("\n");
    print_categories("experience_level", exp_cats, exp_count);
    int exp_choice = ask_for_index("选择 experience_level", exp_count - 1);

    printf("\n");
    print_categories("education_required", edu_cats, edu_count);
    int edu_choice = ask_for_index("选择 education_required", edu_count - 1);

    printf("\n");
    print_categories("job_category", job_cats, job_count);
    int job_choice = ask_for_index("选择 job_category", job_count - 1);

    printf("\n");
    print_categories("company_size", company_cats, company_count);
    int company_choice = ask_for_index("选择 company_size", company_count - 1);

    printf("\n");
    print_categories("country", country_cats, country_count);
    int country_choice = ask_for_index("选择 country", country_count - 1);

    printf("\n");
    print_categories("industry", industry_cats, industry_count);
    int industry_choice = ask_for_index("选择 industry", industry_count - 1);

    double *xrow = calloc((size_t)dim, sizeof(double));
    if (!xrow) return;
    xrow[0] = (years - year_mean) / year_std;
    int offset = 1;
    if (exp_choice >= 0) xrow[offset + exp_choice] = 1.0;
    offset += exp_count;
    if (edu_choice >= 0) xrow[offset + edu_choice] = 1.0;
    offset += edu_count;
    if (job_choice >= 0) xrow[offset + job_choice] = 1.0;
    offset += job_count;
    if (company_choice >= 0) xrow[offset + company_choice] = 1.0;
    offset += company_count;
    if (country_choice >= 0) xrow[offset + country_choice] = 1.0;
    offset += country_count;
    if (industry_choice >= 0) xrow[offset + industry_choice] = 1.0;

    double prediction = predict_one(xrow, dim, weights);
    printf("\n预测年薪 annual_salary_usd = %.2f USD\n", prediction);
    free(xrow);
}

int main(void) {
    // 程序入口：读取 CSV、处理数据、训练模型、评估结果，并支持自定义输入预测。

    // Windows 控制台设置为 UTF-8 编码，避免中文乱码
#ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif

    const char *filename = "ai_jobs_market_2025_2026.csv";
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "无法打开文件：%s\n", filename);
        return EXIT_FAILURE;
    }

    char line[MAX_LINE];
    if (!fgets(line, sizeof(line), file)) {
        fprintf(stderr, "CSV 文件为空或读取失败。\n");
        fclose(file);
        return EXIT_FAILURE;
    }

    int header_fields = 0;
    char *header[MAX_FIELDS];
    header_fields = parse_csv_line(line, header, MAX_FIELDS);
    int idx_years = -1, idx_exp_level = -1, idx_edu = -1, idx_job_cat = -1;
    int idx_company = -1, idx_country = -1, idx_industry = -1, idx_salary = -1;

    for (int i = 0; i < header_fields; ++i) {
        if (strcmp(header[i], "years_of_experience") == 0) idx_years = i;
        if (strcmp(header[i], "experience_level") == 0) idx_exp_level = i;
        if (strcmp(header[i], "education_required") == 0) idx_edu = i;
        if (strcmp(header[i], "job_category") == 0) idx_job_cat = i;
        if (strcmp(header[i], "company_size") == 0) idx_company = i;
        if (strcmp(header[i], "country") == 0) idx_country = i;
        if (strcmp(header[i], "industry") == 0) idx_industry = i;
        if (strcmp(header[i], "annual_salary_usd") == 0) idx_salary = i;
    }

    for (int i = 0; i < header_fields; ++i) free(header[i]);

    if (idx_years < 0 || idx_exp_level < 0 || idx_edu < 0 || idx_job_cat < 0 || idx_company < 0 || idx_country < 0 || idx_industry < 0 || idx_salary < 0) {
        fprintf(stderr, "CSV 缺少必要字段。\n");
        fclose(file);
        return EXIT_FAILURE;
    }

    Row rows[MAX_ROWS];
    int raw_count = 0;
    int dropped_empty = 0;
    int dropped_invalid = 0;
    int dropped_duplicate = 0;

    // 逐行读取 CSV 数据，保留需要的字段并过滤无效行
    while (fgets(line, sizeof(line), file)) {
        char *fields[MAX_FIELDS] = {0};
        int field_count = parse_csv_line(line, fields, MAX_FIELDS);
        if (field_count <= idx_salary) {
            for (int j = 0; j < field_count; ++j) free(fields[j]);
            continue;
        }
        if (is_missing_value(fields[idx_years]) || is_missing_value(fields[idx_exp_level]) || is_missing_value(fields[idx_edu]) || is_missing_value(fields[idx_job_cat]) || is_missing_value(fields[idx_company]) || is_missing_value(fields[idx_country]) || is_missing_value(fields[idx_industry]) || is_missing_value(fields[idx_salary])) {
            dropped_empty++;
            for (int j = 0; j < field_count; ++j) free(fields[j]);
            continue;
        }

        double years, salary;
        if (!safe_parse_double(fields[idx_years], &years) || !safe_parse_double(fields[idx_salary], &salary)) {
            dropped_invalid++;
            for (int j = 0; j < field_count; ++j) free(fields[j]);
            continue;
        }
        if (years < 0.0 || salary < 0.0) {
            dropped_invalid++;
            for (int j = 0; j < field_count; ++j) free(fields[j]);
            continue;
        }

        Row candidate;
        candidate.years_of_experience = years;
        candidate.annual_salary_usd = salary;
        candidate.experience_level = strdup_c(fields[idx_exp_level]);
        candidate.education_required = strdup_c(fields[idx_edu]);
        candidate.job_category = strdup_c(fields[idx_job_cat]);
        candidate.company_size = strdup_c(fields[idx_company]);
        candidate.country = strdup_c(fields[idx_country]);
        candidate.industry = strdup_c(fields[idx_industry]);

        int is_dup = 0;
        for (int j = 0; j < raw_count; ++j) {
            if (rows_are_duplicate(&candidate, &rows[j])) {
                is_dup = 1;
                break;
            }
        }
        if (is_dup) {
            dropped_duplicate++;
            free(candidate.experience_level);
            free(candidate.education_required);
            free(candidate.job_category);
            free(candidate.company_size);
            free(candidate.country);
            free(candidate.industry);
        } else {
            if (raw_count >= MAX_ROWS) break;
            rows[raw_count++] = candidate;
        }

        for (int j = 0; j < field_count; ++j) free(fields[j]);
    }
    fclose(file);

    if (raw_count == 0) {
        fprintf(stderr, "没有有效样本可供训练。\n");
        return EXIT_FAILURE;
    }

    printf("读取到 %d 条有效样本。\n", raw_count);
    if (dropped_empty > 0) printf("丢弃 %d 条缺失值样本。\n", dropped_empty);
    if (dropped_invalid > 0) printf("丢弃 %d 条无效数值样本。\n", dropped_invalid);
    if (dropped_duplicate > 0) printf("丢弃 %d 条重复样本。\n", dropped_duplicate);

    // 统计每个分类特征中的类别出现次数，并为低频类别做归并准备
    Category exp_cats[MAX_CAT_VALUES] = {0};
    Category edu_cats[MAX_CAT_VALUES] = {0};
    Category job_cats[MAX_CAT_VALUES] = {0};
    Category company_cats[MAX_CAT_VALUES] = {0};
    Category country_cats[MAX_CAT_VALUES] = {0};
    Category industry_cats[MAX_CAT_VALUES] = {0};
    int exp_count = 0, edu_count = 0, job_count = 0, company_count = 0, country_count = 0, industry_count = 0;

    for (int i = 0; i < raw_count; ++i) {
        category_add(exp_cats, &exp_count, rows[i].experience_level);
        category_add(edu_cats, &edu_count, rows[i].education_required);
        category_add(job_cats, &job_count, rows[i].job_category);
        category_add(company_cats, &company_count, rows[i].company_size);
        category_add(country_cats, &country_count, rows[i].country);
        category_add(industry_cats, &industry_count, rows[i].industry);
    }

    // 对经验等级和学历按从低到高排序，使 one-hot 索引符合语义顺序
    if (exp_count > 1) qsort(exp_cats, (size_t)exp_count, sizeof(Category), cmp_exp);
    if (edu_count > 1) qsort(edu_cats, (size_t)edu_count, sizeof(Category), cmp_edu);

    build_category_mapping(exp_cats, &exp_count, RARE_CATEGORY_THRESHOLD);
    build_category_mapping(edu_cats, &edu_count, RARE_CATEGORY_THRESHOLD);
    build_category_mapping(job_cats, &job_count, RARE_CATEGORY_THRESHOLD);
    build_category_mapping(company_cats, &company_count, RARE_CATEGORY_THRESHOLD);
    build_category_mapping(country_cats, &country_count, RARE_CATEGORY_THRESHOLD);
    build_category_mapping(industry_cats, &industry_count, RARE_CATEGORY_THRESHOLD);

    printf("\n类别分布和 one-hot 特征映射：\n");
    print_categories("experience_level", exp_cats, exp_count);
    print_categories("education_required", edu_cats, edu_count);
    print_categories("job_category", job_cats, job_count);
    print_categories("company_size", company_cats, company_count);
    print_categories("country", country_cats, country_count);
    print_categories("industry", industry_cats, industry_count);

    int exp_idx[MAX_ROWS], edu_idx[MAX_ROWS], job_idx[MAX_ROWS], company_idx[MAX_ROWS], country_idx[MAX_ROWS], industry_idx[MAX_ROWS];
    double years[MAX_ROWS];
    double salary[MAX_ROWS];
    for (int i = 0; i < raw_count; ++i) {
        exp_idx[i] = map_category_value(exp_cats, exp_count, rows[i].experience_level);
        edu_idx[i] = map_category_value(edu_cats, edu_count, rows[i].education_required);
        job_idx[i] = map_category_value(job_cats, job_count, rows[i].job_category);
        company_idx[i] = map_category_value(company_cats, company_count, rows[i].company_size);
        country_idx[i] = map_category_value(country_cats, country_count, rows[i].country);
        industry_idx[i] = map_category_value(industry_cats, industry_count, rows[i].industry);
        years[i] = rows[i].years_of_experience;
        salary[i] = rows[i].annual_salary_usd;
    }

    int total_features = 1 + exp_count + edu_count + job_count + company_count + country_count + industry_count;
    if (total_features > MAX_FEATURES) {
        fprintf(stderr, "特征维度过大：%d > %d。请减少类别数或增加 MAX_FEATURES。\n", total_features, MAX_FEATURES);
        free_rows(rows, raw_count);
        free_categories(exp_cats, exp_count);
        free_categories(edu_cats, edu_count);
        free_categories(job_cats, job_count);
        free_categories(company_cats, company_count);
        free_categories(country_cats, country_count);
        free_categories(industry_cats, industry_count);
        return EXIT_FAILURE;
    }

    double *X = calloc((size_t)raw_count * total_features, sizeof(double));
    if (!X) {
        fprintf(stderr, "内存分配失败。\n");
        return EXIT_FAILURE;
    }

    // 随机打乱样本顺序，并分成训练集和测试集
    int indices[MAX_ROWS];
    for (int i = 0; i < raw_count; ++i) indices[i] = i;
    srand(123456);
    shuffle_indices(indices, raw_count);
    int train_size = (raw_count * 8 + 5) / 10;
    int test_size = raw_count - train_size;
    int train_idx[MAX_ROWS], test_idx[MAX_ROWS];
    for (int i = 0; i < train_size; ++i) train_idx[i] = indices[i];
    for (int i = 0; i < test_size; ++i) test_idx[i] = indices[train_size + i];

    double year_mean = 0.0, year_std = 1.0;
    // 先在训练集上计算标准化参数，用相同参数处理全部数据
    compute_normalization(years, train_idx, train_size, &year_mean, &year_std);
    build_feature_matrix(X, years, raw_count, year_mean, year_std,
                         exp_idx, exp_count,
                         edu_idx, edu_count,
                         job_idx, job_count,
                         company_idx, company_count,
                         country_idx, country_count,
                         industry_idx, industry_count,
                         total_features);

    double *X_train = calloc((size_t)train_size * total_features, sizeof(double));
    double *y_train = calloc((size_t)train_size, sizeof(double));
    if (!X_train || !y_train) {
        fprintf(stderr, "训练集内存分配失败。\n");
        free(X);
        return EXIT_FAILURE;
    }
    for (int i = 0; i < train_size; ++i) {
        memcpy(X_train + ((ptrdiff_t)i * total_features), X + ((ptrdiff_t)train_idx[i] * total_features), sizeof(double) * total_features);
        y_train[i] = salary[train_idx[i]];
    }

    double *weights = calloc((size_t)total_features + 1, sizeof(double));
    if (!weights) {
        fprintf(stderr, "权重内存分配失败。\n");
        free(X);
        free(X_train);
        free(y_train);
        return EXIT_FAILURE;
    }

    // 训练线性回归模型，并打印训练过程中的损失。
    printf("\n开始训练 Ridge 线性回归模型...\n");
    ridge_train(X_train, y_train, train_size, total_features, weights, 2000, 0.01, 0.1);

    // 评估模型在训练集和测试集上的效果
    double train_mae, train_rmse, train_r2;
    evaluate_model(X, salary, train_idx, train_size, total_features, weights, &train_mae, &train_rmse, &train_r2);
    double test_mae, test_rmse, test_r2;
    evaluate_model(X, salary, test_idx, test_size, total_features, weights, &test_mae, &test_rmse, &test_r2);

    printf("\n训练集评估: MAE=%.2f, RMSE=%.2f, R²=%.4f\n", train_mae, train_rmse, train_r2);
    printf("测试集评估: MAE=%.2f, RMSE=%.2f, R²=%.4f\n", test_mae, test_rmse, test_r2);

    printf("\n模型权重示例（前 10 个权重，包括偏置项）：\n");
    for (int i = 0; i < total_features + 1 && i < 10; ++i) {
        printf("w[%d] = %.4f\n", i, weights[i]);
    }

    // 如果希望进行人工输入预测，则进入交互预测环节
    printf("\n如果你想输入自定义特征进行预测，请在提示中输入。\n");
    predict_custom_example(weights, total_features,
                           exp_cats, exp_count,
                           edu_cats, edu_count,
                           job_cats, job_count,
                           company_cats, company_count,
                           country_cats, country_count,
                           industry_cats, industry_count,
                           year_mean, year_std);

    free(X);
    free(X_train);
    free(y_train);
    free(weights);
    free_rows(rows, raw_count);
    free_categories(exp_cats, exp_count);
    free_categories(edu_cats, edu_count);
    free_categories(job_cats, job_count);
    free_categories(company_cats, company_count);
    free_categories(country_cats, country_count);
    free_categories(industry_cats, industry_count);

    return EXIT_SUCCESS;
}
