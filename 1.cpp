#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>

// Windows 下设置控制台为 UTF-8，避免中文乱码
#ifdef _WIN32
#include <windows.h>
#endif

// rapidcsv — header-only CSV 解析库（参考 usingopencv.cpp 的用法）
#include "rapidcsv.h"

using namespace std;

// 低频类别阈值，小于该数量的类别会被归并为 "Other"
#define RARE_CATEGORY_THRESHOLD 2

// ── 数据结构 ────────────────────────────────────────────

typedef struct {
    string value;        // 类别名称
    int count;           // 出现次数
    int keep;            // 是否保留
    int mapped_index;    // one-hot 索引
} Category;

typedef struct {
    double years_of_experience;
    double annual_salary_usd;
    string experience_level;
    string education_required;
    string job_category;
    string company_size;
    string country;
    string industry;
} Row;

typedef struct {
    double mae;
    double rmse;
    double r2;
    double adjusted_r2;
    double mape;
    double residual_mean;
    double residual_std;
} EvalMetrics;

// ── 类别管理 ────────────────────────────────────────────

static int category_lookup(const vector<Category> &cats, const string &value) {
    for (size_t i = 0; i < cats.size(); ++i)
        if (cats[i].value == value) return (int)i;
    return -1;
}

static int category_add(vector<Category> &cats, const string &value) {
    int idx = category_lookup(cats, value);
    if (idx >= 0) { cats[idx].count++; return idx; }
    Category c;
    c.value = value; c.count = 1; c.keep = 0; c.mapped_index = -1;
    cats.push_back(c);
    return (int)cats.size() - 1;
}

static int find_other_index(const vector<Category> &cats) {
    for (size_t i = 0; i < cats.size(); ++i)
        if (cats[i].value == "Other") return (int)i;
    return -1;
}

// ── 类别排序：按语义从低到高 ────────────────────────────

static int exp_level_order(const string &v) {
    if (v.find("Entry")  != string::npos) return 0;
    if (v.find("Mid")    != string::npos) return 1;
    if (v.find("Senior") != string::npos) return 2;
    if (v.find("Lead")   != string::npos) return 3;
    return 99;
}
static int education_order(const string &v) {
    if (v.find("Bootcamp")    != string::npos) return 0;
    if (v.find("Self-taught") != string::npos) return 0;
    if (v.find("Associate")   != string::npos) return 1;
    if (v.find("Bachelor")    != string::npos) return 2;
    if (v.find("Master")      != string::npos) return 3;
    if (v.find("PhD")         != string::npos) return 4;
    return 99;
}
static int company_size_order(const string &v) {
    if (v.find("(1-50)")     != string::npos) return 0;
    if (v.find("(51-500)")   != string::npos) return 1;
    if (v.find("(501-5000)") != string::npos) return 2;
    if (v.find("(5000+)")    != string::npos) return 3;
    if (v.find("FAANG")      != string::npos) return 4;
    return 99;
}

static void sort_cats(vector<Category> &cats, int (*order)(const string&)) {
    sort(cats.begin(), cats.end(),
        [order](const Category &a, const Category &b) {
            int oa = order(a.value), ob = order(b.value);
            return oa != ob ? oa < ob : a.value < b.value;
        });
}

// ── 类别编码 ────────────────────────────────────────────

static void build_category_mapping(vector<Category> &cats, int min_count) {
    int other_idx = find_other_index(cats);
    bool rare = false;
    for (auto &c : cats) {
        if (c.count < min_count && c.value != "Other") { rare = true; c.keep = 0; }
        else c.keep = 1;
    }
    if (rare && other_idx < 0) {
        Category oc; oc.value = "Other"; oc.count = 0; oc.keep = 1; oc.mapped_index = -1;
        cats.push_back(oc);
    }
    int m = 0;
    for (auto &c : cats) {
        c.mapped_index = c.keep ? m++ : -1;
    }
}

static int map_category_value(const vector<Category> &cats, const string &value) {
    int idx = category_lookup(cats, value);
    if (idx >= 0 && cats[idx].keep) return cats[idx].mapped_index;
    int oi = find_other_index(cats);
    if (oi >= 0) return cats[oi].mapped_index;
    return cats.empty() ? -1 : 0;
}

// ── 数据工具 ────────────────────────────────────────────

static bool rows_duplicate(const Row &a, const Row &b) {
    return a.years_of_experience == b.years_of_experience
        && a.annual_salary_usd   == b.annual_salary_usd
        && a.experience_level    == b.experience_level
        && a.education_required  == b.education_required
        && a.job_category        == b.job_category
        && a.company_size        == b.company_size
        && a.country             == b.country
        && a.industry            == b.industry;
}

static void shuffle(vector<int> &idx) {
    int n = (int)idx.size();
    for (int i = n - 1; i > 0; --i) swap(idx[i], idx[rand() % (i + 1)]);
}

static void compute_norm(const vector<double> &vals, const vector<int> &idx,
                         double &mean, double &stddev) {
    int n = (int)idx.size();
    double s = 0; for (int i = 0; i < n; ++i) s += vals[idx[i]];
    mean = s / n;
    double v = 0; for (int i = 0; i < n; ++i) { double d = vals[idx[i]] - mean; v += d * d; }
    v /= (n > 1 ? (n - 1) : 1);
    stddev = sqrt(v); if (stddev < 1e-8) stddev = 1.0;
}

// ── 特征矩阵 ────────────────────────────────────────────

static void build_X(vector<double> &X, const vector<double> &years, int n,
                    double mean, double std,
                    const vector<int> &exp,  int exp_c,
                    const vector<int> &edu,  int edu_c,
                    const vector<int> &job,  int job_c,
                    const vector<int> &comp, int comp_c,
                    const vector<int> &ctry, int ctry_c,
                    const vector<int> &ind,  int ind_c,
                    int dim) {
    (void)ind_c;
    X.assign((size_t)n * dim, 0.0);
    for (int i = 0; i < n; ++i) {
        double *row = X.data() + (ptrdiff_t)i * dim;
        row[0] = (years[i] - mean) / std;
        int off = 1;
        if (exp[i]  >= 0) row[off + exp[i]]  = 1.0;
        off += exp_c;
        if (edu[i]  >= 0) row[off + edu[i]]  = 1.0;
        off += edu_c;
        if (job[i]  >= 0) row[off + job[i]]  = 1.0;
        off += job_c;
        if (comp[i] >= 0) row[off + comp[i]] = 1.0;
        off += comp_c;
        if (ctry[i] >= 0) row[off + ctry[i]] = 1.0;
        off += ctry_c;
        if (ind[i]  >= 0) row[off + ind[i]]  = 1.0;
    }
}

// ── 模型核心 ────────────────────────────────────────────

static double predict_one(const double *xrow, int dim, const vector<double> &w) {
    double p = w[0];
    for (int j = 0; j < dim; ++j) p += w[j + 1] * xrow[j];
    return p;
}

static void ridge_train(const vector<double> &X, const vector<double> &y,
                        int n, int dim, vector<double> &w,
                        int epochs, double lr, double lambda,
                        const vector<double> &Xt = {}, const vector<double> &yt = {},
                        int nt = 0) {
    w.assign((size_t)dim + 1, 0.0);
    for (int ep = 1; ep <= epochs; ++ep) {
        double bg = 0; vector<double> wg(dim, 0.0); double mse = 0;
        for (int i = 0; i < n; ++i) {
            const double *xr = X.data() + (ptrdiff_t)i * dim;
            double pred = w[0];
            for (int j = 0; j < dim; ++j) pred += w[j + 1] * xr[j];
            double err = pred - y[i];
            mse += err * err;
            bg += 2.0 * err / n;
            for (int j = 0; j < dim; ++j) wg[j] += 2.0 * err * xr[j] / n;
        }
        w[0] -= lr * bg;
        for (int j = 0; j < dim; ++j) {
            double rg = 2.0 * lambda * w[j + 1] / n;
            w[j + 1] -= lr * (wg[j] + rg);
        }
        if (ep % 200 == 0 || ep == 1 || ep == epochs) {
            printf("Epoch %4d/%-4d: train MSE=%12.2f", ep, epochs, mse / n);
            if (nt > 0) {
                double tm = 0;
                for (int i = 0; i < nt; ++i) {
                    const double *xr = Xt.data() + (ptrdiff_t)i * dim;
                    double err = predict_one(xr, dim, w) - yt[i];
                    tm += err * err;
                }
                printf(", test MSE=%12.2f", tm / nt);
            }
            printf("\n");
        }
    }
}

// ── 模型评估 ────────────────────────────────────────────

static EvalMetrics evaluate(const vector<double> &X, const vector<double> &y,
                            const vector<int> &idx, int dim,
                            const vector<double> &w) {
    EvalMetrics m = {};
    int n = (int)idx.size(); if (n == 0) return m;

    double st = 0; for (int i = 0; i < n; ++i) st += y[idx[i]];
    double mt = st / n;
    double tv = 0; for (int i = 0; i < n; ++i) { double d = y[idx[i]] - mt; tv += d * d; }

    vector<double> errs(n);
    double sa = 0, ss = 0, sr = 0, sap = 0;
    for (int i = 0; i < n; ++i) {
        const double *xr = X.data() + (ptrdiff_t)idx[i] * dim;
        double err = predict_one(xr, dim, w) - y[idx[i]];
        errs[i] = err; double ae = fabs(err);
        sa += ae; ss += err * err; sr += err;
        if (y[idx[i]] > 1.0) sap += ae / y[idx[i]];
    }
    m.mae = sa / n; m.rmse = sqrt(ss / n);
    m.r2 = (tv > 1e-12) ? (1.0 - ss / tv) : 0.0;
    int p = dim;
    m.adjusted_r2 = (n > p + 1) ? 1.0 - (1.0 - m.r2) * (double)(n - 1) / (double)(n - p - 1) : m.r2;
    m.mape = (sap / n) * 100.0;
    m.residual_mean = sr / n;
    double rv = 0; for (int i = 0; i < n; ++i) { double d = errs[i] - m.residual_mean; rv += d * d; }
    m.residual_std = sqrt(rv / n);
    return m;
}

// ── 特征重要性 ──────────────────────────────────────────

static void feat_name(int idx,
                      int ec, const vector<Category> &ecats,
                      int dc, const vector<Category> &dcats,
                      int jc, const vector<Category> &jcats,
                      int cc, const vector<Category> &ccats,
                      int tc, const vector<Category> &tcats,
                      int ic, const vector<Category> &icats) {
    if (idx == 0) { printf("(bias/截距)"); return; } idx--;
    if (idx == 0) { printf("years_of_experience"); return; } idx--;
    if (idx < ec) { printf("experience_level=%s",  ecats[idx].value.c_str()); return; } idx -= ec;
    if (idx < dc) { printf("education_required=%s", dcats[idx].value.c_str()); return; } idx -= dc;
    if (idx < jc) { printf("job_category=%s",       jcats[idx].value.c_str()); return; } idx -= jc;
    if (idx < cc) { printf("company_size=%s",       ccats[idx].value.c_str()); return; } idx -= cc;
    if (idx < tc) { printf("country=%s",            tcats[idx].value.c_str()); return; } idx -= tc;
    if (idx < ic) { printf("industry=%s",           icats[idx].value.c_str()); return; }
    printf("feature[%d]", idx);
}

static void print_importance(const vector<double> &w, int dim,
                             int ec, const vector<Category> &ecats,
                             int dc, const vector<Category> &dcats,
                             int jc, const vector<Category> &jcats,
                             int cc, const vector<Category> &ccats,
                             int tc, const vector<Category> &tcats,
                             int ic, const vector<Category> &icats,
                             int top_n) {
    if (top_n <= 0 || dim <= 0) return;
    struct P { int idx; double aw; };
    vector<P> ps((size_t)dim + 1);
    for (int i = 0; i <= dim; ++i) { ps[i].idx = i; ps[i].aw = fabs(w[i]); }
    for (int i = 0; i <= dim && i < top_n; ++i) {
        int best = i;
        for (int j = i + 1; j <= dim; ++j) if (ps[j].aw > ps[best].aw) best = j;
        if (best != i) swap(ps[i], ps[best]);
    }
    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  特征重要性（权重绝对值 Top %d）\n", top_n);
    printf("══════════════════════════════════════════════════════════\n");
    printf("  %-3s %-42s %12s\n", "排名", "特征名称", "权重");
    printf("──────────────────────────────────────────────────────────\n");
    for (int i = 0; i < top_n && i <= dim; ++i) {
        printf("  %-3d ", i + 1);
        feat_name(ps[i].idx, ec, ecats, dc, dcats, jc, jcats, cc, ccats, tc, tcats, ic, icats);
        printf(" %+12.2f\n", w[ps[i].idx]);
    }
    printf("──────────────────────────────────────────────────────────\n");
}

// ── 交叉验证 ────────────────────────────────────────────

static void cross_validate(const vector<double> &X, const vector<double> &y,
                           const vector<int> &all_idx, int dim,
                           int epochs, double lr, double lambda, int k) {
    int n = (int)all_idx.size();
    if (k < 2 || n < k * 2) {
        printf("\n⚠ 样本数不足，跳过交叉验证（需要至少 %d 条）。\n", k * 2);
        return;
    }
    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  %d-Fold 交叉验证\n", k);
    printf("══════════════════════════════════════════════════════════\n");

    vector<int> fi = all_idx;
    srand(789012);
    for (int i = n - 1; i > 0; --i) { int j = rand() % (i + 1); swap(fi[i], fi[j]); }

    vector<double> cmae(k), crmse(k), cr2(k);
    int fs = n / k;

    for (int fold = 0; fold < k; ++fold) {
        int vs = fold * fs, ve = (fold == k - 1) ? n : vs + fs;
        int vn = ve - vs, sn = n - vn;

        vector<double> Xs((size_t)sn * dim), ys(sn), wcv((size_t)dim + 1);
        int pos = 0;
        for (int f = 0; f < k; ++f) {
            if (f == fold) continue;
            int s = f * fs, e = (f == k - 1) ? n : s + fs;
            for (int j = s; j < e; ++j) {
                memcpy(Xs.data() + (ptrdiff_t)pos * dim,
                       X.data() + (ptrdiff_t)fi[j] * dim, sizeof(double) * dim);
                ys[pos++] = y[fi[j]];
            }
        }
        ridge_train(Xs, ys, sn, dim, wcv, epochs, lr, lambda);
        vector<int> vi(vn); for (int j = 0; j < vn; ++j) vi[j] = fi[vs + j];
        EvalMetrics em = evaluate(X, y, vi, dim, wcv);
        cmae[fold] = em.mae; crmse[fold] = em.rmse; cr2[fold] = em.r2;
        printf("  Fold %d/%d: MAE=%10.2f  RMSE=%10.2f  R²=%7.4f\n",
               fold + 1, k, em.mae, em.rmse, em.r2);
    }

    double mm = 0, rm = 0, r2m = 0;
    for (int i = 0; i < k; ++i) { mm += cmae[i]; rm += crmse[i]; r2m += cr2[i]; }
    mm /= k; rm /= k; r2m /= k;
    double ms = 0, rs = 0, r2s = 0;
    for (int i = 0; i < k; ++i) {
        ms  += (cmae[i] - mm)  * (cmae[i] - mm);
        rs  += (crmse[i] - rm) * (crmse[i] - rm);
        r2s += (cr2[i] - r2m)  * (cr2[i] - r2m);
    }
    printf("  ─────────────────────────────────────────────────────\n");
    printf("  %-8s %16s %14s\n", "指标", "Mean", "Std");
    printf("  %-8s %16.2f %14.2f\n", "MAE",   mm,  sqrt(ms/k));
    printf("  %-8s %16.2f %14.2f\n", "RMSE",  rm,  sqrt(rs/k));
    printf("  %-8s %16.4f %14.4f\n", "R²",    r2m, sqrt(r2s/k));
    printf("══════════════════════════════════════════════════════════\n");
}

// ── 输出 ────────────────────────────────────────────────

static void print_cats(const char *name, const vector<Category> &cats) {
    printf("%s categories (%d):\n", name, (int)cats.size());
    for (const auto &c : cats)
        if (c.keep) printf("  [%2d] %s (%d)\n", c.mapped_index, c.value.c_str(), c.count);
}

static int ask_idx(const char *prompt, int max_idx) {
    int idx = -1;
    while (idx < 0 || idx > max_idx) {
        printf("%s (0-%d): ", prompt, max_idx); fflush(stdout);
        if (scanf("%d", &idx) != 1) { while (getchar() != '\n'); printf("输入无效，请输入整数。\n"); idx = -1; continue; }
        if (idx < 0 || idx > max_idx) printf("索引超出范围，请重新输入。\n");
    }
    return idx;
}

// ── 交互预测 ────────────────────────────────────────────

static void predict_custom(const vector<double> &w, int dim,
                           const vector<Category> &ec, int ecn,
                           const vector<Category> &dc, int dcn,
                           const vector<Category> &jc, int jcn,
                           const vector<Category> &cc, int ccn,
                           const vector<Category> &tc, int tcn,
                           const vector<Category> &ic, int icn,
                           double ym, double ys) {
    printf("\n===== 人工输入预测示例 =====\n");
    printf("请根据提示输入特征值来预测 annual_salary_usd。\n");
    printf("Input the feature values according to the prompts to predict annual_salary_usd.\n");

    double years;
    printf("years_of_experience: "); fflush(stdout);
    if (scanf("%lf", &years) != 1) { printf("输入错误，跳过自定义预测。\n"); while (getchar() != '\n'); return; }

    printf("\n"); print_cats("experience_level", ec);
    int ei = ask_idx("选择 experience_level", ecn - 1);
    printf("\n"); print_cats("education_required", dc);
    int di = ask_idx("选择 education_required", dcn - 1);
    printf("\n"); print_cats("job_category", jc);
    int ji = ask_idx("选择 job_category", jcn - 1);
    printf("\n"); print_cats("company_size", cc);
    int ci = ask_idx("选择 company_size", ccn - 1);
    printf("\n"); print_cats("country", tc);
    int ti = ask_idx("选择 country", tcn - 1);
    printf("\n"); print_cats("industry", ic);
    int ii = ask_idx("选择 industry", icn - 1);

    vector<double> xr(dim, 0.0);
    xr[0] = (years - ym) / ys; int off = 1;
    if (ei >= 0) xr[off + ei] = 1.0;
    off += ecn;
    if (di >= 0) xr[off + di] = 1.0;
    off += dcn;
    if (ji >= 0) xr[off + ji] = 1.0;
    off += jcn;
    if (ci >= 0) xr[off + ci] = 1.0;
    off += ccn;
    if (ti >= 0) xr[off + ti] = 1.0;
    off += tcn;
    if (ii >= 0) xr[off + ii] = 1.0;

    printf("\n预测年薪 annual_salary_usd = %.2f USD\n", predict_one(xr.data(), dim, w));
    fflush(stdout);
}

// ── main ────────────────────────────────────────────────

int main(int argc, char **argv) {
#ifdef _WIN32
    SetConsoleOutputCP(65001); SetConsoleCP(65001);
#endif

    // ── 1. 打开 CSV ──────────────────────────────────────
    const char *csv_name = "ai_jobs_market_2025_2026.csv";
    string csv_path;

    // 多级回退查找（仿照原 1.c 的逻辑）
    vector<string> paths = { csv_name };
#ifdef _WIN32
    char ed[1024]; GetModuleFileNameA(NULL, ed, sizeof(ed));
    char *ls = strrchr(ed, '\\'); if (ls) *(ls + 1) = '\0';
    paths.push_back(string(ed) + csv_name);
    paths.push_back(string(ed) + "..\\" + csv_name);
    paths.push_back(string(ed) + "..\\..\\" + csv_name);
#endif
    for (const auto &p : paths) {
        ifstream test(p); if (test.is_open()) { csv_path = p; break; }
    }
    if (csv_path.empty()) {
        fprintf(stderr, "无法打开文件：%s\n请确认 CSV 文件在项目根目录下。\n", csv_name);
        return EXIT_FAILURE;
    }

    // ── 2. 读取数据（rapidcsv, 仿 usingopencv.cpp 风格）───
    rapidcsv::Document doc(csv_path, rapidcsv::LabelParams(),
                           rapidcsv::SeparatorParams(),
                           rapidcsv::ConverterParams(true));

    vector<Row> rows;
    int dropped_empty = 0, dropped_invalid = 0, dropped_duplicate = 0;

    for (size_t i = 0; i < doc.GetRowCount(); i++) {
        double years  = doc.GetCell<double>("years_of_experience", i);
        double salary = doc.GetCell<double>("annual_salary_usd", i);
        string exp_lv  = doc.GetCell<string>("experience_level", i);
        string edu     = doc.GetCell<string>("education_required", i);
        string job_cat = doc.GetCell<string>("job_category", i);
        string comp_sz = doc.GetCell<string>("company_size", i);
        string country = doc.GetCell<string>("country", i);
        string industry= doc.GetCell<string>("industry", i);

        // 过滤缺失（空字符串或空值）
        if (exp_lv.empty() || edu.empty() || job_cat.empty() ||
            comp_sz.empty() || country.empty() || industry.empty()) {
            dropped_empty++; continue;
        }
        // 过滤无效数值
        if (years < 0.0 || salary < 0.0) { dropped_invalid++; continue; }

        Row r;
        r.years_of_experience = years; r.annual_salary_usd = salary;
        r.experience_level = exp_lv;   r.education_required = edu;
        r.job_category = job_cat;      r.company_size = comp_sz;
        r.country = country;           r.industry = industry;

        bool dup = false;
        for (const auto &ex : rows) if (rows_duplicate(r, ex)) { dup = true; break; }
        if (dup) { dropped_duplicate++; continue; }
        rows.push_back(r);
    }

    int raw_count = (int)rows.size();
    if (raw_count == 0) { fprintf(stderr, "没有有效样本可供训练。\n"); return EXIT_FAILURE; }
    printf("读取到 %d 条有效样本。\n", raw_count);
    if (dropped_empty > 0)    printf("丢弃 %d 条缺失值样本。\n", dropped_empty);
    if (dropped_invalid > 0)  printf("丢弃 %d 条无效数值样本。\n", dropped_invalid);
    if (dropped_duplicate > 0) printf("丢弃 %d 条重复样本。\n", dropped_duplicate);

    // ── 3. 类别统计 ──────────────────────────────────────
    vector<Category> exp_cats, edu_cats, job_cats, comp_cats, ctry_cats, ind_cats;
    for (const auto &r : rows) {
        category_add(exp_cats,  r.experience_level);
        category_add(edu_cats,  r.education_required);
        category_add(job_cats,  r.job_category);
        category_add(comp_cats, r.company_size);
        category_add(ctry_cats, r.country);
        category_add(ind_cats,  r.industry);
    }
    sort_cats(exp_cats,  exp_level_order);
    sort_cats(edu_cats,  education_order);
    sort_cats(comp_cats, company_size_order);

    build_category_mapping(exp_cats,  RARE_CATEGORY_THRESHOLD);
    build_category_mapping(edu_cats,  RARE_CATEGORY_THRESHOLD);
    build_category_mapping(job_cats,  RARE_CATEGORY_THRESHOLD);
    build_category_mapping(comp_cats, RARE_CATEGORY_THRESHOLD);
    build_category_mapping(ctry_cats, RARE_CATEGORY_THRESHOLD);
    build_category_mapping(ind_cats,  RARE_CATEGORY_THRESHOLD);

    int ecn = (int)exp_cats.size(),  dcn = (int)edu_cats.size(),  jcn = (int)job_cats.size();
    int ccn = (int)comp_cats.size(), tcn = (int)ctry_cats.size(), icn = (int)ind_cats.size();

    printf("\n类别分布和 one-hot 特征映射：\n");
    print_cats("experience_level", exp_cats);
    print_cats("education_required", edu_cats);
    print_cats("job_category", job_cats);
    print_cats("company_size", comp_cats);
    print_cats("country", ctry_cats);
    print_cats("industry", ind_cats);

    // ── 4. 构建特征矩阵 ──────────────────────────────────
    vector<int> ei(raw_count), di(raw_count), ji(raw_count), ci(raw_count), ti(raw_count), ii(raw_count);
    vector<double> yrs(raw_count), sal(raw_count);
    for (int i = 0; i < raw_count; ++i) {
        ei[i] = map_category_value(exp_cats,  rows[i].experience_level);
        di[i] = map_category_value(edu_cats,  rows[i].education_required);
        ji[i] = map_category_value(job_cats,  rows[i].job_category);
        ci[i] = map_category_value(comp_cats, rows[i].company_size);
        ti[i] = map_category_value(ctry_cats, rows[i].country);
        ii[i] = map_category_value(ind_cats,  rows[i].industry);
        yrs[i] = rows[i].years_of_experience;
        sal[i] = rows[i].annual_salary_usd;
    }
    int dim = 1 + ecn + dcn + jcn + ccn + tcn + icn;

    // ── 5. 训练/测试划分 ─────────────────────────────────
    vector<int> indices(raw_count);
    for (int i = 0; i < raw_count; ++i) indices[i] = i;
    srand(123456); shuffle(indices);
    int train_n = (raw_count * 8 + 5) / 10, test_n = raw_count - train_n;
    vector<int> tr_idx(indices.begin(), indices.begin() + train_n);
    vector<int> te_idx(indices.begin() + train_n, indices.end());

    double ym = 0, ys = 1.0; compute_norm(yrs, tr_idx, ym, ys);

    vector<double> X;
    build_X(X, yrs, raw_count, ym, ys, ei, ecn, di, dcn, ji, jcn, ci, ccn, ti, tcn, ii, icn, dim);

    vector<double> Xt((size_t)train_n * dim), yt(train_n);
    for (int i = 0; i < train_n; ++i) {
        memcpy(Xt.data() + (ptrdiff_t)i * dim, X.data() + (ptrdiff_t)tr_idx[i] * dim, sizeof(double)*dim);
        yt[i] = sal[tr_idx[i]];
    }
    vector<double> Xv((size_t)test_n * dim), yv(test_n);
    for (int i = 0; i < test_n; ++i) {
        memcpy(Xv.data() + (ptrdiff_t)i * dim, X.data() + (ptrdiff_t)te_idx[i] * dim, sizeof(double)*dim);
        yv[i] = sal[te_idx[i]];
    }

    // ── 6. 训练 ──────────────────────────────────────────
    vector<double> weights;
    printf("\n开始训练 Ridge 线性回归模型...\n");
    ridge_train(Xt, yt, train_n, dim, weights, 2000, 0.01, 0.1, Xv, yv, test_n);

    // ── 7. 评估 ──────────────────────────────────────────
    EvalMetrics tr_em = evaluate(X, sal, tr_idx, dim, weights);
    EvalMetrics te_em = evaluate(X, sal, te_idx, dim, weights);

    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  模型评估（单次 8:2 随机划分）\n");
    printf("══════════════════════════════════════════════════════════\n");
    printf("  %-22s %14s %14s\n", "指标", "训练集", "测试集");
    printf("  ──────────────────────────────────────────────────────\n");
    printf("  %-22s %14.2f %14.2f\n", "MAE (美元)",    tr_em.mae, te_em.mae);
    printf("  %-22s %14.2f %14.2f\n", "RMSE (美元)",   tr_em.rmse, te_em.rmse);
    printf("  %-22s %14.4f %14.4f\n", "R²",           tr_em.r2, te_em.r2);
    printf("  %-22s %14.4f %14.4f\n", "Adjusted R²",  tr_em.adjusted_r2, te_em.adjusted_r2);
    printf("  %-22s %13.2f%% %13.2f%%\n", "MAPE",   tr_em.mape, te_em.mape);
    printf("  %-22s %14.2f %14.2f\n", "残差均值",     tr_em.residual_mean, te_em.residual_mean);
    printf("  %-22s %14.2f %14.2f\n", "残差标准差",   tr_em.residual_std, te_em.residual_std);
    printf("══════════════════════════════════════════════════════════\n");

    // ── 8. 特征重要性 ────────────────────────────────────
    print_importance(weights, dim, ecn, exp_cats, dcn, edu_cats, jcn, job_cats,
                     ccn, comp_cats, tcn, ctry_cats, icn, ind_cats, 15);

    // ── 9. 交叉验证 ──────────────────────────────────────
    cross_validate(X, sal, tr_idx, dim, 2000, 0.01, 0.1, 5);

    // ── 10. 预测 ─────────────────────────────────────────
    if (argc == 8) {
        double years = atof(argv[1]);
        int echi = atoi(argv[2]), edi = atoi(argv[3]), joi = atoi(argv[4]);
        int coi = atoi(argv[5]), cti = atoi(argv[6]), ini = atoi(argv[7]);

        vector<double> xr(dim, 0.0);
        xr[0] = (years - ym) / ys; int off = 1;
        xr[off + echi] = 1.0;
        off += ecn;
        xr[off + edi]  = 1.0;
        off += dcn;
        xr[off + joi]  = 1.0;
        off += jcn;
        xr[off + coi]  = 1.0;
        off += ccn;
        xr[off + cti]  = 1.0;
        off += tcn;
        xr[off + ini]  = 1.0;

        printf("\n预测年薪 annual_salary_usd = %.2f USD\n", predict_one(xr.data(), dim, weights));
        fflush(stdout);
    } else if (argc > 1) {
        printf("用法: %s  years  exp_level  edu  job_category  company_size  country  industry\n", argv[0]);
        printf("示例: %s  7  0  2  0  2  0  1\n", argv[0]);
    } else {
        printf("\n输入自定义特征进行预测...\n");
        predict_custom(weights, dim, exp_cats, ecn, edu_cats, dcn, job_cats, jcn,
                       comp_cats, ccn, ctry_cats, tcn, ind_cats, icn, ym, ys);
    }

    return EXIT_SUCCESS;
}
