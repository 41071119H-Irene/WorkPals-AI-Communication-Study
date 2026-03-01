import argparse
import os
import re
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =========================
# Config
# =========================
DEFAULT_XLSX_PATH = "data_all.xlsx"
RESULTS_DIR = "results"

LIKERT5_MAP = {
    "非常不同意": 1,
    "不同意": 2,
    "普通": 3,
    "同意": 4,
    "非常同意": 5,
    "-": np.nan,
}

warnings.filterwarnings(
    "ignore",
    message="The default of observed=False is deprecated",
    category=FutureWarning
)

# =========================
# Logging & Helpers
# =========================
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def make_run_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def outpath(results_dir: str, run_tag: str, name: str) -> str:
    return os.path.join(results_dir, f"{run_tag}_{name}")

def save_df(df: pd.DataFrame, results_dir: str, run_tag: str, name: str, index: bool = False) -> str:
    p = outpath(results_dir, run_tag, name)
    df.to_csv(p, index=index, encoding="utf-8-sig")
    log(f"儲存 CSV: {p}")
    return p

def save_plotly(fig, results_dir: str, run_tag: str, filename_stub: str) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    html_path = outpath(results_dir, run_tag, f"{filename_stub}.html")
    fig.write_html(html_path)
    paths["html"] = html_path
    png_path = outpath(results_dir, run_tag, f"{filename_stub}.png")
    try:
        fig.write_image(png_path, scale=2)
        paths["png"] = png_path
    except Exception:
        pass
    return paths

# =========================
# Data Helpers
# =========================
def to_likert_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return series.astype(str).str.strip().map(LIKERT5_MAP).astype(float)

def cronbach_alpha(df_items: pd.DataFrame) -> float:
    df = df_items.dropna(axis=0, how="any")
    k = df.shape[1]
    if k < 2 or df.shape[0] < 3: return np.nan
    item_vars = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var == 0: return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)

def extract_group(subject_id: str) -> str:
    if isinstance(subject_id, str) and len(subject_id) > 0:
        return subject_id.strip()[0].upper()
    return np.nan

def keep_valid_ids(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    pat = re.compile(r"^[AB]\d+")
    out = df.copy()
    out[id_col] = out[id_col].astype(str).str.strip()
    return out[out[id_col].apply(lambda x: bool(pat.match(x)))].copy()

def find_cols_by_prefix(df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]

# =========================
# Scoring
# =========================
def score_scales(df: pd.DataFrame, comm_cols, ps_cols, stai_cols, trust_cols, ps_reverse, stai_reverse):
    out = df.copy()
    cols_to_num = [c for c in comm_cols+ps_cols+stai_cols+trust_cols if c in out.columns]
    for c in cols_to_num: out[c] = to_likert_numeric(out[c])
    for c in [c for c in ps_reverse if c in out.columns]: out[c] = 6 - out[c]
    for c in [c for c in stai_reverse if c in out.columns]: out[c] = 6 - out[c]

    def row_mean_with_min(x, min_n):
        v = x.dropna()
        return float(v.mean()) if len(v) >= min_n else np.nan

    out["comm_eff"] = out[[c for c in comm_cols if c in out.columns]].apply(lambda r: row_mean_with_min(r, 3), axis=1)
    out["psych_safety"] = out[[c for c in ps_cols if c in out.columns]].apply(lambda r: row_mean_with_min(r, 4), axis=1)
    out["stai6"] = out[[c for c in stai_cols if c in out.columns]].apply(lambda r: row_mean_with_min(r, 3), axis=1)
    out["trust_ai"] = out[[c for c in trust_cols if c in out.columns]].apply(lambda r: row_mean_with_min(r, 3), axis=1)
    return out

# =========================
# Modeling Helpers
# =========================
def extract_model_table(result, outcome, model_type, extra):
    params = getattr(result, "params", pd.Series(dtype=float))
    bse = getattr(result, "bse", pd.Series([np.nan]*len(params), index=params.index))
    pvalues = getattr(result, "pvalues", pd.Series([np.nan]*len(params), index=params.index))
    try:
        ci = result.conf_int()
        low, high = ci.iloc[:, 0], ci.iloc[:, 1]
    except:
        low = high = pd.Series([np.nan]*len(params), index=params.index)
    
    df = pd.DataFrame({
        "outcome": outcome, 
        "model_type": model_type, 
        "term": params.index, 
        "coef": params.values, 
        "se": bse.values, 
        "pvalue": pvalues.values, 
        "ci_low": low.values, 
        "ci_high": high.values
    })
    for k, v in extra.items(): df[k] = v
    return df

def fit_repeated_model(long_df, outcome, warnings_bucket):
    df = long_df.dropna(subset=[outcome, "group", "time", "id"]).copy()
    df["time"] = pd.Categorical(df["time"], categories=["pre", "post", "delay"], ordered=True)
    df["group"] = pd.Categorical(df["group"], categories=["A", "B"], ordered=True)
    cnt = df.groupby("id")[outcome].count()
    df = df[df["id"].isin(cnt[cnt >= 2].index)].copy()
    formula = f"{outcome} ~ C(group, Treatment(reference='A')) * C(time, Treatment(reference='pre'))"
    methods = ["lbfgs", "powell", "cg", "nm"]
    for m in methods:
        try:
            model = smf.mixedlm(formula, data=df, groups=df["id"], re_formula="1")
            res = model.fit(reml=False, method=m, maxiter=2000, disp=False)
            return "MixedLM", res, df, {"optimizer": m, "n_obs": int(res.nobs), "n_subjects": df["id"].nunique()}
        except Exception as e:
            warnings_bucket.append(f"MixedLM({outcome},{m}) 失敗: {e}")
    gee = smf.gee(formula, groups="id", data=df, cov_struct=sm.cov_struct.Exchangeable(), family=sm.families.Gaussian()).fit()
    return "GEE", gee, df, {"optimizer": "NA", "n_obs": int(gee.nobs), "n_subjects": df["id"].nunique()}

# =========================
# Plotting (Optimized Colors & Styles)
# =========================
def plot_means_plotly(long_df, outcome, title, filename_stub, results_dir, run_tag):
    # 1. 定義英文圖例名稱
    LEGEND_MAP = {
        "A_在學實習生": "Exp: Student Intern",
        "A_職場新鮮人": "Exp: Early Career",
        "B_在學實習生": "Ctrl: Student Intern",
        "B_職場新鮮人": "Ctrl: Early Career"
    }

    # 2. 定義顏色映射 (實驗組深色/控制組淺色；新鮮人紅/實習生藍)
    COLOR_MAP = {
        "Exp: Early Career": "#B22222",      # 深紅 (Firebrick)
        "Ctrl: Early Career": "#FF6B6B",     # 淺紅
        "Exp: Student Intern": "#191970",    # 深藍 (Midnight Blue)
        "Ctrl: Student Intern": "#74C0FC"    # 淺藍
    }

    # 3. 定義線條樣式映射 (新鮮人實線/實習生虛線)
    DASH_MAP = {
        "Exp: Early Career": "solid",
        "Ctrl: Early Career": "solid",
        "Exp: Student Intern": "dash",
        "Ctrl: Student Intern": "dash"
    }
    
    Y_LABEL_MAP = {
        "psych_safety": "Psychological Safety Score",
        "stai6": "State Anxiety Score (STAI-6)",
        "comm_eff": "Communication Self-Efficacy Score"
    }

    df = long_df.dropna(subset=[outcome, "group", "role"]).copy()
    df["role"] = df["role"].astype(str).str.strip()
    df = df[df["role"].isin(["在學實習生", "職場新鮮人"])].copy()
    
    if df.empty:
        log(f"⚠️ Warning: No data for {outcome} after filtering.")
        return None, {}

    df["time"] = pd.Categorical(df["time"], categories=["pre", "post", "delay"], ordered=True)
    df["group_role"] = df["group"].astype(str) + "_" + df["role"].astype(str)

    summary = (
        df.groupby(["group_role", "time"], observed=True)[outcome]
          .agg(["mean", "count", "std"])
          .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["Legend"] = summary["group_role"].map(LEGEND_MAP)

    fig = px.line(
        summary,
        x="time",
        y="mean",
        color="Legend",
        line_dash="Legend",
        color_discrete_map=COLOR_MAP, # 套用自定義顏色
        line_dash_map=DASH_MAP,       # 套用自定義實虛線
        error_y="se",
        markers=True,
        title=title,
        category_orders={"Legend": ["Exp: Early Career", "Ctrl: Early Career", "Exp: Student Intern", "Ctrl: Student Intern"]}
    )
    
    fig.update_layout(
        xaxis_title="Time Point", 
        yaxis_title=Y_LABEL_MAP.get(outcome, outcome), 
        legend_title="Group & Status",
        font=dict(family="Arial", size=13),
        # 加強背景對比度
        plot_bgcolor="white"
    )
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    fig_paths = save_plotly(fig, results_dir, run_tag, filename_stub)
    return summary, fig_paths

# =========================
# Main Execution
# =========================
def main(xlsx_path: str):
    ensure_dir(RESULTS_DIR)
    run_tag = make_run_tag()
    log(f"開始分析... 執行標籤: {run_tag}")

    log("Step 1/7 讀取 Excel 分頁...")
    try:
        bg = pd.read_excel(xlsx_path, sheet_name="受試者背景")
        pre = pd.read_excel(xlsx_path, sheet_name="前測").rename(columns={"實驗編號": "id"})
        post = pd.read_excel(xlsx_path, sheet_name="後測").rename(columns={"您的實驗編號": "id"})
        delay = pd.read_excel(xlsx_path, sheet_name="延後後測").rename(columns={"您的實驗編號": "id"})
    except Exception as e:
        log(f"❌ 讀取 Excel 失敗: {e}")
        return

    bg = keep_valid_ids(bg, "編號 (A實驗組；B控制組)").rename(columns={"編號 (A實驗組；B控制組)": "id", "身分別": "role"})
    bg["group"] = bg["id"].apply(extract_group)
    pre, post, delay = keep_valid_ids(pre, "id"), keep_valid_ids(post, "id"), keep_valid_ids(delay, "id")

    log("Step 2/7 偵測量表題項...")
    comm_cols = find_cols_by_prefix(pre, "第一部分")
    ps_cols = find_cols_by_prefix(pre, "第二部分")
    stai_cols = find_cols_by_prefix(pre, "第三部分")
    trust_cols = find_cols_by_prefix(pre, "第四部分")
    ps_reverse = [c for c in ps_cols if any(k in c for k in ["反向", "排斥"])]
    stai_reverse = [c for c in stai_cols if any(k in c for k in ["反向", "平靜", "放鬆", "滿足"])]

    log("Step 3/7 計算量表得分...")
    pre_s = score_scales(pre, comm_cols, ps_cols, stai_cols, trust_cols, ps_reverse, stai_reverse)
    post_s = score_scales(post, comm_cols, ps_cols, stai_cols, trust_cols, ps_reverse, stai_reverse)
    delay_s = score_scales(delay, comm_cols, ps_cols, stai_cols, trust_cols, ps_reverse, stai_reverse)

    log("Step 4/7 信度分析...")
    reli = pd.DataFrame([
        {"scale": "comm_eff", "alpha": cronbach_alpha(pre_s[[c for c in comm_cols if c in pre_s.columns]])},
        {"scale": "psych_safety", "alpha": cronbach_alpha(pre_s[[c for c in ps_cols if c in pre_s.columns]])},
        {"scale": "stai6", "alpha": cronbach_alpha(pre_s[[c for c in stai_cols if c in pre_s.columns]])},
        {"scale": "trust_ai", "alpha": cronbach_alpha(pre_s[[c for c in trust_cols if c in pre_s.columns]])},
    ])
    save_df(reli, RESULTS_DIR, run_tag, "reliability_pre.csv")

    log("Step 5/7 資料合併...")
    def to_long(df, t):
        k = df[["id", "comm_eff", "psych_safety", "stai6", "trust_ai"]].copy()
        k["time"] = t
        return k

    long_df = pd.concat([to_long(pre_s, "pre"), to_long(post_s, "post"), to_long(delay_s, "delay")], axis=0, ignore_index=True)
    long_df = long_df.merge(bg[["id", "group", "role"]], on="id", how="left")
    long_df = long_df[long_df["group"].isin(["A", "B"])].copy()
    save_df(long_df, RESULTS_DIR, run_tag, "long_df.csv")

    log("Step 6/7 執行統計模型...")
    all_models, warnings_bucket = [], []
    for outcome in ["stai6", "psych_safety", "comm_eff"]:
        try:
            mtype, res, _, meta = fit_repeated_model(long_df, outcome, warnings_bucket)
            all_models.append(extract_model_table(res, outcome, mtype, meta))
        except Exception as e:
            log(f"⚠️ 模型 {outcome} 失敗: {e}")

    if all_models:
        save_df(pd.concat(all_models), RESULTS_DIR, run_tag, "models.csv")

    log("Step 7/7 產生視覺化圖表...")
    for outcome, title, stub in [
        ("psych_safety", "Psychological Safety across Time by Group & Status", "fig_psafety_by_group_time"),
        ("stai6", "State Anxiety (STAI-6) across Time by Group & Status", "fig_stai6_by_group_time"),
        ("comm_eff", "Communication Self-Efficacy across Time by Group & Status", "fig_comm_by_group_time"),
    ]:
        plot_means_plotly(long_df, outcome, title, stub, RESULTS_DIR, run_tag)

    log(f"分析完成！請至 '{RESULTS_DIR}' 資料夾查看結果。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=str, default=DEFAULT_XLSX_PATH)
    args = parser.parse_args()
    main(args.xlsx)