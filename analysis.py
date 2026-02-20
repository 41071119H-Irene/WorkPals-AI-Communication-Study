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

# Clean console warnings (keep important ones in models)
warnings.filterwarnings(
    "ignore",
    message="The default of observed=False is deprecated",
    category=FutureWarning
)


# =========================
# Logging (console progress)
# =========================
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# =========================
# Output helpers
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_run_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def outpath(results_dir: str, run_tag: str, name: str) -> str:
    return os.path.join(results_dir, f"{run_tag}_{name}")


def save_df(df: pd.DataFrame, results_dir: str, run_tag: str, name: str, index: bool = False) -> str:
    p = outpath(results_dir, run_tag, name)
    df.to_csv(p, index=index, encoding="utf-8-sig")
    return p


def save_plotly(fig, results_dir: str, run_tag: str, filename_stub: str) -> Dict[str, str]:
    """Save Plotly figure to HTML (always) and PNG (if kaleido available)."""
    paths: Dict[str, str] = {}
    html_path = outpath(results_dir, run_tag, f"{filename_stub}.html")
    fig.write_html(html_path)
    paths["html"] = html_path

    png_path = outpath(results_dir, run_tag, f"{filename_stub}.png")
    try:
        fig.write_image(png_path, scale=2)
        paths["png"] = png_path
    except Exception:
        # kaleido not installed or image export failed
        pass

    return paths


# =========================
# Data helpers
# =========================
def to_likert_numeric(series: pd.Series) -> pd.Series:
    """Chinese Likert text -> 1..5; keep numeric if already numeric."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return series.astype(str).str.strip().map(LIKERT5_MAP).astype(float)


def cronbach_alpha(df_items: pd.DataFrame) -> float:
    """Cronbach's alpha; input dataframe each column is an item."""
    df = df_items.dropna(axis=0, how="any")
    k = df.shape[1]
    if k < 2 or df.shape[0] < 3:
        return np.nan
    item_vars = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def extract_group(subject_id: str) -> str:
    """Axx / Bxx -> A or B"""
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
# Core scoring
# =========================
def score_scales(
    df: pd.DataFrame,
    comm_cols: List[str],
    ps_cols: List[str],
    stai_cols: List[str],
    trust_cols: List[str],
    ps_reverse: List[str],
    stai_reverse: List[str],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Score scales for a single sheet:
    - converts to numeric
    - reverse codes
    - computes mean scale scores with a minimum answered-items threshold
    Returns: (scored_df, missing_dict)
    """
    out = df.copy()

    comm_now = [c for c in comm_cols if c in out.columns]
    ps_now = [c for c in ps_cols if c in out.columns]
    stai_now = [c for c in stai_cols if c in out.columns]
    trust_now = [c for c in trust_cols if c in out.columns]

    ps_rev_now = [c for c in ps_reverse if c in out.columns]
    stai_rev_now = [c for c in stai_reverse if c in out.columns]

    # convert to numeric
    for c in comm_now + ps_now + stai_now + trust_now:
        out[c] = to_likert_numeric(out[c])

    # reverse code (1..5 -> 6-score)
    for c in ps_rev_now:
        out[c] = 6 - out[c]
    for c in stai_rev_now:
        out[c] = 6 - out[c]

    def row_mean_with_min(x: pd.Series, min_n: int) -> float:
        valid = x.dropna()
        if len(valid) < min_n:
            return np.nan
        return float(valid.mean())

    out["comm_eff"] = (
        out[comm_now].apply(lambda r: row_mean_with_min(r, min_n=max(3, len(comm_now) // 2)), axis=1)
        if comm_now
        else np.nan
    )
    out["psych_safety"] = (
        out[ps_now].apply(lambda r: row_mean_with_min(r, min_n=max(4, len(ps_now) // 2)), axis=1)
        if ps_now
        else np.nan
    )
    out["stai6"] = (
        out[stai_now].apply(lambda r: row_mean_with_min(r, min_n=max(3, len(stai_now) // 2)), axis=1)
        if stai_now
        else np.nan
    )
    out["trust_ai"] = (
        out[trust_now].apply(lambda r: row_mean_with_min(r, min_n=max(3, len(trust_now) // 2)), axis=1)
        if trust_now
        else np.nan
    )

    missing = {
        "missing_comm": sorted(set(comm_cols) - set(comm_now)),
        "missing_ps": sorted(set(ps_cols) - set(ps_now)),
        "missing_stai": sorted(set(stai_cols) - set(stai_now)),
        "missing_trust": sorted(set(trust_cols) - set(trust_now)),
    }
    return out, missing


# =========================
# Modeling helpers
# =========================
def extract_model_table(result, outcome: str, model_type: str, extra: Dict[str, Any]) -> pd.DataFrame:
    """
    Standardize coefficient table across MixedLM / GEE / OLS.
    Output columns: outcome, model_type, term, coef, se, pvalue, ci_low, ci_high, + extra cols
    """
    params = getattr(result, "params", pd.Series(dtype=float))
    bse = getattr(result, "bse", pd.Series([np.nan] * len(params), index=params.index))
    pvalues = getattr(result, "pvalues", pd.Series([np.nan] * len(params), index=params.index))

    try:
        ci = result.conf_int()
        ci_low = ci.iloc[:, 0]
        ci_high = ci.iloc[:, 1]
    except Exception:
        ci_low = pd.Series([np.nan] * len(params), index=params.index)
        ci_high = pd.Series([np.nan] * len(params), index=params.index)

    df = pd.DataFrame({
        "outcome": outcome,
        "model_type": model_type,
        "term": params.index.astype(str),
        "coef": params.values.astype(float),
        "se": bse.values.astype(float),
        "pvalue": pvalues.values.astype(float),
        "ci_low": ci_low.values.astype(float),
        "ci_high": ci_high.values.astype(float),
    })

    for k, v in extra.items():
        df[k] = v

    return df


def fit_repeated_model(
    long_df: pd.DataFrame,
    outcome: str,
    warnings_bucket: List[str],
) -> Tuple[str, Any, pd.DataFrame, Dict[str, Any]]:
    """
    Fit repeated measures model:
    - Try MixedLM with multiple optimizers
    - If fails, fallback to GEE (Exchangeable)
    Returns: (model_type, result_obj, used_data_df, fit_meta)
    """
    df = long_df.dropna(subset=[outcome, "group", "time", "id"]).copy()

    # enforce ordering & baselines
    df["time"] = pd.Categorical(df["time"], categories=["pre", "post", "delay"], ordered=True)
    df["group"] = pd.Categorical(df["group"], categories=["A", "B"], ordered=True)

    # keep subjects with >=2 timepoints for this outcome
    cnt = df.groupby("id")[outcome].count()
    keep_ids = cnt[cnt >= 2].index
    df = df[df["id"].isin(keep_ids)].copy()

    formula = f"{outcome} ~ C(group, Treatment(reference='A')) * C(time, Treatment(reference='pre'))"

    methods = ["lbfgs", "powell", "cg", "nm"]
    last_err: Optional[Exception] = None

    for m in methods:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model = smf.mixedlm(formula, data=df, groups=df["id"], re_formula="1")
                res = model.fit(reml=False, method=m, maxiter=2000, disp=False)
                for wi in w:
                    warnings_bucket.append(f"MixedLM({outcome},{m}): {wi.category.__name__}: {wi.message}")

            fit_meta = {
                "optimizer": m,
                "n_obs": int(getattr(res, "nobs", np.nan)),
                "n_subjects": int(df["id"].nunique()),
                "llf": float(getattr(res, "llf", np.nan)),
                "aic": float(getattr(res, "aic", np.nan)) if hasattr(res, "aic") else np.nan,
                "bic": float(getattr(res, "bic", np.nan)) if hasattr(res, "bic") else np.nan,
            }
            return "MixedLM", res, df, fit_meta
        except Exception as e:
            last_err = e
            warnings_bucket.append(f"MixedLM({outcome},{m}) failed: {type(e).__name__}: {e}")

    # fallback to GEE
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gee = smf.gee(
            formula,
            groups="id",
            data=df,
            cov_struct=sm.cov_struct.Exchangeable(),
            family=sm.families.Gaussian(),
        ).fit()
        for wi in w:
            warnings_bucket.append(f"GEE({outcome}): {wi.category.__name__}: {wi.message}")

    fit_meta = {
        "optimizer": "NA",
        "n_obs": int(getattr(gee, "nobs", np.nan)),
        "n_subjects": int(df["id"].nunique()),
        "llf": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "note": f"MixedLM failed; fallback to GEE. Last error: {type(last_err).__name__}: {last_err}" if last_err else "MixedLM failed; fallback to GEE.",
    }
    return "GEE", gee, df, fit_meta


# =========================
# Plotting (paper-oriented)
# =========================
def plot_means_plotly(
    long_df: pd.DataFrame,
    outcome: str,
    title: str,
    filename_stub: str,
    results_dir: str,
    run_tag: str,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = long_df.dropna(subset=[outcome]).copy()
    df["time"] = pd.Categorical(df["time"], categories=["pre", "post", "delay"], ordered=True)

    # ✅ fix pandas FutureWarning by specifying observed=True
    summary = (
        df.groupby(["group", "time"], observed=True)[outcome]
          .agg(["mean", "count", "std"])
          .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])

    fig = px.line(
        summary,
        x="time",
        y="mean",
        color="group",
        error_y="se",
        markers=True,
        title=title,
    )
    fig.update_layout(xaxis_title="time", yaxis_title=outcome)

    fig_paths = save_plotly(fig, results_dir, run_tag, filename_stub)
    return summary, fig_paths


# =========================
# Mediation / Moderation
# =========================
def bootstrap_mediation(
    df: pd.DataFrame,
    x: str,
    m: str,
    y: str,
    n_boot: int = 5000,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float], int]:
    """
    Simple bootstrap mediation:
      a: m ~ x
      b: y ~ m + x
      indirect = a*b
    """
    rng = np.random.default_rng(seed)
    df0 = df.dropna(subset=[x, m, y]).copy()
    n = int(df0.shape[0])

    vals = []
    for _ in range(n_boot):
        samp = df0.sample(frac=1.0, replace=True, random_state=int(rng.integers(1_000_000_000)))
        a = smf.ols(f"{m} ~ {x}", data=samp).fit().params.get(x, np.nan)
        b = smf.ols(f"{y} ~ {m} + {x}", data=samp).fit().params.get(m, np.nan)
        vals.append(a * b)

    vals = np.asarray(vals, dtype=float)
    est = float(np.mean(vals))
    ci_low, ci_high = np.quantile(vals, [0.025, 0.975])
    return est, (float(ci_low), float(ci_high)), n


# =========================
# Main
# =========================
def main(xlsx_path: str) -> None:
    ensure_dir(RESULTS_DIR)
    run_tag = make_run_tag()

    manifest_rows: List[Dict[str, Any]] = []
    warnings_bucket: List[str] = []

    log("Step 1/7  Reading Excel sheets...")
    bg = pd.read_excel(xlsx_path, sheet_name="受試者背景")
    pre = pd.read_excel(xlsx_path, sheet_name="前測")
    post = pd.read_excel(xlsx_path, sheet_name="後測")
    delay = pd.read_excel(xlsx_path, sheet_name="延後後測")

    # background: keep only valid IDs
    bg = keep_valid_ids(bg, "編號 (A實驗組；B控制組)")
    bg = bg.rename(columns={"編號 (A實驗組；B控制組)": "id", "身分別": "role"})
    bg["group"] = bg["id"].apply(extract_group)

    # unify id columns
    pre = pre.rename(columns={"實驗編號": "id"})
    post = post.rename(columns={"您的實驗編號": "id"})
    delay = delay.rename(columns={"您的實驗編號": "id"})

    pre = keep_valid_ids(pre, "id")
    post = keep_valid_ids(post, "id")
    delay = keep_valid_ids(delay, "id")

    manifest_rows.append({"key": "run_tag", "value": run_tag})
    manifest_rows.append({"key": "xlsx_path", "value": xlsx_path})
    manifest_rows.append({"key": "n_bg_valid_ids", "value": int(bg["id"].nunique())})

    log("Step 2/7  Detecting item columns (from PRE)...")
    COMM_PREFIX = "第一部分：個人職場溝通觀點評估"
    PS_PREFIX = "第二部分：職場互動環境感知"
    STAI_PREFIX = "第三部分：個人情緒感受評估"
    TRUST_PREFIX = "第四部分：科技工具使用傾向評估"

    comm_cols = find_cols_by_prefix(pre, COMM_PREFIX)
    ps_cols = find_cols_by_prefix(pre, PS_PREFIX)
    stai_cols = find_cols_by_prefix(pre, STAI_PREFIX)
    trust_cols = find_cols_by_prefix(pre, TRUST_PREFIX)

    # reverse items (keyword-based, robust)
    ps_reverse = [c for c in ps_cols if ("反向" in c) or ("排斥" in c)]
    stai_reverse = [c for c in stai_cols if ("反向" in c) or ("平靜" in c) or ("放鬆" in c) or ("滿足" in c)]

    manifest_rows += [
        {"key": "comm_items", "value": len(comm_cols)},
        {"key": "ps_items", "value": len(ps_cols)},
        {"key": "stai_items", "value": len(stai_cols)},
        {"key": "trust_items", "value": len(trust_cols)},
        {"key": "ps_reverse_items", "value": len(ps_reverse)},
        {"key": "stai_reverse_items", "value": len(stai_reverse)},
    ]

    log("Step 3/7  Scoring scales (PRE/POST/DELAY)...")
    pre_s, miss_pre = score_scales(pre, comm_cols, ps_cols, stai_cols, trust_cols, ps_reverse, stai_reverse)
    post_s, miss_post = score_scales(post, comm_cols, ps_cols, stai_cols, trust_cols, ps_reverse, stai_reverse)
    delay_s, miss_delay = score_scales(delay, comm_cols, ps_cols, stai_cols, trust_cols, ps_reverse, stai_reverse)

    # missing summary in manifest (keep it compact)
    def miss_to_str(miss: Dict[str, List[str]]) -> str:
        total = sum(len(v) for v in miss.values())
        # store only counts; full lists are too long
        return f"total_missing={total}; comm={len(miss['missing_comm'])}; ps={len(miss['missing_ps'])}; stai={len(miss['missing_stai'])}; trust={len(miss['missing_trust'])}"

    manifest_rows += [
        {"key": "missing_items_pre", "value": miss_to_str(miss_pre)},
        {"key": "missing_items_post", "value": miss_to_str(miss_post)},
        {"key": "missing_items_delay", "value": miss_to_str(miss_delay)},
    ]

    log("Step 4/7  Reliability (Cronbach alpha on PRE)...")
    reli = pd.DataFrame(
        [
            {"scale": "comm_eff", "alpha": cronbach_alpha(pre_s[[c for c in comm_cols if c in pre_s.columns]]), "n_items": len(comm_cols)},
            {"scale": "psych_safety", "alpha": cronbach_alpha(pre_s[[c for c in ps_cols if c in pre_s.columns]]), "n_items": len(ps_cols)},
            {"scale": "stai6", "alpha": cronbach_alpha(pre_s[[c for c in stai_cols if c in pre_s.columns]]), "n_items": len(stai_cols)},
            {"scale": "trust_ai", "alpha": cronbach_alpha(pre_s[[c for c in trust_cols if c in pre_s.columns]]), "n_items": len(trust_cols)},
        ]
    )
    reli_path = save_df(reli, RESULTS_DIR, run_tag, "reliability_pre.csv")
    manifest_rows.append({"key": "reliability_csv", "value": reli_path})

    log("Step 5/7  Building tidy long_df (for repeated measures)...")
    def to_long(df_scored: pd.DataFrame, time_label: str) -> pd.DataFrame:
        keep = df_scored[["id", "comm_eff", "psych_safety", "stai6", "trust_ai"]].copy()
        keep["time"] = time_label
        return keep

    long_df = pd.concat(
        [to_long(pre_s, "pre"), to_long(post_s, "post"), to_long(delay_s, "delay")],
        axis=0,
        ignore_index=True,
    )

    long_df = long_df.merge(bg[["id", "group", "role"]], on="id", how="left")
    baseline_trust = pre_s[["id", "trust_ai"]].rename(columns={"trust_ai": "trust_ai_pre"})
    long_df = long_df.merge(baseline_trust, on="id", how="left")
    long_df = long_df[long_df["group"].isin(["A", "B"])].copy()

    # counts table (compact: store to manifest)
    counts = long_df.groupby(["time", "group"])["id"].nunique().reset_index().rename(columns={"id": "n_unique_subjects"})
    for _, r in counts.iterrows():
        manifest_rows.append({"key": f"n_{r['time']}_{r['group']}", "value": int(r["n_unique_subjects"])})

    long_path = save_df(long_df, RESULTS_DIR, run_tag, "long_df.csv")
    manifest_rows.append({"key": "long_df_csv", "value": long_path})

    log("Step 6/7  Fitting models (MixedLM -> fallback GEE if needed)...")
    all_models: List[pd.DataFrame] = []

    for outcome in ["stai6", "psych_safety", "comm_eff"]:
        log(f"  - Modeling outcome: {outcome}")
        model_type, res, used_df, fit_meta = fit_repeated_model(long_df, outcome, warnings_bucket)
        extra = {
            "optimizer": fit_meta.get("optimizer", "NA"),
            "n_obs": fit_meta.get("n_obs", np.nan),
            "n_subjects": fit_meta.get("n_subjects", np.nan),
            "llf": fit_meta.get("llf", np.nan),
            "aic": fit_meta.get("aic", np.nan),
            "bic": fit_meta.get("bic", np.nan),
        }
        if "note" in fit_meta:
            extra["note"] = fit_meta["note"]

        all_models.append(extract_model_table(res, outcome, model_type, extra))

        manifest_rows.append({"key": f"model_{outcome}_type", "value": model_type})
        manifest_rows.append({"key": f"model_{outcome}_n_subjects", "value": int(used_df['id'].nunique())})
        manifest_rows.append({"key": f"model_{outcome}_n_obs", "value": int(used_df.shape[0])})

    models_df = pd.concat(all_models, axis=0, ignore_index=True)
    models_path = save_df(models_df, RESULTS_DIR, run_tag, "models.csv")
    manifest_rows.append({"key": "models_csv", "value": models_path})

    log("Step 7/7  Hypothesis tests + Plotly figures...")
    # Paper-oriented plots: group x time for key outcomes
    fig_meta = []

    for outcome, title, stub in [
        ("psych_safety", "Psychological Safety across time by group", "fig_psafety_by_group_time"),
        ("stai6", "State Anxiety (STAI-6) across time by group", "fig_stai6_by_group_time"),
        ("comm_eff", "Communication Self-Efficacy across time by group", "fig_comm_by_group_time"),
    ]:
        log(f"  - Plot: {stub}")
        summary, paths = plot_means_plotly(long_df, outcome, title, stub, RESULTS_DIR, run_tag)
        # store only figure paths in manifest; summary can be regenerated from long_df
        fig_meta.append({"figure": stub, "html": paths.get("html", ""), "png": paths.get("png", "")})

    # Wide format for mediation/moderation (kept in memory; not saved as separate CSV)
    wide = long_df.pivot_table(
        index=["id", "group", "trust_ai_pre"],
        columns="time",
        values=["stai6", "psych_safety", "comm_eff"],
        aggfunc="mean",
        observed=True,
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    # change scores (post-pre)
    wide["d_stai_post"] = wide.get("stai6_post") - wide.get("stai6_pre")
    wide["d_ps_post"] = wide.get("psych_safety_post") - wide.get("psych_safety_pre")
    wide["d_comm_post"] = wide.get("comm_eff_post") - wide.get("comm_eff_pre")
    wide["groupA"] = (wide["group"] == "A").astype(int)

    hyp_rows: List[Dict[str, Any]] = []

    # Mediation: group -> (Δanxiety) -> (Δpsych safety)
    log("  - Mediation (bootstrap): groupA -> d_stai_post -> d_ps_post")
    ind, (ci_low, ci_high), n_med = bootstrap_mediation(
        wide, x="groupA", m="d_stai_post", y="d_ps_post", n_boot=5000, seed=42
    )
    hyp_rows.append({
        "analysis": "mediation",
        "x": "groupA",
        "m": "d_stai_post",
        "y": "d_ps_post",
        "effect": "indirect_ab",
        "estimate": ind,
        "ci_low_95": ci_low,
        "ci_high_95": ci_high,
        "n_used": n_med,
        "n_boot": 5000,
    })

    # Moderation: trust_ai_pre moderates group effect on Δcomm
    log("  - Moderation (OLS): d_comm_post ~ groupA * trust_ai_pre")
    mod_df = wide.dropna(subset=["d_comm_post", "groupA", "trust_ai_pre"]).copy()
    mod = smf.ols("d_comm_post ~ groupA * trust_ai_pre", data=mod_df).fit()

    # keep only key terms for write-up clarity
    for term in ["groupA", "trust_ai_pre", "groupA:trust_ai_pre"]:
        hyp_rows.append({
            "analysis": "moderation",
            "dv": "d_comm_post",
            "term": term,
            "coef": float(mod.params.get(term, np.nan)),
            "se": float(mod.bse.get(term, np.nan)),
            "pvalue": float(mod.pvalues.get(term, np.nan)),
            "n_used": int(mod_df.shape[0]),
        })

    hyp_df = pd.DataFrame(hyp_rows)
    hyp_path = save_df(hyp_df, RESULTS_DIR, run_tag, "hypothesis_results.csv")
    manifest_rows.append({"key": "hypothesis_results_csv", "value": hyp_path})

    # Add figure paths into manifest (compact)
    for f in fig_meta:
        manifest_rows.append({"key": f"figure_{f['figure']}_html", "value": f["html"]})
        if f.get("png"):
            manifest_rows.append({"key": f"figure_{f['figure']}_png", "value": f["png"]})

    # store warnings compactly (top 15)
    if warnings_bucket:
        manifest_rows.append({"key": "warnings_count", "value": len(warnings_bucket)})
        top = warnings_bucket[:15]
        for i, w in enumerate(top, start=1):
            manifest_rows.append({"key": f"warning_{i}", "value": w})

    # Save manifest LAST (so it references all artifacts)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = save_df(manifest, RESULTS_DIR, run_tag, "manifest.csv")
    log(f"Done ✅  Outputs saved under: {RESULTS_DIR}/")
    log(f"Key files: manifest={manifest_path}, models={models_path}, long_df={long_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=str, default=DEFAULT_XLSX_PATH, help="Path to the Batch 1 Excel file")
    args = parser.parse_args()
    main(args.xlsx)