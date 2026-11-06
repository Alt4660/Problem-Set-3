"""Difference-in-differences toolkit, using Lab 11 did_utils.py."""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

NEVER = 10 ** 9      # the cohort of a never-treated unit: later than any period

# Lab 11's functions, carried over

def two_by_two(df: pd.DataFrame, outcome: str, treated: str, post: str) -> float:
    """The canonical 2x2: difference of differences of the four group means."""
    g = df.groupby([treated, post])[outcome].mean()
    return (g[1, 1] - g[1, 0]) - (g[0, 1] - g[0, 0])


def att_by_cohort_period(df: pd.DataFrame, time: str, outcome: str,
                         cohort: str) -> pd.DataFrame:
    """ATT(g, t) for every cohort g and period t >= g, using not-yet-treated controls."""
    out = []
    cohorts = sorted(c for c in df[cohort].unique() if c < NEVER)
    for g in cohorts:
        tr = df[df[cohort] == g]
        for t in sorted(df.loc[df[time] >= g, time].unique()):
            co = df[df[cohort] > t]              # not yet treated at t
            if co.empty or (tr[time] == g - 1).sum() == 0:
                continue
            d_tr = (tr.loc[tr[time] == t, outcome].mean()
                    - tr.loc[tr[time] == g - 1, outcome].mean())
            d_co = (co.loc[co[time] == t, outcome].mean()
                    - co.loc[co[time] == g - 1, outcome].mean())
            out.append({"g": g, "t": t, "e": t - g,
                        "att": d_tr - d_co, "n": int((tr[time] == t).sum())})
    return pd.DataFrame(out)


def overall_att(cells: pd.DataFrame) -> float:
    """Size-weighted average of the ATT(g, t) cells."""
    return float(np.average(cells["att"], weights=cells["n"]))


def event_study(cells: pd.DataFrame) -> pd.DataFrame:
    """Average the ATT
    (g, t) cells by event time e = t - g, weighted by size."""
    rows = []
    for e, group in cells.groupby("e"):
        rows.append({"e": e, "att": np.average(group["att"], weights=group["n"])})
    return pd.DataFrame(rows)


# your clustered-inference wrapper (Step 3.3)

def twfe_clustered(df: pd.DataFrame, outcome: str, treat: str, unit: str,
                   time: str, cluster: str | None = None) -> dict:
    """TWFE estimate of the treat coefficient, with default or clustered standard errors. Pass cluster=None for the default (iid) standard error. 
    Or the name of the column to cluster on. Returns the estimate, standard error and p-value."""
    formula = f"{outcome} ~ {treat} + C({unit}) + C({time})"
    model = smf.ols(formula, data=df)
    if cluster is None:
        fit = model.fit()
    else:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster]})
    return {"estimate": float(fit.params[treat]),
            "se": float(fit.bse[treat]),
            "p": float(fit.pvalues[treat])}

def wild_cluster_bootstrap(df, cluster, outcome="y", treat="D",
                           fe="C(county) + C(t)", tol=1e-9):
    df = df.reset_index(drop=True)
    codes, _ = pd.factorize(df[cluster])
    G = codes.max() + 1
    unrestricted = smf.ols(f"{outcome} ~ {treat} + {fe}", data=df)
    X = unrestricted.exog
    j = unrestricted.exog_names.index(treat)
    y = unrestricted.endog
    cov_kw = {"groups": codes}
    fit_z = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds=cov_kw)
    fit_t = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds=cov_kw, use_t=True)
    t_actual = float(fit_z.tvalues[j])
    restricted = smf.ols(f"{outcome} ~ {fe}", data=df).fit()
    y_tilde = restricted.fittedvalues.to_numpy()
    u_tilde = restricted.resid.to_numpy()
    weights = np.array(list(itertools.product([-1.0, 1.0], repeat=G)))
    t_star = np.empty(len(weights))
    for k, w in enumerate(weights):
        y_star = y_tilde + w[codes] * u_tilde
        f = sm.OLS(y_star, X).fit(cov_type="cluster", cov_kwds=cov_kw)
        t_star[k] = f.tvalues[j]
    p_boot = float(np.mean(np.abs(t_star) >= abs(t_actual) - tol * max(1.0, abs(t_actual))))
    return {"beta": float(fit_z.params[j]), "t_actual": t_actual, "G": int(G),
            "n_weight_vectors": len(weights), "p_boot": p_boot,
            "p_cluster_normal": float(fit_z.pvalues[j]),
            "p_cluster_t_G_minus_1": float(fit_t.pvalues[j]),
            "t_star": t_star, "weights": weights}


if __name__ == "__main__":
    # a tiny panel with a known answer: 4 units, 4 periods, treated from period 2
    toy = pd.DataFrame({
        "unit": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
        "t": [0, 1, 2, 3] * 4,
        "yr_adopt": [2] * 8 + [NEVER] * 8,
        "board": [1] * 8 + [2] * 8,
    })
    toy["D"] = (toy["t"] >= toy["yr_adopt"]).astype(int)
    toy["y"] = toy["unit"] * 1.0 + toy["t"] * 0.5 + 3.0 * toy["D"]

    toy["post"] = (toy["t"] >= 2).astype(int)
    toy["treated"] = (toy["yr_adopt"] < NEVER).astype(int)
    assert abs(two_by_two(toy, "y", "treated", "post") - 3.0) < 1e-9

    cells = att_by_cohort_period(toy, "t", "y", "yr_adopt")
    assert abs(overall_att(cells) - 3.0) < 1e-9
    assert set(event_study(cells)["e"]) == {0, 1}

    out = twfe_clustered(toy, "y", "D", "unit", "t")
    assert abs(out["estimate"] - 3.0) < 1e-9
    assert set(out) == {"estimate", "se", "p"}
    print("did_toolkit: all assertions passed")

if __name__ == "__main__":
    ...   # passing assertions
