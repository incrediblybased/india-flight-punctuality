"""Model monthly on-time performance (OTP) for each airline at each major Indian airport.

Input : data/otp_airline_airport.csv (from parse_dgca.py), data/weather_monthly.csv (from fetch_weather.py)
Output: model/lgb.txt, model/meta.json, model/table.parquet, model/climate.parquet

OTP is DGCA's share of an airline's departures at that airport that left on time in the month.
Test: every report from April 2023 onwards (10 months, including the 2025 and 2026 reports with
Chennai, Kolkata and four more airports) is held out; the model only learns from March 2021 to March 2023.
Small data (about 1,000 rows), so the trees are kept small (4 leaves). The model predicts the change from
the last reported value (or, when there is none, the airline's average in the previous report), which
beats predicting the raw percentage on data this small. 4 threads, under a minute.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "model"
OUT.mkdir(exist_ok=True)
WX = ["rain_mm", "rainy_days", "heavy_rain_days", "thunder_days", "gusty_days", "max_temp", "min_temp", "fog_risk_hours"]
FEATURES = ["airline", "airport", "cal_month", "months_since_2021", "last_otp", "months_since_last",
            "airline_last_mean", "airport_last_mean"] + WX
CATEGORICAL = ["airline", "airport"]
TEST_FROM = "2023-04"


def build():
    c = pd.read_csv(DATA / "otp_airline_airport.csv")
    w = pd.read_csv(DATA / "weather_monthly.csv")
    c = c.merge(w, on=["airport", "month"], how="left")
    t = pd.to_datetime(c["month"])
    c["cal_month"] = t.dt.month
    c["months_since_2021"] = (t.dt.year - 2021) * 12 + t.dt.month - 1
    c = c.sort_values(["airline", "airport", "month"]).reset_index(drop=True)
    # recent form: the last report before this one (whatever the gap), never the same month
    g = c.groupby(["airline", "airport"])
    c["last_otp"] = g["otp"].shift(1)
    c["months_since_last"] = c["months_since_2021"] - g["months_since_2021"].shift(1)
    # the airline's and the airport's average across the previous report
    months = sorted(c["month"].unique())
    prev = {m: months[i - 1] if i else None for i, m in enumerate(months)}
    a_mean = c.groupby(["month", "airline"])["otp"].mean()
    p_mean = c.groupby(["month", "airport"])["otp"].mean()
    c["airline_last_mean"] = [a_mean.get((prev[m], a), np.nan) for m, a in zip(c["month"], c["airline"])]
    c["airport_last_mean"] = [p_mean.get((prev[m], p), np.nan) for m, p in zip(c["month"], c["airport"])]
    return c


def matrix(c, cats):
    X = c[FEATURES].copy()
    for k in CATEGORICAL:
        X[k] = pd.Categorical(X[k], categories=cats[k])
    for k in FEATURES:
        if k not in CATEGORICAL:
            X[k] = X[k].astype("float32")
    return X


def main():
    c = build()
    cats = {k: sorted(c[k].unique()) for k in CATEGORICAL}
    tr, te = c["month"] < TEST_FROM, c["month"] >= TEST_FROM
    va = tr & (c["month"] >= "2022-10")           # last six training months choose the tree count
    fit = tr & ~va
    X = matrix(c, cats)
    anchor = c["last_otp"].fillna(c["airline_last_mean"]).fillna(c.loc[tr, "otp"].mean())
    yd = c["otp"] - anchor
    params = dict(objective="l1", learning_rate=0.03, num_leaves=4, min_data_in_leaf=15, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, cat_smooth=20, min_data_per_group=10,
                  num_threads=4, verbose=-1, seed=7)
    m1 = lgb.train(params, lgb.Dataset(X[fit], yd[fit], categorical_feature=CATEGORICAL),
                   num_boost_round=2000, valid_sets=[lgb.Dataset(X[va], yd[va])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    n_trees = max(m1.best_iteration, 50)
    model = lgb.train(params, lgb.Dataset(X[tr], yd[tr], categorical_feature=CATEGORICAL), num_boost_round=n_trees)
    p = np.clip(anchor[te].values + model.predict(X[te]), 0, 100)
    y = c.loc[te, "otp"].values

    last = anchor[te].values          # last reported value, or the airline's previous average when there is none
    a_mean = c[tr].groupby("airline")["otp"].mean()
    base_airline = c.loc[te, "airline"].map(a_mean).fillna(c.loc[tr, "otp"].mean()).values

    def m(pred):
        ok = ~np.isnan(pred)
        e = pred[ok] - y[ok]
        return dict(mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                    within5=float(np.mean(np.abs(e) <= 5)), n=int(ok.sum()))
    res = {"LightGBM": m(p), "Last reported value": m(last), "Airline average": m(base_airline)}
    by_period = {}
    for name, mask in (("Apr to Sep 2023", c.loc[te, "month"] < "2024-01"), ("2025 and 2026", c.loc[te, "month"] >= "2024-01")):
        mk = mask.values
        by_period[name] = {k: float(np.mean(np.abs(v[mk] - y[mk]))) for k, v in
                           (("LightGBM", p), ("Last reported value", last),
                            ("Airline average", base_airline))}
    print({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in res.items()})
    print(by_period, "trees", n_trees)

    # final model on every report, for the app
    final = lgb.train(params, lgb.Dataset(X, yd, categorical_feature=CATEGORICAL), num_boost_round=n_trees)
    final.save_model(str(OUT / "lgb.txt"))
    out = c.copy()
    out["anchor"] = anchor
    out["pred_test"] = np.nan
    out.loc[te, "pred_test"] = p
    out.to_parquet(OUT / "table.parquet", index=False)
    w = pd.read_csv(DATA / "weather_monthly.csv")
    w["cal_month"] = pd.to_datetime(w["month"]).dt.month
    w.groupby(["airport", "cal_month"])[WX].mean().reset_index().to_parquet(OUT / "climate.parquet", index=False)
    reasons = pd.read_csv(DATA / "delay_reasons.csv")
    meta = dict(results=res, by_period=by_period, trees=int(n_trees), features=FEATURES, categorical=CATEGORICAL,
                categories=cats, months=sorted(c["month"].unique()), test_from=TEST_FROM, n_rows=int(len(c)),
                n_train=int(tr.sum()), n_test=int(te.sum()),
                importance=dict(zip(FEATURES, final.feature_importance("gain").tolist())),
                reasons=reasons.to_dict("records"))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print("saved")


if __name__ == "__main__":
    main()
