"""Daily weather 2021 to 2026 at the 10 airports DGCA reports on, from the Open-Meteo archive (free, no key).
Output: data/weather_daily.csv and data/weather_monthly.csv (rain, rainy days, fog days, gusts, heat, cold)."""
import gzip
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
AIRPORTS = {
    "DEL": ("Delhi", 28.5562, 77.1000), "BOM": ("Mumbai", 19.0896, 72.8656), "BLR": ("Bengaluru", 13.1986, 77.7066),
    "HYD": ("Hyderabad", 17.2403, 78.4294), "MAA": ("Chennai", 12.9941, 80.1709), "CCU": ("Kolkata", 22.6547, 88.4467),
    "AMD": ("Ahmedabad", 23.0772, 72.6347), "COK": ("Kochi", 10.1520, 76.4019), "GAU": ("Guwahati", 26.1061, 91.5859),
    "LKO": ("Lucknow", 26.7606, 80.8893),
}
VARS = ["precipitation_sum", "weather_code", "wind_gusts_10m_max", "temperature_2m_max", "temperature_2m_min"]


def main():
    rows = []
    for code, (_city, lat, lon) in AIRPORTS.items():
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               f"&start_date=2021-01-01&end_date=2026-10-31&daily={','.join(VARS)}&timezone=Asia%2FKolkata")
        req = urllib.request.Request(url, headers={"User-Agent": "india-otp-study/1.0", "Accept-Encoding": "gzip"})
        for attempt in range(6):
            try:
                r = urllib.request.urlopen(req, timeout=120)
                body = r.read()
                if body[:2] == bytes([0x1F, 0x8B]):   # gzip magic number
                    body = gzip.decompress(body)
                d = pd.DataFrame(json.loads(body)["daily"])
                break
            except Exception as e:
                print(code, "retry", attempt, repr(e)[:120])
                time.sleep(15 * (attempt + 1))
        d["airport"] = code
        rows.append(d)
        print(code, len(d))
        time.sleep(1)
    w = pd.concat(rows, ignore_index=True)
    w["time"] = pd.to_datetime(w["time"])
    w.to_csv(DATA / "weather_daily.csv", index=False)
    w["month"] = w["time"].dt.strftime("%Y-%m")
    w["rainy"] = (w["precipitation_sum"] >= 2.5).astype(int)
    w["heavy_rain"] = (w["precipitation_sum"] >= 64.5).astype(int)     # IMD "heavy rain" threshold
    w["fog"] = w["weather_code"].isin([45, 48]).astype(int)
    w["thunder"] = w["weather_code"].isin([95, 96, 99]).astype(int)
    w["gusty"] = (w["wind_gusts_10m_max"] >= 50).astype(int)
    m = w.groupby(["airport", "month"]).agg(
        rain_mm=("precipitation_sum", "sum"), rainy_days=("rainy", "sum"), heavy_rain_days=("heavy_rain", "sum"),
        fog_days=("fog", "sum"), thunder_days=("thunder", "sum"), gusty_days=("gusty", "sum"),
        max_temp=("temperature_2m_max", "mean"), min_temp=("temperature_2m_min", "mean")).reset_index()
    m.to_csv(DATA / "weather_monthly.csv", index=False)
    print("monthly rows", len(m))


if __name__ == "__main__" and "--fog" not in __import__("sys").argv:
    main()


def fog_risk():
    """Fog-risk hours per month: 00:00-09:59 local with relative humidity >= 97% and wind under 10 km/h.
    The archive's weather code rarely reports fog, so this standard proxy is used instead."""
    out = []
    for code, (_city, lat, lon) in AIRPORTS.items():
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               "&start_date=2021-01-01&end_date=2026-10-31&hourly=relative_humidity_2m,wind_speed_10m"
               "&timezone=Asia%2FKolkata")
        req = urllib.request.Request(url, headers={"User-Agent": "india-otp-study/1.0", "Accept-Encoding": "gzip"})
        for attempt in range(6):
            try:
                body = urllib.request.urlopen(req, timeout=180).read()
                if body[:2] == bytes([0x1F, 0x8B]):
                    body = gzip.decompress(body)
                h = pd.DataFrame(json.loads(body)["hourly"])
                break
            except Exception as e:
                print(code, "retry", attempt, repr(e)[:120])
                time.sleep(30 * (attempt + 1))
        h["time"] = pd.to_datetime(h["time"])
        fog = (h["relative_humidity_2m"] >= 97) & (h["wind_speed_10m"] < 10) & (h["time"].dt.hour < 10)
        g = fog.groupby(h["time"].dt.strftime("%Y-%m")).sum().rename("fog_risk_hours").reset_index()
        g.columns = ["month", "fog_risk_hours"]
        g["airport"] = code
        out.append(g)
        print(code, "fog-risk hours", int(g["fog_risk_hours"].sum()))
        time.sleep(2)
    f = pd.concat(out)
    m = pd.read_csv(DATA / "weather_monthly.csv").drop(columns=["fog_days", "fog_risk_hours"], errors="ignore")
    m.merge(f, on=["airport", "month"], how="left").to_csv(DATA / "weather_monthly.csv", index=False)


if __name__ == "__main__" and "--fog" in __import__("sys").argv:
    fog_risk()
