# India flight punctuality

Forecasts each airline's on-time share at 10 major Indian airports for a coming month, ranks the airlines, and shows
each airport's fog and monsoon pattern, the official track record and the causes of delay.

- **Data:** 993 airline-by-airport on-time figures read from 35 DGCA monthly domestic traffic reports (2021 to 2026),
  plus monthly airport weather from the Open-Meteo archive, including early-morning fog-risk hours.
- **Model:** LightGBM with small trees predicting the change from the last reported figure.
- **Test:** trained on March 2021 to March 2023; all 314 later figures held out. Average error 8.9 percentage points,
  against 9.2 for repeating the last report and 12.0 for the airline's average.
- **Limits:** monthly and airline-wide; India publishes no flight-by-flight delay data.

Run locally: `pip install -r requirements.txt` then `streamlit run app.py`. `train.py` and `fetch_weather.py` are
included for reference; the DGCA PDFs are not.
