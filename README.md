# Market Panic Detector

A dark-mode dashboard that shows whether 50 large S&P 500 stocks are moving on their own news
(calm) or stampeding together (panic). It's written so you can use it without a quant background.

## Run
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/streamlit run app.py
```

## What's inside
- **Engine** (`panic_detector.py`): 5 years of yfinance adjusted closes (7 optional), daily log returns,
  a 60-day rolling correlation matrix, mean pairwise correlation and the top eigenvalue λ₁.
- **Status badge**: Calm · Dispersed below 0.25 mean correlation, Elevated Stress from 0.25 to 0.40,
  Panic · High Clustering at 0.40 and above.
- **Sidebar**: a Time Machine (date scrubber, day/week/month steps, Play/Pause, speed), a network
  threshold slider and layout picker, the guide drawer, the quick tour and advanced settings.
- **Presets**: Calm Market, COVID Crash (automatically switches to the 7-year history), Rate Shock 2022,
  Bank Scare 2023, Yen Unwind 2024 and Tariff Shock 2025. Each preset lands on the most clustered day of its episode.
- **Tabs**: Overview (network plus a timeline you can click to jump to a date), Correlation heatmap,
  Strongest links, and a plain-English Function & Metric Guide.
- **Help everywhere**: hover a `?` for a short definition, or click `ⓘ Explain` for more detail.
