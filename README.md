# Outage energy loss estimator

Rebuilds the wind at a wind park that lost communication from neighbouring parks,
and turns it into lost energy (kWh) with an uncertainty range.

## Run the app

```bash
uv sync
uv run streamlit run app/app.py
```

Then open http://localhost:8501. Switch on "Use the hackathon example data", or upload
CSV/Excel files (examples in `data/example_csv/`): one file per turbine, plus power curves.

## Layout

- `app/app.py` – the web page
- `src/app_core.py` – upload parsing, outage detection, analysis, email
- `src/pipeline.py` – the chosen model recipe (`outputs/search/final_recipe.json`)
- `src/fe.py`, `src/search_*.py`, `src/final_compare.py` – feature/model search
- `src/energy.py` – energy correction and uncertainty range; `src/density.py` – air density option
- `tests/` – `uv run pytest`
