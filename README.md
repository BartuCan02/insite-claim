# InSite Claim

**Know what a wind park lost while it was offline.**

When a wind park loses its communication link, its wind measurements disappear too, but the insurer still needs the
lost energy in kWh. InSite Claim rebuilds the missing 10-minute wind from neighbouring parks, turns it into lost energy
with the turbines' power curves, puts an honest uncertainty range on it, and prepares the insurance claim: an email and a
report that cite the clauses of your policy.

Built for the BayWa r.e. data hackathon (TUM, October 2026).

## Run the demo

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. install uv (skip if you have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. get the code
git clone https://github.com/BartuCan02/insite-claim.git
cd insite-claim

# 3. install the dependencies and start the app
uv sync
uv run streamlit run app/app.py
```

The app opens at **http://localhost:8501**. The first analysis takes about 10 seconds; after that results are cached.

### What to click

1. **Overview**: the hackathon dataset is loaded by default ("Example dataset"). The outage of turbines T1–T3 on
   1–3 September 2026 is found automatically. The portfolio lists wind parks T, H and O plus 20 BayWa r.e. parks
   (public information, no data connected).
2. **Analysis**: rebuilt wind speed, lost energy per turbine, how the loss built up, and the loss by wind-speed band.
   Drag the outage window or change the affected turbines; open **Settings** for the training window, confidence
   level and the air-density option.
3. **Accuracy**: the judging metrics (wind RMSE/MAE, power RMSE/MAE, summed energy) on 4–9 September, where measured
   data exists, plus the back-test and the validation on 84 simulated outages.
4. **Method**: the five steps, the automatic reference selection and the correlation between turbines.
5. **Claim report**: pick the **sample contract** (or upload your own PDF/Word/text policy). The app reads its clauses
   (waiting period, rate per kWh, limit, notification deadline, ...), computes the claimable energy and amount, and
   writes an insurance email that cites each clause. Download the claim report (HTML, print to PDF), the 10-minute time
   series (CSV) and a JSON summary.

### Use your own data

Switch **Data source** on the Overview page to **Upload files**:

- **Wind data**: one CSV or Excel file per turbine with `turbine_name`, `timestamp`, `wind_speed`
  (optional: `wind_direction`, `ambient_temperature`, `air_pressure`, `wind_speed_stddev`, `test_set`).
  Turbine names start with a park letter (`T1`, `H3`, `O2`); turbines of other parks become reference candidates.
- **Power curves**: `turbine_name`, `wind_speed` (m/s), `power` (kW).

Ready-made CSV copies of the hackathon data are in `data/example_csv/`.

## How it works

1. **Find the outage** from the missing data (longest gap shared by a park, adjustable).
2. **Pick references automatically**: the 5 turbines in other parks whose wind followed the target most closely over
   the 30 days before the outage.
3. **Rebuild the wind**: 70 % ridge regression + 30 % LightGBM, trained only on those 30 days, using the references'
   wind now and 10/30/60 minutes earlier and later, a direction-dependent travel time between parks, wind direction,
   turbulence and how much the references agree.
4. **Turn wind into energy**: the power curve is averaged over the typical spread of real wind around the estimate.
   Reading it at a single point under-counts energy by about 10 %.
5. **Put a range on it**: calibrated on 84 simulated 3-day outages (±26.5 % per turbine, ±17.1 % per park at 90 %).

The model was chosen with a staged grid search (65,000+ trained models, each scored on 84 simulated outages); the
4–9 September test window stayed untouched until the end.

## Results

| | wind RMSE | wind MAE | power RMSE | power MAE | summed energy |
|---|---|---|---|---|---|
| Test window 4–9 Sep (all turbines) | 1.36 m/s | 1.03 m/s | 837 kW | 602 kW | +3.1 % |
| 84 simulated outages, simple baseline | 1.19 m/s | | 677 kW | | −13.7 % |
| 84 simulated outages, InSite Claim | 1.11 m/s | | 615 kW | | −2.3 % |

**Estimated loss for the outage on 1–3 September 2026: 318,440 kWh** (T1 99,189 · T2 122,575 · T3 96,676),
90 % range about 264–373 MWh.

## Project layout

| Path | What it holds |
|---|---|
| `app/app.py` | the Streamlit app (five pages) |
| `src/app_core.py` | file parsing, outage detection, analysis runner, email text |
| `src/pipeline.py`, `src/recipe.json` | the model recipe |
| `src/core.py`, `src/fe.py`, `src/advect.py` | outage detection, features, travel-time feature |
| `src/energy.py`, `src/density.py` | energy correction, uncertainty range, air density |
| `src/claim.py` | contract parsing, claim amount, clause-citing email |
| `src/search_*.py`, `src/final_compare.py`, `src/step2_eval.py` | the model search and validation runs |
| `data/raw/` | hackathon data (Excel) · `data/example_csv/` CSV copies · `data/sample_contract/` sample policy |
| `tests/` | `uv run pytest` |

## Notes

- The sample insurance contract uses **fictional parties** and is for demonstration only.
- Figures are estimates based on reference-park measurements, not a confirmation of insurance coverage.
- The app only listens on your own machine (`localhost`).
