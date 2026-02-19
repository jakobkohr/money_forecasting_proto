# money_forecasting_proto

Streamlit banking-assistant prototype that forecasts month-end cash risk from daily spending habits and upcoming obligations.

## Features
- Banking-style navigation: `Home`, `Insights`, `Transactions`, `Upcoming`
- Positive starting point with prevention alerts for upcoming obligations
- Salary-based bad purchase flagging (income class aware)
- Habit spend forecast + event-aware projected balances
- Demo calendar sandbox (connected status + add event UI)
- ML daily spend regression model loaded from local artifacts

## Data
Default demo uses:
- `data/synthetic_revolut_transactions.csv`
- Columns: `date, merchant, category, amount, is_recurring, balance_after, tags`

## Environment (micromamba / conda)
```bash
micromamba env create -f environment.yml
micromamba activate moneyml
```

(or)

```bash
conda env create -f environment.yml
conda activate moneyml
```

## Train model
```bash
python -m train.train_model
```

Artifacts created/updated:
- `models/daily_spend_model.joblib`
- `models/model_meta.json`

## Run app
```bash
streamlit run app.py
```

## Notes
- Forecasting is computed once per run state and reused across pages.
- Uploaded CSVs must include all required columns.
