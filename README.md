# DSFinalProject — Setup & Run Instructions

## 1. Install Dependencies
Before running any scripts, install the required Python packages:
```
pip install -r requirements.txt
```

## 2. Run Scripts in Order
The scripts must be run in the following order:

**Step 1 — Build the data pipeline (run this first):**
```
python "code files/pipeline.py"
```

**Step 2 — Run temporal analysis:**
```
python "code files/temporal_analysis.py"
```

**Step 3 — Run models:**
```
python "code files/models.py"
```

## Notes
- `pipeline.py` must run before the other two scripts — it generates the data files they depend on.
- `temporal_analysis.py` and `models.py` can be run in either order after `pipeline.py`.
- All output CSVs will be saved to the `csvs/` folder.
- All output graphs will be saved to the `graphs/` folder.
