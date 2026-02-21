# NYC Yellow Taxi Trip Analysis Dashboard

An end-to-end data pipeline that ingests, transforms, and analyzes NYC Yellow Taxi Trip data from January 2024, with an interactive Streamlit dashboard.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <repo-name>
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook
```bash
jupyter notebook assignment1.ipynb
```
Run all cells in order. The notebook will:
- Download the dataset automatically into `data/raw/`
- Clean and transform the data
- Run SQL analyses using DuckDB

### 5. Run the Streamlit Dashboard Locally
```bash
streamlit run app.py
```

## Deployed Dashboard

**Live Dashboard URL:** https://your-app-name.streamlit.app

## Project Structure

```
├── assignment1.ipynb   # Main analysis notebook (Parts 1–3)
├── app.py              # Streamlit dashboard application
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .gitignore          # Excludes data/ and other generated files
```

## Data Sources

- [NYC TLC Yellow Taxi Trip Records – January 2024](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet)
- [Taxi Zone Lookup Table](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv)

## Technologies Used

- **Python** (Pandas, Polars) – data ingestion and transformation
- **DuckDB** – in-process SQL analytics
- **Plotly** – interactive visualizations
- **Streamlit** – dashboard framework
