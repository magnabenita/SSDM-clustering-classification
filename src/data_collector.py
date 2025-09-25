# src/data_collector.py
import os
import requests
import pandas as pd
from datetime import datetime, timedelta

USGS_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

def fetch_earthquake_data(starttime: str, endtime: str, minmagnitude: float = 0.0,
                          limit: int = 20000, out_csv: str = "data/earthquakes.csv") -> pd.DataFrame:
    """
    Fetch earthquake events from the USGS API and save as CSV.

    Args:
        starttime (str): Start date (YYYY-MM-DD)
        endtime (str): End date (YYYY-MM-DD)
        minmagnitude (float): Minimum magnitude filter
        limit (int): Max number of events to fetch
        out_csv (str): Path to save the CSV

    Returns:
        DataFrame with earthquake data
    """
    params = {
        "format": "geojson",
        "starttime": starttime,
        "endtime": endtime,
        "minmagnitude": minmagnitude,
        "limit": limit
    }

    print(f"Fetching data from {starttime} to {endtime} (min magnitude: {minmagnitude}) ...")
    response = requests.get(USGS_API_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    features = data.get("features", [])
    rows = []
    for f in features:
        props = f["properties"]
        coords = f.get("geometry", {}).get("coordinates", [None, None, None])
        rows.append({
            "id": f.get("id"),
            "time": pd.to_datetime(props["time"], unit="ms"),
            "latitude": coords[1],
            "longitude": coords[0],
            "depth": coords[2],
            "magnitude": props.get("mag"),
            "place": props.get("place"),
            "type": props.get("type"),
            "status": props.get("status"),
        })

    df = pd.DataFrame(rows)

    # ensure output directory exists
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} records to {out_csv}")
    return df


if __name__ == "__main__":
    # Example: fetch last 30 days of data
    end = datetime.utcnow()
    start = end - timedelta(days=30)
    fetch_earthquake_data(start.strftime("%Y-%m-%d"),
                          end.strftime("%Y-%m-%d"),
                          minmagnitude=2.5,
                          out_csv="data/earthquakes_last30days.csv")
