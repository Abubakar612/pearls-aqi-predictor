import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

REALTIME_DIR = (
    BASE_DIR
    / "data"
    / "realtime"
)

REALTIME_FILE = (
    REALTIME_DIR
    / "lahore_hourly.csv"
)


# ============================================================
# COLUMNS
# ============================================================

COLUMNS = [
    "timestamp",
    "temperature",
    "humidity",
    "dew_point",
    "feels_like",
    "pressure",
    "surface_pressure",
    "clouds",
    "wind_speed",
    "wind_direction",
    "wind_gust",
    "precipitation",
    "pm10",
    "pm2_5",
    "co",
    "no2",
    "so2",
    "o3",
]


# ============================================================
# INITIALIZE STORE
# ============================================================

def initialize_store():

    REALTIME_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not REALTIME_FILE.exists():

        df = pd.DataFrame(
            columns=COLUMNS
        )

        df.to_csv(
            REALTIME_FILE,
            index=False
        )


# ============================================================
# LOAD STORE
# ============================================================

def load_store():

    initialize_store()

    df = pd.read_csv(
        REALTIME_FILE,
        parse_dates=["timestamp"]
    )

    if df.empty:
        return df

    df = df.sort_values(
        "timestamp"
    )

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    return df.reset_index(drop=True)


# ============================================================
# SAVE OBSERVATION
# ============================================================

def save_observation(observation):

    initialize_store()

    df = load_store()

    new_row = pd.DataFrame(
        [observation]
    )

    new_row["timestamp"] = pd.to_datetime(
        new_row["timestamp"],
        utc=True
    )

    df = pd.concat(
        [
            df,
            new_row
        ],
        ignore_index=True
    )

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    df = df.sort_values(
        "timestamp"
    )

    df.to_csv(
        REALTIME_FILE,
        index=False
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# ADD HISTORICAL DATA
# ============================================================

def seed_from_historical():

    historical_file = (
        BASE_DIR
        / "data"
        / "processed"
        / "historical_cleaned.csv"
    )

    if not historical_file.exists():

        raise FileNotFoundError(
            f"Historical dataset not found: "
            f"{historical_file}"
        )

    historical = pd.read_csv(
        historical_file,
        parse_dates=["timestamp"]
    )

    historical = historical[
        COLUMNS
    ].copy()

    historical["timestamp"] = pd.to_datetime(
        historical["timestamp"],
        utc=True
    )

    historical = historical.drop_duplicates(
        subset=["timestamp"]
    )

    historical = historical.sort_values(
        "timestamp"
    )

    REALTIME_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    historical.to_csv(
        REALTIME_FILE,
        index=False
    )

    print(
        f"Realtime store seeded with "
        f"{len(historical)} observations."
    )

    print(
        f"Saved to:\n{REALTIME_FILE}"
    )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("REAL-TIME DATA STORE")
    print("=" * 60)

    seed_from_historical()

    df = load_store()

    print(
        f"\nRows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"\nDate range:"
        f"\n{df['timestamp'].min()}"
        f"\n→ {df['timestamp'].max()}"
    )

    print(
        "\nLatest observation:"
    )

    print(
        df.tail(1).to_string(
            index=False
        )
    )