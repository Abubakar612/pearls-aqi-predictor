import numpy as np
import pandas as pd


# ============================================================
# EPA AQI BREAKPOINTS
# ============================================================

# PM2.5: µg/m³, 24-hour concentration
PM25_BREAKPOINTS = [
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
]


# PM10: µg/m³, 24-hour concentration
PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 500),
]


# CO: ppm, 8-hour concentration
CO_BREAKPOINTS = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]


# NO2: ppb, 1-hour concentration
NO2_BREAKPOINTS = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]


# SO2: ppb, 1-hour concentration
SO2_BREAKPOINTS = [
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
    (305, 604, 201, 300),
    (605, 804, 301, 400),
    (805, 1004, 401, 500),
]


# O3: ppm, 8-hour concentration
O3_BREAKPOINTS = [
    (0.000, 0.054, 0, 50),
    (0.055, 0.070, 51, 100),
    (0.071, 0.085, 101, 150),
    (0.086, 0.105, 151, 200),
    (0.106, 0.200, 201, 300),
]


def calculate_sub_index(
    concentration,
    breakpoints
):
    """
    Calculate AQI sub-index using linear interpolation.
    """

    if pd.isna(concentration):
        return np.nan

    for (
        concentration_low,
        concentration_high,
        aqi_low,
        aqi_high
    ) in breakpoints:

        if (
            concentration >= concentration_low
            and concentration <= concentration_high
        ):

            aqi = (
                (
                    aqi_high - aqi_low
                )
                /
                (
                    concentration_high
                    - concentration_low
                )
            ) * (
                concentration
                - concentration_low
            ) + aqi_low

            return round(aqi)

    return np.nan


# ============================================================
# UNIT CONVERSIONS
# ============================================================

def ugm3_to_ppm(
    concentration,
    molecular_weight
):
    """
    Convert µg/m³ to ppm.

    ppm = µg/m³ * 24.45 / (molecular weight * 1000)
    """

    return (
        concentration
        * 24.45
        / (molecular_weight * 1000)
    )


def ugm3_to_ppb(
    concentration,
    molecular_weight
):
    """
    Convert µg/m³ to ppb.

    ppb = µg/m³ * 24.45 / molecular weight
    """

    return (
        concentration
        * 24.45
        / molecular_weight
    )


# Molecular weights:
# CO  = 28.01
# NO2 = 46.01
# SO2 = 64.066
# O3  = 48.00


# ============================================================
# POLLUTANT AQI FUNCTIONS
# ============================================================

def calculate_pm25_aqi(value):
    return calculate_sub_index(
        value,
        PM25_BREAKPOINTS
    )


def calculate_pm10_aqi(value):
    return calculate_sub_index(
        value,
        PM10_BREAKPOINTS
    )


def calculate_co_aqi(value):
    """
    Open-Meteo CO is µg/m³.
    EPA breakpoint is ppm.
    """

    ppm = ugm3_to_ppm(
        value,
        molecular_weight=28.01
    )

    return calculate_sub_index(
        ppm,
        CO_BREAKPOINTS
    )


def calculate_no2_aqi(value):
    """
    Open-Meteo NO2 is µg/m³.
    EPA breakpoint is ppb.
    """

    ppb = ugm3_to_ppb(
        value,
        molecular_weight=46.01
    )

    return calculate_sub_index(
        ppb,
        NO2_BREAKPOINTS
    )


def calculate_so2_aqi(value):
    """
    Open-Meteo SO2 is µg/m³.
    EPA breakpoint is ppb.
    """

    ppb = ugm3_to_ppb(
        value,
        molecular_weight=64.066
    )

    return calculate_sub_index(
        ppb,
        SO2_BREAKPOINTS
    )


def calculate_o3_aqi(value):
    """
    Open-Meteo O3 is µg/m³.
    EPA breakpoint is ppm.
    """

    ppm = ugm3_to_ppm(
        value,
        molecular_weight=48.00
    )

    return calculate_sub_index(
        ppm,
        O3_BREAKPOINTS
    )


# ============================================================
# OVERALL AQI
# ============================================================

def calculate_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate pollutant-specific AQI values and overall AQI.
    """

    df = df.copy()

    print("Calculating pollutant AQI sub-indices...")

    df["pm25_aqi"] = df["pm2_5"].apply(
        calculate_pm25_aqi
    )

    df["pm10_aqi"] = df["pm10"].apply(
        calculate_pm10_aqi
    )

    df["co_aqi"] = df["co"].apply(
        calculate_co_aqi
    )

    df["no2_aqi"] = df["no2"].apply(
        calculate_no2_aqi
    )

    df["so2_aqi"] = df["so2"].apply(
        calculate_so2_aqi
    )

    df["o3_aqi"] = df["o3"].apply(
        calculate_o3_aqi
    )

    aqi_columns = [
        "pm25_aqi",
        "pm10_aqi",
        "co_aqi",
        "no2_aqi",
        "so2_aqi",
        "o3_aqi",
    ]

    # Overall AQI is the highest pollutant sub-index.
    df["target_aqi"] = df[
        aqi_columns
    ].max(axis=1)

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    input_file = (
        "data/processed/"
        "historical_cleaned.csv"
    )

    output_file = (
        "data/processed/"
        "historical_with_aqi.csv"
    )

    print("=" * 60)
    print("AQI CALCULATION")
    print("=" * 60)

    print(
        f"\nLoading:\n{input_file}"
    )

    df = pd.read_csv(
        input_file
    )

    print(
        f"\nInput shape: {df.shape}"
    )

    df = calculate_aqi(df)

    print("\nAQI calculation completed.")

    print("\nAQI statistics:")
    print(
        df["target_aqi"].describe()
    )

    print("\nFirst 10 observations:")

    print(
        df[
            [
                "timestamp",
                "pm2_5",
                "pm10",
                "co",
                "no2",
                "so2",
                "o3",
                "pm25_aqi",
                "pm10_aqi",
                "co_aqi",
                "no2_aqi",
                "so2_aqi",
                "o3_aqi",
                "target_aqi",
            ]
        ]
        .head(10)
        .to_string()
    )

    print("\nMissing target AQI:")

    print(
        df["target_aqi"]
        .isnull()
        .sum()
    )

    print("\nAQI category distribution:")

    print(
        pd.cut(
            df["target_aqi"],
            bins=[
                -1,
                50,
                100,
                150,
                200,
                300,
                500
            ],
            labels=[
                "Good",
                "Moderate",
                "Unhealthy for Sensitive Groups",
                "Unhealthy",
                "Very Unhealthy",
                "Hazardous",
            ]
        )
        .value_counts()
        .sort_index()
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved to:\n{output_file}"
    )