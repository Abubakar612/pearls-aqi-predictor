import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "forecast_dataset.csv"
)

EDA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "eda"
)


def load_data():

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


def create_aqi_categories(df):

    bins = [
        -np.inf,
        50,
        100,
        150,
        200,
        300,
        np.inf
    ]

    labels = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous"
    ]

    df["aqi_category"] = pd.cut(
        df["target_aqi"],
        bins=bins,
        labels=labels
    )

    return df


def print_basic_statistics(df):

    print("\n" + "=" * 60)
    print("BASIC DATASET INFORMATION")
    print("=" * 60)

    print(
        f"\nRows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"\nDate range:"
    )

    print(
        df["timestamp"].min(),
        "→",
        df["timestamp"].max()
    )

    print(
        "\nMissing values:"
    )

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("None")
    else:
        print(missing)


def print_aqi_statistics(df):

    print("\n" + "=" * 60)
    print("AQI STATISTICS")
    print("=" * 60)

    print(
        df["target_aqi"].describe()
    )

    print(
        "\nAQI category distribution:"
    )

    print(
        df["aqi_category"]
        .value_counts()
        .sort_index()
    )


def print_pollutant_statistics(df):

    pollutants = [
        "pm2_5",
        "pm10",
        "co",
        "no2",
        "so2",
        "o3"
    ]

    print("\n" + "=" * 60)
    print("POLLUTANT STATISTICS")
    print("=" * 60)

    print(
        df[pollutants].describe()
    )


def print_correlations(df):

    columns = [
        "target_aqi",
        "pm2_5",
        "pm10",
        "co",
        "no2",
        "so2",
        "o3",
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "clouds"
    ]

    print("\n" + "=" * 60)
    print("CORRELATIONS WITH AQI")
    print("=" * 60)

    correlations = (
        df[columns]
        .corr()["target_aqi"]
        .sort_values(
            ascending=False
        )
    )

    print(
        correlations
    )

    return correlations


def create_aqi_timeseries(df):

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        df["timestamp"],
        df["target_aqi"]
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "AQI"
    )

    plt.title(
        "Lahore AQI Over Time"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        EDA_DIR
        / "aqi_timeseries.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    plt.close()

    print(
        f"\nSaved: {output}"
    )


def create_pollutant_plots(df):

    pollutants = [
        "pm2_5",
        "pm10",
        "co",
        "no2",
        "so2",
        "o3"
    ]

    for pollutant in pollutants:

        plt.figure(
            figsize=(12, 5)
        )

        plt.plot(
            df["timestamp"],
            df[pollutant]
        )

        plt.xlabel(
            "Time"
        )

        plt.ylabel(
            pollutant
        )

        plt.title(
            f"{pollutant.upper()} Over Time"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        output = (
            EDA_DIR
            / f"{pollutant}_timeseries.png"
        )

        plt.savefig(
            output,
            dpi=150
        )

        plt.close()

        print(
            f"Saved: {output}"
        )


def create_correlation_heatmap(df):

    columns = [
        "target_aqi",
        "pm2_5",
        "pm10",
        "co",
        "no2",
        "so2",
        "o3",
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "clouds"
    ]

    correlation = (
        df[columns]
        .corr()
    )

    plt.figure(
        figsize=(12, 10)
    )

    plt.imshow(
        correlation,
        aspect="auto"
    )

    plt.colorbar(
        label="Correlation"
    )

    plt.xticks(
        range(len(columns)),
        columns,
        rotation=90
    )

    plt.yticks(
        range(len(columns)),
        columns
    )

    plt.title(
        "Feature Correlation Matrix"
    )

    plt.tight_layout()

    output = (
        EDA_DIR
        / "correlation_heatmap.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


def create_target_comparison(df):

    targets = [
        "target_aqi_24h",
        "target_aqi_48h",
        "target_aqi_72h"
    ]

    statistics = df[
        targets
    ].describe().T

    print(
        "\n" + "=" * 60
    )

    print(
        "FORECAST TARGET STATISTICS"
    )

    print(
        "=" * 60
    )

    print(
        statistics[
            [
                "mean",
                "std",
                "min",
                "max"
            ]
        ]
    )

    plt.figure(
        figsize=(12, 6)
    )

    for target in targets:

        plt.plot(
            df["timestamp"],
            df[target],
            label=target
        )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "AQI"
    )

    plt.title(
        "24h / 48h / 72h AQI Targets"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        EDA_DIR
        / "forecast_targets.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


def main():

    EDA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Loading forecast dataset..."
    )

    df = load_data()

    df = create_aqi_categories(
        df
    )

    print_basic_statistics(
        df
    )

    print_aqi_statistics(
        df
    )

    print_pollutant_statistics(
        df
    )

    print_correlations(
        df
    )

    create_aqi_timeseries(
        df
    )

    create_pollutant_plots(
        df
    )

    create_correlation_heatmap(
        df
    )

    create_target_comparison(
        df
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "EDA COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        f"\nPlots saved to:\n{EDA_DIR}"
    )


if __name__ == "__main__":
    main()