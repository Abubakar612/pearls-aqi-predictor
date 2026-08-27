const API_URL = "/api/forecast";


function categoryClass(category) {

    if (!category) {
        return "";
    }

    if (category === "Good") {
        return "good";
    }

    if (category === "Moderate") {
        return "moderate";
    }

    if (category === "Unhealthy for Sensitive Groups") {
        return "sensitive";
    }

    if (category === "Unhealthy") {
        return "unhealthy";
    }

    if (category === "Very Unhealthy") {
        return "very";
    }

    return "hazardous";
}


function pollutantStatus(value, type) {

    if (value === null || value === undefined || isNaN(value)) {
        return "--";
    }

    if (type === "pm25") {

        if (value <= 35) return "Low";
        if (value <= 55) return "Moderate";

        return "High";
    }

    if (type === "pm10") {

        if (value <= 50) return "Low";
        if (value <= 100) return "Moderate";

        return "High";
    }

    if (type === "no2") {

        if (value <= 40) return "Low";
        if (value <= 100) return "Moderate";

        return "High";
    }

    if (type === "o3") {

        if (value <= 50) return "Low";
        if (value <= 100) return "Moderate";

        return "High";
    }

    if (type === "co") {

        if (value <= 400) return "Low";
        if (value <= 1000) return "Moderate";

        return "High";
    }

    return "--";
}


function setPollutant(id, value, type, maxValue) {

    const valueElement = document.getElementById(id);

    const barElement = document.getElementById(
        `${id}-bar`
    );

    const statusElement = document.getElementById(
        `${id}-status`
    );

    if (
        value === null ||
        value === undefined ||
        isNaN(value)
    ) {

        valueElement.textContent = "--";
        statusElement.textContent = "--";

        if (barElement) {
            barElement.style.width = "0%";
        }

        return;
    }

    const numericValue = Number(value);

    valueElement.textContent =
        numericValue.toFixed(2);

    statusElement.textContent =
        pollutantStatus(
            numericValue,
            type
        );

    const percentage =
        Math.min(
            100,
            Math.max(
                3,
                (numericValue / maxValue) * 100
            )
        );

    if (barElement) {
        barElement.style.width =
            `${percentage}%`;
    }
}


function updateCategory(elementId, category) {

    const element =
        document.getElementById(elementId);

    if (!element) {
        return;
    }

    element.textContent =
        category || "--";

    element.classList.remove(
        "good",
        "moderate",
        "sensitive",
        "unhealthy",
        "very",
        "hazardous"
    );

    const cssClass =
        categoryClass(category);

    if (cssClass) {
        element.classList.add(cssClass);
    }
}


function updateChart(values) {

    const svgWidth = 1000;
    const svgHeight = 300;

    const maxAQI = 200;

    const xPositions = [
        60,
        350,
        640,
        940
    ];

    const points = values.map(
        (value, index) => {

            const safeValue =
                Math.max(
                    0,
                    Number(value) || 0
                );

            const x =
                xPositions[index];

            const y =
                svgHeight -
                (
                    Math.min(
                        safeValue,
                        maxAQI
                    ) / maxAQI
                ) * svgHeight;

            return {
                x,
                y
            };
        }
    );

    const pointString =
        points
            .map(
                point =>
                    `${point.x},${point.y}`
            )
            .join(" ");

    const chartLine =
        document.getElementById(
            "chart-line"
        );

    if (chartLine) {

        chartLine.setAttribute(
            "points",
            pointString
        );
    }


    if (points.length > 0) {

        const areaPath =
            `M ${points[0].x} ${points[0].y}
             ${points
                .slice(1)
                .map(
                    point =>
                        `L ${point.x} ${point.y}`
                )
                .join(" ")}
             L ${points[points.length - 1].x} ${svgHeight}
             L ${points[0].x} ${svgHeight}
             Z`;

        const area =
            document.getElementById(
                "chart-area-fill"
            );

        if (area) {

            area.setAttribute(
                "d",
                areaPath
            );
        }
    }


    const pointsGroup =
        document.getElementById(
            "chart-points"
        );

    if (!pointsGroup) {
        return;
    }

    pointsGroup.innerHTML = "";


    points.forEach(point => {

        const circle =
            document.createElementNS(
                "http://www.w3.org/2000/svg",
                "circle"
            );

        circle.setAttribute(
            "cx",
            point.x
        );

        circle.setAttribute(
            "cy",
            point.y
        );

        circle.setAttribute(
            "r",
            "7"
        );

        circle.setAttribute(
            "class",
            "chart-point"
        );

        pointsGroup.appendChild(
            circle
        );
    });
}


function formatTimestamp(timestamp) {

    if (!timestamp) {
        return "--";
    }

    const date =
        new Date(timestamp);

    if (isNaN(date.getTime())) {
        return timestamp;
    }

    return date.toLocaleString(
        "en-GB",
        {
            day: "2-digit",
            month: "short",
            year: "numeric",
            hour: "2-digit",
            minute: "2-digit"
        }
    );
}


function setLoadingState() {

    const currentAQI =
        document.getElementById(
            "current-aqi"
        );

    if (currentAQI) {
        currentAQI.textContent = "--";
    }


    const currentCategory =
        document.getElementById(
            "current-category"
        );

    if (currentCategory) {
        currentCategory.textContent =
            "Loading";
    }
}


function showError(message) {

    console.error(
        "Dashboard error:",
        message
    );

    const category =
        document.getElementById(
            "current-category"
        );

    if (category) {
        category.textContent =
            "Unable to load data";
    }

    const timestamp =
        document.getElementById(
            "data-timestamp"
        );

    if (timestamp) {
        timestamp.textContent =
            "--";
    }
}


async function loadDashboard() {

    const refreshButton =
        document.getElementById(
            "refresh-button"
        );

    if (refreshButton) {

        refreshButton.textContent =
            "Loading...";

        refreshButton.disabled =
            true;
    }


    try {

        setLoadingState();


        /*
         * The local FastAPI endpoint reads
         * the latest forecast directly from S3.
         */
        const response =
            await fetch(
                API_URL,
                {
                    method: "GET",

                    headers: {
                        "Accept":
                            "application/json"
                    },

                    cache: "no-store"
                }
            );


        if (!response.ok) {

            throw new Error(
                `API error ${response.status}`
            );
        }


        const data =
            await response.json();


        console.log(
            "Forecast data:",
            data
        );


        /*
         * CURRENT AQI
         */

        const currentAQI =
            Number(
                data.current_aqi
            );

        const currentAQIElement =
            document.getElementById(
                "current-aqi"
            );

        if (
            currentAQIElement &&
            !isNaN(currentAQI)
        ) {

            currentAQIElement.textContent =
                Math.round(
                    currentAQI
                );
        }


        /*
         * CURRENT CATEGORY
         *
         * Use the current AQI itself,
         * rather than the 24h forecast category.
         */

        let currentCategory =
            "Unknown";


        if (currentAQI <= 50) {

            currentCategory = "Good";

        } else if (currentAQI <= 100) {

            currentCategory = "Moderate";

        } else if (currentAQI <= 150) {

            currentCategory =
                "Unhealthy for Sensitive Groups";

        } else if (currentAQI <= 200) {

            currentCategory =
                "Unhealthy";

        } else if (currentAQI <= 300) {

            currentCategory =
                "Very Unhealthy";

        } else {

            currentCategory =
                "Hazardous";
        }


        updateCategory(
            "current-category",
            currentCategory
        );


        /*
         * TIMESTAMP
         */

        const timestampElement =
            document.getElementById(
                "data-timestamp"
            );

        if (timestampElement) {

            timestampElement.textContent =
                formatTimestamp(
                    data.data_timestamp
                );
        }


        /*
         * FORECASTS
         */

        const horizons = [
            "24h",
            "48h",
            "72h"
        ];


        horizons.forEach(horizon => {

            const forecast =
                data.forecast?.[horizon];


            if (!forecast) {
                return;
            }


            const forecastElement =
                document.getElementById(
                    `forecast-${horizon}`
                );


            if (forecastElement) {

                forecastElement.textContent =
                    Number(
                        forecast.aqi
                    ).toFixed(1);
            }


            updateCategory(
                `category-${horizon}`,
                forecast.category
            );
        });


        /*
         * POLLUTANTS
         */

        if (data.pollutants) {

            setPollutant(
                "pm25",
                data.pollutants.pm25,
                "pm25",
                100
            );


            setPollutant(
                "pm10",
                data.pollutants.pm10,
                "pm10",
                150
            );


            setPollutant(
                "no2",
                data.pollutants.no2,
                "no2",
                100
            );


            setPollutant(
                "o3",
                data.pollutants.o3,
                "o3",
                100
            );


            setPollutant(
                "co",
                data.pollutants.co,
                "co",
                1000
            );
        }


        /*
         * GRAPH
         */

        updateChart(
            [
                currentAQI,

                Number(
                    data.forecast?.["24h"]?.aqi
                ),

                Number(
                    data.forecast?.["48h"]?.aqi
                ),

                Number(
                    data.forecast?.["72h"]?.aqi
                )
            ]
        );


    } catch (error) {

        showError(
            error.message
        );

    } finally {

        if (refreshButton) {

            refreshButton.textContent =
                "↻ Refresh";

            refreshButton.disabled =
                false;
        }
    }
}


/*
 * INITIAL LOAD
 */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        loadDashboard();


        const refreshButton =
            document.getElementById(
                "refresh-button"
            );


        if (refreshButton) {

            refreshButton.addEventListener(
                "click",
                loadDashboard
            );
        }
    }
);