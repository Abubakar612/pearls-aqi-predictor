const API_URL =
    "https://3i8a8njdia.execute-api.ap-south-1.amazonaws.com/predict";


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

    if (
        category ===
        "Unhealthy for Sensitive Groups"
    ) {
        return "sensitive";
    }

    if (category === "Unhealthy") {
        return "unhealthy";
    }

    return "hazardous";
}


function pollutantStatus(value, type) {

    if (value === null || value === undefined) {
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


function setPollutant(
    id,
    value,
    type,
    maxValue
) {

    const valueElement =
        document.getElementById(id);

    const barElement =
        document.getElementById(
            `${id}-bar`
        );

    const statusElement =
        document.getElementById(
            `${id}-status`
        );

    if (
        value === null ||
        value === undefined
    ) {
        valueElement.textContent = "--";
        statusElement.textContent = "--";
        return;
    }

    valueElement.textContent =
        Number(value).toFixed(2);

    statusElement.textContent =
        pollutantStatus(
            value,
            type
        );

    const percentage =
        Math.min(
            100,
            Math.max(
                3,
                (value / maxValue) * 100
            )
        );

    barElement.style.width =
        `${percentage}%`;
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

    const points =
        values.map(
            (value, index) => {

                const x =
                    xPositions[index];

                const y =
                    svgHeight -
                    (
                        (value / maxAQI)
                        * svgHeight
                    );

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

    document
        .getElementById("chart-line")
        .setAttribute(
            "points",
            pointString
        );


    const areaPath =
        `
        M ${points[0].x} ${points[0].y}
        ${points
            .slice(1)
            .map(
                point =>
                    `L ${point.x} ${point.y}`
            )
            .join(" ")}
        L ${points[points.length - 1].x}
          ${svgHeight}
        L ${points[0].x}
          ${svgHeight}
        Z
        `;

    document
        .getElementById(
            "chart-area-fill"
        )
        .setAttribute(
            "d",
            areaPath
        );


    const pointsGroup =
        document.getElementById(
            "chart-points"
        );

    pointsGroup.innerHTML = "";


    points.forEach(
        point => {

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
        }
    );
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


async function loadDashboard() {

    const refreshButton =
        document.getElementById(
            "refresh-button"
        );

    refreshButton.textContent =
        "Loading...";

    refreshButton.disabled = true;


    try {

        const response =
            await fetch(
                API_URL,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: "{}"
                }
            );


        if (!response.ok) {
            throw new Error(
                `API error ${response.status}`
            );
        }


        const data =
            await response.json();


        /* CURRENT AQI */

        document
            .getElementById(
                "current-aqi"
            )
            .textContent =
            Math.round(
                data.current_aqi
            );


        const category =
            data.forecast?.["24h"]?.category
            || "--";


        document
            .getElementById(
                "current-category"
            )
            .textContent =
            category;


        /* TIMESTAMP */

        document
            .getElementById(
                "data-timestamp"
            )
            .textContent =
            formatTimestamp(
                data.data_timestamp
            );


        /* FORECAST */

        const horizons = [
            "24h",
            "48h",
            "72h"
        ];


        horizons.forEach(
            horizon => {

                const forecast =
                    data.forecast[horizon];


                document
                    .getElementById(
                        `forecast-${horizon}`
                    )
                    .textContent =
                    Math.round(
                        forecast.aqi
                    );


                document
                    .getElementById(
                        `category-${horizon}`
                    )
                    .textContent =
                    forecast.category;
            }
        );


        /* POLLUTANTS */

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


        /* GRAPH */

        updateChart(
            [
                Number(
                    data.current_aqi
                ),

                Number(
                    data.forecast["24h"].aqi
                ),

                Number(
                    data.forecast["48h"].aqi
                ),

                Number(
                    data.forecast["72h"].aqi
                )
            ]
        );


    } catch (error) {

        console.error(
            "Dashboard error:",
            error
        );

        document
            .getElementById(
                "current-category"
            )
            .textContent =
            "Unable to load data";

    } finally {

        refreshButton.textContent =
            "↻ Refresh";

        refreshButton.disabled =
            false;
    }
}


document.addEventListener(
    "DOMContentLoaded",
    () => {

        loadDashboard();

        document
            .getElementById(
                "refresh-button"
            )
            .addEventListener(
                "click",
                loadDashboard
            );
    }
);