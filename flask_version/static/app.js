let seriesChart = null;
let lossChart = null;
let currentJobId = null;


function getParameters() {
    return {
        series_length: Number(
            document.querySelector(
                "#series-length"
            ).value
        ),
        window_size: Number(
            document.querySelector(
                "#window-size"
            ).value
        ),
        epochs: Number(
            document.querySelector(
                "#epochs"
            ).value
        ),
        seed: Number(
            document.querySelector(
                "#seed"
            ).value
        ),
        use_mps: document.querySelector(
            "#use-mps"
        ).checked,
    };
}


async function loadPreview() {
    const parameters = getParameters();

    const query = new URLSearchParams({
        series_length: parameters.series_length,
        seed: parameters.seed,
    });

    const response = await fetch(
        `/api/preview?${query.toString()}`
    );

    if (!response.ok) {
        throw new Error(
            `预览请求失败：${response.status}`
        );
    }

    const data = await response.json();

    renderSeriesChart(data.series);
}


function renderSeriesChart(series) {
    const canvas = document.querySelector(
        "#series-chart"
    );

    if (seriesChart !== null) {
        seriesChart.destroy();
    }

    seriesChart = new Chart(
        canvas,
        {
            type: "line",
            data: {
                labels: series.map(
                    item => item.time
                ),
                datasets: [
                    {
                        label: "模拟数值",
                        data: series.map(
                            item => item.value
                        ),
                        borderColor: "#2563eb",
                        backgroundColor:
                            "rgba(37, 99, 235, 0.10)",
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: true,
                        tension: 0.15,
                    },
                ],
            },
            options: {
                responsive: true,
                animation: false,
                interaction: {
                    intersect: false,
                    mode: "index",
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "时间点",
                        },
                    },
                    y: {
                        title: {
                            display: true,
                            text: "模拟数值",
                        },
                    },
                },
            },
        }
    );
}


function renderLossChart(history) {
    const canvas = document.querySelector(
        "#loss-chart"
    );

    const labels = history.map(
        item => item.epoch
    );

    const losses = history.map(
        item => item.loss
    );

    if (lossChart === null) {
        lossChart = new Chart(
            canvas,
            {
                type: "line",
                data: {
                    labels,
                    datasets: [
                        {
                            label: "训练损失",
                            data: losses,
                            borderColor: "#dc2626",
                            backgroundColor:
                                "rgba(220, 38, 38, 0.10)",
                            borderWidth: 2,
                            tension: 0.2,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    animation: false,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: "Epoch",
                            },
                        },
                        y: {
                            title: {
                                display: true,
                                text: "Loss",
                            },
                        },
                    },
                },
            }
        );

        return;
    }

    lossChart.data.labels = labels;
    lossChart.data.datasets[0].data = losses;
    lossChart.update("none");
}


function updateTrainingStatus(job) {
    const progressBar = document.querySelector(
        "#progress-bar"
    );

    const statusText = document.querySelector(
        "#status-text"
    );

    const logArea = document.querySelector(
        "#log-area"
    );

    progressBar.style.width =
        `${job.progress}%`;

    if (job.epoch === 0) {
        statusText.textContent =
            "训练任务已创建，等待第一个 Epoch……";
    } else {
        statusText.textContent =
            `Epoch ${job.epoch}/${job.epochs}｜` +
            `Loss：${job.loss.toFixed(4)}｜` +
            `训练准确率：` +
            `${(job.accuracy * 100).toFixed(2)}%`;
    }

    const logs = job.history.map(
        item =>
            `Epoch ${String(item.epoch).padStart(2, "0")} | ` +
            `Loss ${item.loss.toFixed(4)} | ` +
            `Accuracy ${(item.accuracy * 100).toFixed(2)}%`
    );

    logArea.textContent =
        logs.slice(-10).join("\n");

    renderLossChart(job.history);
}


function renderResults(output) {
    const resultSection = document.querySelector(
        "#result-section"
    );

    resultSection.classList.remove("hidden");

    const metrics = output.metrics;

    document.querySelector(
        "#test-accuracy"
    ).textContent =
        `${(metrics.test_accuracy * 100).toFixed(2)}%`;

    document.querySelector(
        "#correct-count"
    ).textContent =
        metrics.correct_count;

    document.querySelector(
        "#incorrect-count"
    ).textContent =
        metrics.incorrect_count;

    document.querySelector(
        "#test-samples"
    ).textContent =
        metrics.test_samples;

    const tableBody = document.querySelector(
        "#result-table-body"
    );

    tableBody.innerHTML = "";

    for (const row of output.results) {
        const tableRow =
            document.createElement("tr");

        tableRow.className = row.correct
            ? "correct-row"
            : "incorrect-row";

        const resultLabel = row.correct
            ? "✅ 正确"
            : "❌ 错误";

        const resultClass = row.correct
            ? "correct-label"
            : "incorrect-label";

        tableRow.innerHTML = `
            <td>${row.time}</td>
            <td>${row.current_value}</td>
            <td>${row.next_value}</td>
            <td>${row.actual_direction}</td>
            <td>${row.predicted_direction}</td>
            <td>
                ${(row.confidence * 100).toFixed(2)}%
            </td>
            <td class="${resultClass}">
                ${resultLabel}
            </td>
        `;

        tableBody.appendChild(tableRow);
    }

    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start",
    });
}


async function pollTrainingJob(jobId) {
    const response = await fetch(
        `/api/status/${jobId}`
    );

    if (!response.ok) {
        throw new Error(
            `状态请求失败：${response.status}`
        );
    }

    const job = await response.json();

    updateTrainingStatus(job);

    if (job.status === "completed") {
        renderResults(job.output);

        document.querySelector(
            "#train-button"
        ).disabled = false;

        document.querySelector(
            "#status-text"
        ).textContent +=
            `｜训练完成，设备：` +
            `${job.output.device.toUpperCase()}`;

        return;
    }

    if (job.status === "failed") {
        throw new Error(
            job.error || "训练失败"
        );
    }

    // 500毫秒后再次查询状态
    window.setTimeout(
        () => {
            pollTrainingJob(jobId).catch(
                handleError
            );
        },
        500
    );
}


async function startTraining() {
    const trainButton = document.querySelector(
        "#train-button"
    );

    trainButton.disabled = true;

    document.querySelector(
        "#result-section"
    ).classList.add("hidden");

    document.querySelector(
        "#progress-bar"
    ).style.width = "0%";

    document.querySelector(
        "#status-text"
    ).textContent = "正在创建训练任务……";

    document.querySelector(
        "#log-area"
    ).textContent = "等待第一个 Epoch……";

    if (lossChart !== null) {
        lossChart.destroy();
        lossChart = null;
    }

    const response = await fetch(
        "/api/train",
        {
            method: "POST",
            headers: {
                "Content-Type":
                    "application/json",
            },
            body: JSON.stringify(
                getParameters()
            ),
        }
    );

    if (!response.ok) {
        throw new Error(
            `创建任务失败：${response.status}`
        );
    }

    const job = await response.json();

    currentJobId = job.job_id;

    await pollTrainingJob(currentJobId);
}


function handleError(error) {
    console.error(error);

    document.querySelector(
        "#status-text"
    ).textContent =
        `发生错误：${error.message}`;

    document.querySelector(
        "#train-button"
    ).disabled = false;
}


document.addEventListener(
    "DOMContentLoaded",
    () => {
        document.querySelector(
            "#preview-button"
        ).addEventListener(
            "click",
            () => {
                loadPreview().catch(handleError);
            }
        );

        document.querySelector(
            "#train-button"
        ).addEventListener(
            "click",
            () => {
                startTraining().catch(handleError);
            }
        );

        loadPreview().catch(handleError);
    }
);