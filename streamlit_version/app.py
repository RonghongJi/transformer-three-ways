import pandas as pd
import streamlit as st
import torch

from shared.transformer_core import (
    ExperimentConfig,
    generate_series,
    run_experiment,
)


st.set_page_config(
    page_title="Streamlit Transformer",
    page_icon="📈",
    layout="wide",
)

st.title("时序 Transformer：Streamlit 版本")

st.caption(
    "页面、输入组件、图表和表格均由 Streamlit 的 Python API 生成。"
)


# ----------------------------
# 1. 页面参数
# ----------------------------
with st.sidebar:
    st.header("实验参数")

    series_length = st.slider(
        "时序长度",
        min_value=300,
        max_value=1200,
        value=600,
        step=100,
    )

    window_size = st.slider(
        "输入窗口",
        min_value=12,
        max_value=60,
        value=24,
        step=4,
    )

    epochs = st.slider(
        "训练轮数",
        min_value=5,
        max_value=30,
        value=10,
    )

    seed = st.number_input(
        "随机种子",
        min_value=0,
        max_value=9999,
        value=42,
    )

    use_mps = st.checkbox(
        "使用 Apple GPU",
        value=torch.backends.mps.is_available(),
        disabled=not torch.backends.mps.is_available(),
    )


# ----------------------------
# 2. 先展示模拟时序
# ----------------------------
_, preview_prices = generate_series(
    length=series_length,
    seed=int(seed),
)

preview_df = pd.DataFrame(
    {
        "时间点": range(len(preview_prices)),
        "模拟数值": preview_prices,
    }
).set_index("时间点")

st.subheader("模拟时序")

st.line_chart(
    preview_df,
    height=320,
)


# ----------------------------
# 3. 开始训练
# ----------------------------
if st.button(
    "开始训练",
    type="primary",
):
    config = ExperimentConfig(
        series_length=series_length,
        window_size=window_size,
        epochs=epochs,
        seed=int(seed),
        use_mps=use_mps,
        epoch_delay=0.1,
    )

    progress_bar = st.progress(0)
    status_area = st.empty()

    chart_column, log_column = st.columns([2, 1])

    loss_chart_area = chart_column.empty()
    log_area = log_column.empty()

    training_history = []
    logs = []

    def update_training(progress):
        training_history.append(progress)

        progress_bar.progress(
            progress["progress"]
        )

        status_area.write(
            f"Epoch "
            f"{progress['epoch']}/{progress['epochs']}｜"
            f"Loss：{progress['loss']:.4f}｜"
            f"训练准确率：{progress['accuracy']:.2%}"
        )

        history_df = pd.DataFrame(
            training_history
        ).set_index("epoch")

        loss_chart_area.line_chart(
            history_df[["loss"]],
            height=280,
        )

        logs.append(
            f"Epoch {progress['epoch']:02d} | "
            f"Loss {progress['loss']:.4f} | "
            f"Accuracy {progress['accuracy']:.2%}"
        )

        log_area.code(
            "\n".join(logs[-10:])
        )

    output = run_experiment(
        config=config,
        progress_callback=update_training,
    )

    st.success(
        f"训练与预测完成，运行设备："
        f"{output['device'].upper()}"
    )

    # ----------------------------
    # 4. 最终指标
    # ----------------------------
    metrics = output["metrics"]

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "测试准确率",
        f"{metrics['test_accuracy']:.2%}",
    )

    metric_columns[1].metric(
        "预测正确",
        metrics["correct_count"],
    )

    metric_columns[2].metric(
        "预测错误",
        metrics["incorrect_count"],
    )

    metric_columns[3].metric(
        "测试样本",
        metrics["test_samples"],
    )

    # ----------------------------
    # 5. 预测结果表格
    # ----------------------------
    st.subheader("逐条预测结果")

    result_df = pd.DataFrame(
        output["results"]
    )

    result_df = result_df.rename(
        columns={
            "time": "时间点",
            "current_value": "当前值",
            "next_value": "下一时刻真实值",
            "actual_direction": "实际方向",
            "predicted_direction": "预测方向",
            "confidence": "置信度",
            "correct": "是否正确",
        }
    )

    result_df["置信度"] = result_df[
        "置信度"
    ].map(
        lambda value: f"{value:.2%}"
    )

    result_df["是否正确"] = result_df[
        "是否正确"
    ].map(
        {
            True: "✅ 正确",
            False: "❌ 错误",
        }
    )

    def highlight_result(row):
        styles = [""] * len(row)

        result_column = list(
            result_df.columns
        ).index("是否正确")

        if row["是否正确"] == "✅ 正确":
            styles[result_column] = (
                "background-color: #d1fae5;"
                "color: #065f46;"
            )
        else:
            styles[result_column] = (
                "background-color: #fee2e2;"
                "color: #991b1b;"
            )

        return styles

    styled_result = result_df.style.apply(
        highlight_result,
        axis=1,
    )

    st.dataframe(
        styled_result,
        use_container_width=True,
        height=500,
    )

    # ----------------------------
    # 6. 正确/错误柱状图
    # ----------------------------
    st.subheader("预测结果汇总")

    summary_df = pd.DataFrame(
        {
            "数量": [
                metrics["correct_count"],
                metrics["incorrect_count"],
            ]
        },
        index=["预测正确", "预测错误"],
    )

    st.bar_chart(summary_df)