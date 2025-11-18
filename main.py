def plot_metric_inline(metric_dict: Dict[str, List[float]], title: str, ylabel: str):
    import matplotlib.pyplot as plt
    import streamlit as st

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, series in metric_dict.items():
        ax.plot(series, label=label)

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)   # <-- PASS FIGURE, NOT MODULE
    plt.close(fig)
