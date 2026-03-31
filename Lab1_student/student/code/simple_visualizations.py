import altair as alt
import pandas as pd
import numpy as np


def visualize_module(X, module):
    if X.ndim == 1:
        data = pd.DataFrame({'x': X})
        data['y'] = module(X)
        chart = alt.Chart(data).mark_line().encode(
            x='x',
            y='y'
        ).interactive()
        chart.show()


def visualize_dataset(dataset, is_classification=True):
    X = []
    Y = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X.append(x)
        Y.append(y)

    X_np = np.asarray(X)
    Y_np = np.asarray(Y)

    df = pd.DataFrame({
        "x1": X_np[:, 0],
        "x2": X_np[:, 1],
        "label": Y_np.astype(int) if is_classification else Y_np
    })

    color_encoding = (
        alt.Color("label:N", title="Label ($y$)") if is_classification
        else alt.Color("label:Q", title="Label ($y$)", scale=alt.Scale(scheme='viridis'))
    )

    scatter = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X("x1", title="Feature $x_1$"),
        y=alt.Y("x2", title="Feature $x_2$"),
        color=color_encoding,
        tooltip=["x1", "x2", "label"]
    )

    chart = scatter.interactive()

    return chart