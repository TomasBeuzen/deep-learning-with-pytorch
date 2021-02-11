import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchvision import utils
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16, 'axes.labelweight': 'bold'})
from .optimization import *


def plot_pokemon(
    x, y, y_hat=None, x_range=[10, 130], y_range=[10, 130], dx=20, dy=20
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", marker=dict(size=10), name="data")
    )
    if y_hat is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_hat,
                line_color="red",
                mode="lines",
                line=dict(width=3),
                name="Fitted line",
            )
        )
        width = 550
        title_x = 0.46
    else:
        width = 500
        title_x = 0.5
    fig.update_layout(
        width=width,
        height=500,
        title="Pokemon stats",
        title_x=title_x,
        title_y=0.93,
        xaxis_title="defense",
        yaxis_title="attack",
        margin=dict(t=60),
    )
    fig.update_xaxes(range=x_range, tick0=x_range[0], dtick=dx)
    fig.update_yaxes(range=y_range, tick0=y_range[0], dtick=dy)
    return fig


def plot_logistic(
    x,
    y,
    y_hat=None,
    threshold=None,
    x_range=[-3, 3],
    y_range=[-0.25, 1.25],
    dx=1,
    dy=0.25,
):
    fig = go.Figure()
    fig.update_xaxes(range=x_range, tick0=x_range[0], dtick=dx)
    fig.update_yaxes(range=y_range, tick0=y_range[0], dtick=dy)
    if threshold is not None:
        threshold_ind = (np.abs(y_hat - threshold)).argmin()
        fig.add_trace(
            go.Scatter(
                x=[x_range[0], x_range[0], x[threshold_ind], x[threshold_ind]],
                y=[y_range[0], y_range[1], y_range[1], y_range[0]],
                mode="lines",
                fill="toself",
                fillcolor="limegreen",
                opacity=0.2,
                line=dict(width=0),
                name="0 prediction",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x[threshold_ind], x[threshold_ind], x_range[1], x_range[1]],
                y=[y_range[0], y_range[1], y_range[1], y_range[0]],
                mode="lines",
                fill="toself",
                fillcolor="lightsalmon",
                opacity=0.3,
                line=dict(width=0),
                name="1 prediction",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=10,
                color="#636EFA",
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            name="data",
        )
    )
    if y_hat is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_hat,
                line_color="red",
                mode="lines",
                line=dict(width=3),
                name="Fitted line",
            )
        )
        width = 650
        title_x = 0.46
    else:
        width = 600
        title_x = 0.5
    if threshold is not None:
        fig.add_trace(
            go.Scatter(
                x=[x[threshold_ind]],
                y=[threshold],
                mode="markers",
                marker=dict(
                    size=18,
                    color="gold",
                    line=dict(width=1, color="DarkSlateGrey"),
                ),
                name="Threshold",
            )
        )
    fig.update_layout(
        width=width,
        height=500,
        title="Pokemon stats",
        title_x=title_x,
        title_y=0.93,
        xaxis_title="defense",
        yaxis_title="legendary",
        margin=dict(t=60),
    )
    return fig


def plot_gradient_m(x, y, m, slopes, mse, grad_func):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=slopes,
            y=mse,
            line_color="#1ac584",
            line=dict(width=3),
            mode="lines",
            name="MSE",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=slopes,
            y=mean_squared_error(y, m * x) + grad_func(x, y, m) * (slopes - m),
            line_color="red",
            mode="lines",
            line=dict(width=2),
            name="gradient",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[m],
            y=[mean_squared_error(y, m * x)],
            line_color="red",
            marker=dict(size=14, line=dict(width=1, color="DarkSlateGrey")),
            mode="markers",
            name=f"slope {m}",
        )
    )
    fig.update_layout(
        width=520,
        height=450,
        xaxis_title="slope (w)",
        yaxis_title="MSE",
        title=f"slope {m:.1f}, gradient {grad_func(x, y, m):.1f}",
        title_x=0.46,
        title_y=0.93,
        margin=dict(t=60),
    )
    fig.update_xaxes(range=[0.4, 1.6], tick0=0.4, dtick=0.2)
    fig.update_yaxes(range=[0, 2500])
    return fig


def plot_grid_search(
    x,
    y,
    slopes,
    loss_function,
    title="Mean Squared Error",
    y_range=[0, 2500],
    y_title="MSE",
):
    mse = []
    df = pd.DataFrame()
    for m in slopes:
        df[f"{m:.2f}"] = m * x  # store predictions for plotting later
        mse.append(loss_function(y, m * x))  # calc MSE
    mse = pd.DataFrame({"slope": slopes, "squared_error": mse})
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Data & Fitted Line", title)
    )
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", marker=dict(size=10), name="Data"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df.iloc[:, 0],
            line_color="red",
            mode="lines",
            line=dict(width=3),
            name="Fitted line",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=mse["slope"],
            y=mse["squared_error"],
            mode="markers",
            marker=dict(size=7),
            name="MSE",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=mse.iloc[[0]]["slope"],
            y=mse.iloc[[0]]["squared_error"],
            line_color="red",
            mode="markers",
            marker=dict(size=14, line=dict(width=1, color="DarkSlateGrey")),
            name="MSE for line",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(width=900, height=475)
    fig.update_xaxes(
        range=[10, 130],
        tick0=10,
        dtick=20,
        row=1,
        col=1,
        title="defense",
        title_standoff=0,
    )
    fig.update_xaxes(
        range=[0.3, 1.6],
        tick0=0.3,
        dtick=0.2,
        row=1,
        col=2,
        title="slope",
        title_standoff=0,
    )
    fig.update_yaxes(
        range=[10, 130],
        tick0=10,
        dtick=20,
        row=1,
        col=1,
        title="attack",
        title_standoff=0,
    )
    fig.update_yaxes(
        range=y_range, row=1, col=2, title=y_title, title_standoff=0
    )
    frames = [
        dict(
            name=f"{slope:.2f}",
            data=[
                go.Scatter(x=x, y=y),
                go.Scatter(x=x, y=df[f"{slope:.2f}"]),
                go.Scatter(x=mse["slope"], y=mse["squared_error"]),
                go.Scatter(
                    x=mse.iloc[[n]]["slope"], y=mse.iloc[[n]]["squared_error"]
                ),
            ],
            traces=[0, 1, 2, 3],
        )
        for n, slope in enumerate(slopes)
    ]

    sliders = [
        {
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "slope: ",
                "visible": True,
            },
            "pad": {"b": 10, "t": 30},
            "steps": [
                {
                    "args": [
                        [f"{slope:.2f}"],
                        {
                            "frame": {
                                "duration": 0,
                                "easing": "linear",
                                "redraw": False,
                            },
                            "transition": {"duration": 0, "easing": "linear"},
                        },
                    ],
                    "label": f"{slope:.2f}",
                    "method": "animate",
                }
                for slope in slopes
            ],
        }
    ]
    fig.update(frames=frames), fig.update_layout(sliders=sliders)
    return fig


def plot_grid_search_2d(x, y, slopes, intercepts):
    mse = np.zeros((len(slopes), len(intercepts)))
    for i, slope in enumerate(slopes):
        for j, intercept in enumerate(intercepts):
            mse[i, j] = mean_squared_error(y, x * slope + intercept)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Surface Plot", "Contour Plot"),
        specs=[[{"type": "surface"}, {"type": "contour"}]],
    )
    fig.add_trace(
        go.Surface(
            z=mse, x=intercepts, y=slopes, name="", colorscale="viridis"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Contour(
            z=mse,
            x=intercepts,
            y=slopes,
            name="",
            showscale=False,
            colorscale="viridis",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        scene=dict(
            zaxis=dict(title="MSE"),
            yaxis=dict(title="slope (w<sub>1</sub>)"),
            xaxis=dict(title="intercept (w<sub>0</sub>)"),
        ),
        scene_camera=dict(eye=dict(x=2, y=1.1, z=1.2)),
        margin=dict(l=0, r=0, b=60, t=90),
    )
    fig.update_xaxes(
        title="intercept (w<sub>0</sub>)",
        range=[intercepts.max(), intercepts.min()],
        tick0=intercepts.max(),
        row=1,
        col=2,
        title_standoff=0,
    )
    fig.update_yaxes(
        title="slope (w<sub>1</sub>)",
        range=[slopes.min(), slopes.max()],
        tick0=slopes.min(),
        row=1,
        col=2,
        title_standoff=0,
    )
    fig.update_layout(width=900, height=475, margin=dict(t=60))
    return fig


def plot_gradient_descent(x, y, w, alpha, tolerance=2e-4, max_iterations=5000):
    if x.ndim == 1:
        x = np.array(x).reshape(-1, 1)
    slopes, losses = gradient_descent(
        x, y, [w], alpha, tolerance, max_iterations, history=True
    )
    slopes = [_[0] for _ in slopes]
    x = x.flatten()
    mse = []
    df = pd.DataFrame()
    for w in slopes:
        df[f"{w:.2f}"] = w * x  # store predictions for plotting later
    slope_range = np.arange(0.4, 1.65, 0.05)
    for w in slope_range:
        mse.append(mean_squared_error(y, w * x))  # calc MSE
    mse = pd.DataFrame({"slope": slope_range, "squared_error": mse})

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Data & Fitted Line", "Mean Squared Error"),
    )
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", marker=dict(size=10), name="Data"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df.iloc[:, 0],
            line_color="red",
            mode="lines",
            line=dict(width=3),
            name="Fitted line",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=mse["slope"],
            y=mse["squared_error"],
            line_color="#1ac584",
            line=dict(width=3),
            mode="lines",
            name="MSE",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(slopes[:1]),
            y=np.array(losses[:1]),
            line_color="salmon",
            line=dict(width=4),
            marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")),
            mode="markers+lines",
            name="Slope history",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(slopes[0]),
            y=np.array(losses[0]),
            line_color="red",
            mode="markers",
            marker=dict(size=18, line=dict(width=1, color="DarkSlateGrey")),
            name="MSE for line",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[30.3],
            y=[120],
            mode="text",
            text=f"<b>Slope {slopes[0]:.2f}<b>",
            textfont=dict(size=16, color="red"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.update_layout(width=900, height=475, margin=dict(t=60))
    fig.update_xaxes(
        range=[10, 130], tick0=10, dtick=20, title="defense", title_standoff=0, row=1, col=1
    ), fig.update_xaxes(range=[0.4, 1.6], tick0=0.4, dtick=0.2, title="slope (w)", title_standoff=0, row=1, col=2)
    fig.update_yaxes(
        range=[10, 130], tick0=10, dtick=20, title="attack", title_standoff=0, row=1, col=1
    ), fig.update_yaxes(range=[0, 2500], title="MSE", title_standoff=0, row=1, col=2)

    frames = [
        dict(
            name=n,
            data=[
                go.Scatter(x=x, y=y),
                go.Scatter(x=x, y=df[f"{slope:.2f}"]),
                go.Scatter(x=mse["slope"], y=mse["squared_error"]),
                go.Scatter(
                    x=np.array(slopes[: n + 1]),
                    y=np.array(losses[: n + 1]),
                    mode="markers" if n == 0 else "markers+lines",
                ),
                go.Scatter(x=np.array(slopes[n]), y=np.array(losses[n])),
                go.Scatter(text=f"<b>Slope {slope:.2f}<b>"),
            ],
            traces=[0, 1, 2, 3, 4, 5],
        )
        for n, slope in enumerate(slopes)
    ]

    sliders = [
        {
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Iteration: ",
                "visible": True,
            },
            "pad": {"b": 10, "t": 30},
            "steps": [
                {
                    "args": [
                        [n],
                        {
                            "frame": {
                                "duration": 0,
                                "easing": "linear",
                                "redraw": False,
                            },
                            "transition": {"duration": 0, "easing": "linear"},
                        },
                    ],
                    "label": n,
                    "method": "animate",
                }
                for n in range(len(slopes))
            ],
        }
    ]
    fig.update(frames=frames), fig.update_layout(sliders=sliders)
    return fig


def plot_gradient_descent_2d(
    x,
    y,
    w,
    alpha,
    m_range,
    b_range,
    tolerance=2e-5,
    max_iterations=5000,
    step_size=1,
    markers=False,
    stochastic=False,
    batch_size=None,
    seed=None,
):
    if x.ndim == 1:
        x = np.array(x).reshape(-1, 1)
    if stochastic:
        if batch_size is None:
            weights, losses = stochastic_gradient_descent(
                np.hstack((np.ones((len(x), 1)), x)),
                y,
                w,
                alpha,
                tolerance,
                max_iterations,
                history=True,
                seed=seed,
            )
            title = "Stochastic Gradient Descent"
        else:
            weights, losses = minibatch_gradient_descent(
                np.hstack((np.ones((len(x), 1)), x)),
                y,
                w,
                alpha,
                batch_size,
                tolerance,
                max_iterations,
                history=True,
                seed=seed,
            )
            title = "Minibatch Gradient Descent"
    else:
        weights, losses = gradient_descent(
            np.hstack((np.ones((len(x), 1)), x)),
            y,
            w,
            alpha,
            tolerance,
            max_iterations,
            history=True,
        )
        title = "Gradient Descent"
    weights = np.array(weights)
    intercepts, slopes = weights[:, 0], weights[:, 1]
    mse = np.zeros((len(m_range), len(b_range)))
    for i, slope in enumerate(m_range):
        for j, intercept in enumerate(b_range):
            mse[i, j] = mean_squared_error(y, x * slope + intercept)

    fig = make_subplots(
        rows=1,
        subplot_titles=[title],  # . Iterations = {len(intercepts) - 1}."],
    )
    fig.add_trace(
        go.Contour(z=mse, x=b_range, y=m_range, name="", colorscale="viridis")
    )
    mode = "markers+lines" if markers else "lines"
    fig.add_trace(
        go.Scatter(
            x=intercepts[::step_size],
            y=slopes[::step_size],
            mode=mode,
            line=dict(width=2.5),
            line_color="coral",
            marker=dict(
                opacity=1,
                size=np.linspace(19, 1, len(intercepts[::step_size])),
                line=dict(width=2, color="DarkSlateGrey"),
            ),
            name="Descent Path",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[intercepts[0]],
            y=[slopes[0]],
            mode="markers",
            marker=dict(size=20, line=dict(width=2, color="DarkSlateGrey")),
            marker_color="orangered",
            name="Start",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[intercepts[-1]],
            y=[slopes[-1]],
            mode="markers",
            marker=dict(size=20, line=dict(width=2, color="DarkSlateGrey")),
            marker_color="yellowgreen",
            name="End",
        )
    )
    fig.update_layout(
        width=700,
        height=600,
        margin=dict(t=60),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_xaxes(
        title="intercept (w<sub>0</sub>)",
        range=[b_range.min(), b_range.max()],
        tick0=b_range.min(),
        row=1,
        col=1,
        title_standoff=0,
    )
    fig.update_yaxes(
        title="slope (w<sub>1</sub>)",
        range=[m_range.min(), m_range.max()],
        tick0=m_range.min(),
        row=1,
        col=1,
        title_standoff=0,
    )
    return fig


def plot_random_gradients(x, y, w, num_of_points=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    randn = np.random.randint(0, len(x), num_of_points)
    fig = go.Figure()
    slopes = np.arange(-30, 60, 1)
    mse = np.array([mean_squared_error(y, m * x) for m in slopes])
    fig.add_trace(
        go.Scatter(
            x=slopes,
            y=mse,
            line_color="#1ac584",
            line=dict(width=3),
            mode="lines",
            name="MSE",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=slopes,
            y=mean_squared_error(y, w * x)
            + gradient(x[randn[0], None], y[randn[0], None], [w])[0]
            * (slopes - w),
            line_color="black",
            mode="lines",
            line=dict(width=2),
            name="gradient for one data point",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=slopes,
            y=mean_squared_error(y, w * x)
            + gradient(x[:, None], y, [w])[0][0] * (slopes - w),
            line_color="red",
            mode="lines",
            line=dict(width=3),
            name="gradient for all data",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[w],
            y=[mean_squared_error(y, w * x)],
            line_color="red",
            marker=dict(size=14, line=dict(width=2, color="DarkSlateGrey")),
            mode="markers",
            name=f"slope {w}",
        )
    )

    frames = [
        dict(
            name=f"{n}",
            data=[
                go.Scatter(x=slopes, y=mse),
                go.Scatter(
                    x=slopes,
                    y=mean_squared_error(y, w * x)
                    + gradient(x[n, None], y[n, None], [w])[0] * (slopes - w),
                ),
                go.Scatter(
                    x=slopes,
                    y=mean_squared_error(y, w * x)
                    + gradient(x[:, None], y, [w])[0][0] * (slopes - w),
                ),
                go.Scatter(x=[w], y=[mean_squared_error(y, w * x)]),
            ],
            traces=[0, 1, 2, 3],
        )
        for n in randn
    ]

    sliders = [
        {
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Data point: ",
                "visible": True,
            },
            "pad": {"b": 10, "t": 30},
            "steps": [
                {
                    "args": [
                        [f"{n}"],
                        {
                            "frame": {
                                "duration": 0,
                                "easing": "linear",
                                "redraw": False,
                            },
                            "transition": {"duration": 0, "easing": "linear"},
                        },
                    ],
                    "label": f"{n}",
                    "method": "animate",
                }
                for n in randn
            ],
        }
    ]
    fig.update_layout(
        width=650,
        height=500,
        xaxis_title="slope",
        yaxis_title="MSE",
        title_x=0.46,
        title_y=0.93,
        margin=dict(b=20, t=60),
    )
    fig.update(frames=frames), fig.update_layout(sliders=sliders)
    fig.update_xaxes(range=[-30, 60], dtick=10, title_standoff=0)
    fig.update_yaxes(range=[6500, 9000], dtick=500, title_standoff=0)
    return fig


def plot_gradient_histogram(x, y, w):
    gradients = [
        gradient(x[i, None], y[i, None], [w])[0] for i in range(len(x))
    ]
    fig = go.Figure(data=[go.Histogram(x=gradients)])
    fig.update_layout(
        width=600,
        height=400,
        margin=dict(t=60),
        title=f"Histogram of gradients at slope {w}",
        title_x=0.5,
        title_y=0.9,
    )
    fig.update_xaxes(title="gradient", title_standoff=0)
    fig.update_yaxes(title="frequency", title_standoff=0)
    return fig


def plot_minibatch_gradients(x, y, w, batch_sizes=[1], seed=None):
    if seed is not None:
        np.random.seed(seed)
    batches = []
    for _ in batch_sizes:
        batches.append(np.random.choice(range(len(x)), _, replace=False))
    fig = go.Figure()
    slopes = np.arange(-30, 60, 1)
    mse = np.array([mean_squared_error(y, m * x) for m in slopes])
    fig.add_trace(
        go.Scatter(
            x=slopes,
            y=mse,
            line_color="#1ac584",
            line=dict(width=3),
            mode="lines",
            name="MSE",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=slopes,
            y=mean_squared_error(y, w * x)
            + gradient(x[batches[0]], y[batches[0]], [w])[0] * (slopes - w),
            line_color="black",
            mode="lines",
            line=dict(width=2),
            name=f"gradient for batch",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=slopes,
            y=mean_squared_error(y, w * x)
            + gradient(x[:, None], y, [w])[0][0] * (slopes - w),
            line_color="red",
            mode="lines",
            line=dict(width=3),
            name="gradient for all data",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[w],
            y=[mean_squared_error(y, w * x)],
            line_color="red",
            marker=dict(size=14, line=dict(width=2, color="DarkSlateGrey")),
            mode="markers",
            name=f"slope {w}",
        )
    )

    frames = [
        dict(
            name=f"{len(batch)}",
            data=[
                go.Scatter(x=slopes, y=mse),
                go.Scatter(
                    x=slopes,
                    y=mean_squared_error(y, w * x)
                    + gradient(x[batch, None], y[batch], [w])[0] * (slopes - w),
                ),
                go.Scatter(
                    x=slopes,
                    y=mean_squared_error(y, w * x)
                    + gradient(x[:, None], y, [w])[0][0] * (slopes - w),
                ),
                go.Scatter(x=[w], y=[mean_squared_error(y, w * x)]),
            ],
            traces=[0, 1, 2, 3],
        )
        for batch in batches
    ]

    sliders = [
        {
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Batch size: ",
                "visible": True,
            },
            "pad": {"b": 10, "t": 30},
            "steps": [
                {
                    "args": [
                        [f"{len(batch)}"],
                        {
                            "frame": {
                                "duration": 0,
                                "easing": "linear",
                                "redraw": False,
                            },
                            "transition": {"duration": 0, "easing": "linear"},
                        },
                    ],
                    "label": f"{len(batch)}",
                    "method": "animate",
                }
                for batch in batches
            ],
        }
    ]
    fig.update_layout(
        width=650,
        height=500,
        xaxis_title="slope",
        yaxis_title="MSE",
        title_x=0.46,
        title_y=0.93,
        margin=dict(b=20, t=60),
    )
    fig.update(frames=frames), fig.update_layout(sliders=sliders)
    fig.update_xaxes(range=[-30, 60], dtick=10, title_standoff=0)
    fig.update_yaxes(range=[6500, 9000], dtick=500, title_standoff=0)
    return fig


def plot_panel(f1, f2, f3):
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Gradient Descent",
            "Stochastic Gradient Descent",
            "Minibatch Gradient Descent",
        ),
    )
    for n, f in enumerate((f1, f2, f3)):
        for _ in range(len(f.data)):
            fig.add_trace(f.data[_], row=1, col=n + 1)
        fig.update_xaxes(range=[-40, 138], row=1, col=n + 1)
        fig.update_yaxes(range=[-30, 58], row=1, col=n + 1)
    fig.update_layout(
        width=1000, height=400, margin=dict(t=60), showlegend=False
    )
    return fig


def plot_regression(
    x, y, y_hat=None, x_range=[-3, 3], y_range=[-150, 150], dx=1, dy=30
):
    if x.ndim > 1:
        x = np.squeeze(x)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", marker=dict(size=10), name="data")
    )
    if y_hat is not None:
        if y_hat.ndim > 1:
            y_hat = np.squeeze(y_hat)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_hat,
                line_color="red",
                mode="lines",
                line=dict(width=3),
                name="Fitted line",
            )
        )
        width = 550
        title_x = 0.46
    else:
        width = 500
        title_x = 0.5
    fig.update_layout(
        width=width,
        height=500,
        # title="Pokemon stats",
        title_x=title_x,
        title_y=0.93,
        xaxis_title="x",
        yaxis_title="y",
        margin=dict(t=60),
    )
    fig.update_xaxes(range=x_range, tick0=x_range[0], dtick=dx)
    fig.update_yaxes(range=y_range, tick0=y_range[0], dtick=dy)
    return fig


def plot_nodes(
    x,
    y,
    model,
    x_range=[-3, 3],
    y_range=[-25, 25],
    dx=1,
    dy=5,
):
    m = model.main.state_dict()
    y_nodes = (
        sigmoid((x * m["0.weight"].numpy()).T + m["0.bias"].numpy())
        * m["2.weight"].numpy()
    )
    nodes = [str(_) for _ in range(1, y_nodes.shape[1] + 1)] + ["Bias"]
    nodes = [nodes[: n + 1] for n, _ in enumerate(nodes)]
    y_nodes = np.concatenate(
        (y_nodes, np.ones((len(y_nodes), 1)) * m["2.bias"].numpy()), axis=1
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Node Decomposition", "Interactive Re-composition"),
    )
    # PLOT 1
    for i in range(y_nodes.shape[1] - 1):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_nodes[:, i],
                mode="lines",
                name=f"Node {i+1}",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_nodes[:, -1],
            mode="lines",
            name="Output Bias",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Prediction",
            line_color="red",
            line=dict(width=4, dash="dot"),
        ),
        row=1,
        col=1,
    )

    # SUBPLOT 2
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            showlegend=False,
            line_color="red",
            line=dict(width=4, dash="dot"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_nodes[:, :1].sum(axis=1),
            line=dict(width=2),
            mode="lines",
            line_color="darkslategrey",
            name="Node Sum",
        ),
        row=1,
        col=2,
    )

    frames = [
        dict(
            name=f"{', '.join(node)}",
            data=[
                go.Scatter(x=x, y=y_nodes[:, i])
                for i in range(y_nodes.shape[1] - 1)
            ]
            + [
                go.Scatter(x=x, y=y_nodes[:, -1]),
                go.Scatter(x=x, y=y),
                go.Scatter(x=x, y=y),
                go.Scatter(x=x, y=y_nodes[:, : n + 1].sum(axis=1)),
            ],
            traces=list(range(y_nodes.shape[1] - 1 + 4)),
        )
        for n, node in enumerate(nodes)
    ]

    sliders = [
        {
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Sum of: ",
                "visible": True,
            },
            "pad": {"b": 10, "t": 40, "l": 430},
            "steps": [
                {
                    "args": [
                        [f"{', '.join(node)}"],
                        {
                            "frame": {
                                "duration": 0,
                                "easing": "linear",
                                "redraw": False,
                            },
                            "transition": {
                                "duration": 0,
                                "easing": "linear",
                            },
                        },
                    ],
                    "label": f"{', '.join(node)}",
                    "method": "animate",
                }
                for node in nodes
            ],
        }
    ]
    fig.update(frames=frames), fig.update_layout(sliders=sliders)

    fig.update_layout(width=1000, height=500, margin=dict(t=60))
    fig.update_xaxes(range=x_range, tick0=x_range[0], dtick=dx, row=1, col=1)
    fig.update_xaxes(range=x_range, tick0=x_range[0], dtick=dx, row=1, col=2)
    fig.update_yaxes(range=y_range, tick0=y_range[0], dtick=dy, row=1, col=1)
    fig.update_yaxes(range=y_range, tick0=y_range[0], dtick=dy, row=1, col=2)
    return fig


def plot_activations(x, functions, rows=2, cols=3, width=800, height=500):
    names = [_.__name__ for _ in functions]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=names,
        # horizontal_spacing=None,
        vertical_spacing=0.12,
    )
    i, j = 1, 1
    for f in functions:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=f()(x),
                mode="lines",
                name=f"{f.__name__}",
                line=dict(width=3),
            ),
            row=i,
            col=j,
        )
        j += 1
        if j > cols:
            i += 1
            j = 1
    fig.update_layout(
        width=width, height=height, margin=dict(b=30, t=60), showlegend=False
    )
    return fig


def plot_classification_2d(X, y, model=None, transform="Sigmoid"):
    c = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]
    yy = [c[_] for _ in y]
    fig = go.Figure()
    if model is None:
        for class_ in np.unique(y):
            mask = y == class_
            fig.add_trace(
                go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode="markers",
                    marker=dict(size=10),
                    name=f"Class {class_}",
                )
            )
        fig.update_layout(
            width=500,
            height=500,
            title="Binary Classification",
            title_x=0.5,
            title_y=0.93,
            xaxis_title="x",
            yaxis_title="y",
            margin=dict(t=60),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        fig.update_xaxes(range=[-1.5, 1.5], tick0=-1.5, dtick=0.5)
        fig.update_yaxes(range=[-1.5, 1.5], tick0=-1.5, dtick=0.5)
    else:
        yy = [c[_] for _ in y]
        x_t = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        xx1, xx2 = torch.meshgrid(
            torch.linspace(-1.5, 1.5, 25),
            torch.linspace(-1.5, 1.5, 25),
        )
        if transform == "Sigmoid":
            Z = model(
                torch.cat((xx1.reshape(1, -1).T, xx2.reshape(1, -1).T), dim=1)
            ).reshape(xx1.shape)
            Z = torch.nn.Sigmoid()(Z)
            fig = go.Figure(
                data=go.Contour(
                    z=Z.detach(),
                    x=xx1[:, 0].detach(),
                    y=xx2[0, :].detach(),
                    colorscale="rdbu",
                    reversescale=True,
                    name="Predictions",
                    contours=dict(start=0, end=1, size=0.1),
                    colorbar=dict(title="Probability of Class 1"),
                )
            )
        elif transform == "Softmax":
            Z = torch.nn.Softmax(dim=0)(
                model(
                    torch.cat(
                        (xx1.reshape(1, -1).T, xx2.reshape(1, -1).T), dim=1
                    )
                )
            ).argmax(dim=1)
            ZZ = [c[_] for _ in Z]
            fig.add_trace(
                go.Scatter(
                    x=xx1.reshape(1, -1).T[:, 0].detach(),
                    y=xx2.reshape(1, -1).T[:, 0].detach(),
                    mode="markers",
                    marker=dict(size=15, color=ZZ, opacity=0.3),
                    showlegend=False,
                )
            )
        for class_ in np.unique(y):
            mask = y == class_
            fig.add_trace(
                go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=c[class_],
                        line=dict(width=1, color="DarkSlateGrey"),
                    ),
                    name=f"Class {class_}",
                ),
            )
        fig.update_layout(
            width=550,
            height=500,
            title="Binary Classification",
            title_x=0.5,
            title_y=0.93,
            xaxis_title="x",
            yaxis_title="y",
            margin=dict(t=60),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        fig.update_xaxes(range=[-1.5, 1.5], tick0=-1.5, dtick=0.5)
        fig.update_yaxes(range=[-1.5, 1.5], tick0=-1.5, dtick=0.5)
    return fig


def plot_loss(train_loss, validation_loss=None, train_acc=None, valid_acc=None):
    # Make figure
    if train_acc is not None:
        fig = make_subplots(rows=1, cols=2)
    else:
        fig = go.Figure()
    # Add losses
    fig.add_trace(
        go.Scatter(x=np.arange(len(train_loss)), y=train_loss, mode="lines", line=dict(width=2), name="Training loss")
    )
    if validation_loss is not None:
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(validation_loss)),
                y=validation_loss,
                mode="lines",
                line=dict(width=2),
                name="Validation loss",
            )
        )
        width = 400
    else:
        width = 400
    # add accuracy
    if train_acc is not None:
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(train_acc)),
                y=train_acc,
                mode="lines",
                line=dict(width=2),
                name="Training accuracy",
            ),
            row=1,
            col=2
        )
        width = 550
        title_x = 0.46
        if valid_acc is not None:
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(valid_acc)),
                    y=valid_acc,
                    mode="lines",
                    line=dict(width=2),
                    name="Validation accuracy",
                ),
                row=1,
                col=2
            )
        fig.update_layout(
            width=800,
            height=400,
            title_y=0.93,
            margin=dict(t=60),
        )
        fig.update_xaxes(title_text="Epochs", title_standoff=0, row=1, col=1)
        fig.update_xaxes(title_text="Epochs", title_standoff=0, row=1, col=2)
        fig.update_yaxes(title_text="Loss", title_standoff=0, row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", title_standoff=0, row=1, col=2)
        
    else:
        fig.update_layout(
            width=width,
            height=400,
            title_y=0.93,
            xaxis_title="Epochs",
            yaxis_title="Loss",
            margin=dict(t=60),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.434
            )
        )
    return fig


def plot_bitmojis(sample_batch, rgb=False):
    plt.figure(figsize=(10, 8))
    plt.axis("off")
    plt.title("Sample Training Images")
    if rgb:
        plt.imshow(np.transpose(utils.make_grid(sample_batch, padding=1, normalize=True),(1,2,0)));
    else:
        plt.imshow(np.transpose(utils.make_grid(sample_batch[0], padding=1, normalize=True),(1,2,0)));

    
def plot_bitmoji(image, label):
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.title(f"Prediction: {['not_tom', 'tom'][label]}", pad=10)
    plt.imshow(image[0, 0], cmap='gray');
    

def plot_conv(image, filter):
    """Plot convs with matplotlib."""
    d = filter.shape[-1]
    conv = torch.nn.Conv2d(1, 1, kernel_size=(d, d), padding=1)
    conv.weight[:] = filter
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 4), ncols=2)
    ax1.imshow(image, cmap='gray')
    ax1.axis('off')
    ax1.set_title("Original")
    ax2.imshow(conv(image[None, None, :]).detach().squeeze(), cmap='gray')  
    ax2.set_title("Filtered")
    ax2.axis('off')
    plt.tight_layout();
    
def plot_convs(image, conv_layer, axis=False):
    """Plot convs with matplotlib. Sorry for this lazy code :D"""
    filtered_image = conv_layer(image[None, None, :])
    n = filtered_image.shape[1]
    if n == 1:
        fig, (ax1, ax2) = plt.subplots(figsize=(8, 4), ncols=2)
        ax1.imshow(image, cmap='gray')
        ax1.set_title("Original")
        ax2.imshow(filtered_image.detach().squeeze(), cmap='gray')  
        ax2.set_title("Filter 1")
        ax1.grid(False)
        ax2.grid(False)
        if not axis:
            ax1.axis(False)
            ax2.axis(False)
        plt.tight_layout();
    elif n == 2:
        filtered_image_1 = filtered_image[:,0,:,:]
        filtered_image_2 = filtered_image[:,1,:,:]
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(10, 4), ncols=3)
        ax1.imshow(image, cmap='gray')
        ax1.set_title("Original")
        ax2.imshow(filtered_image_1.detach().squeeze(), cmap='gray')  
        ax2.set_title("Filter 1")
        ax3.imshow(filtered_image_2.detach().squeeze(), cmap='gray')  
        ax3.set_title("Filter 2")
        ax1.grid(False)
        ax2.grid(False)
        ax3.grid(False)
        if not axis:
            ax1.axis(False)
            ax2.axis(False)
            ax3.axis(False)
        plt.tight_layout();
    elif n == 3:
        filtered_image_1 = filtered_image[:,0,:,:]
        filtered_image_2 = filtered_image[:,1,:,:]
        filtered_image_3 = filtered_image[:,2,:,:]
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(12, 4), ncols=4)
        ax1.imshow(image, cmap='gray')
        ax1.set_title("Original")
        ax2.imshow(filtered_image_1.detach().squeeze(), cmap='gray')  
        ax2.set_title("Filter 1")
        ax3.imshow(filtered_image_2.detach().squeeze(), cmap='gray')  
        ax3.set_title("Filter 2")
        ax4.imshow(filtered_image_3.detach().squeeze(), cmap='gray')  
        ax4.set_title("Filter 3")
        ax1.grid(False)
        ax2.grid(False)
        ax3.grid(False)
        ax4.grid(False)
        if not axis:
            ax1.axis(False)
            ax2.axis(False)
            ax3.axis(False)
            ax4.axis(False)
        plt.tight_layout();
        
def plot_scatter3D(X, y):
    fig = go.Figure(data=[go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], mode='markers', marker=dict(size=4, color=y, colorscale='rdylbu'))])
    fig.update_layout(margin=dict(b=30, t=60), showlegend=False, scene=dict(
        zaxis=dict(title="X3"),
        yaxis=dict(title="X2"),
        xaxis=dict(title="X1"),
    )
    )
    return fig

def plot_scatter2D(X, y):
    fig = go.Figure(data=[go.Scatter(x=X[:,0].detach(), y=X[:,1].detach(), mode='markers', marker=dict(color=y, size=4, colorscale='rdylbu'))])
    fig.update_layout(width=500, height=400, margin=dict(b=30, t=60), showlegend=False
)
    fig.update_xaxes(title="Encoded feature 1", title_standoff=0)
    fig.update_yaxes(title="Encoded feature 2", title_standoff=0)
    return fig


def plot_eights(X, noise=0.5):
    fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(7, 5))
    noise = noise * torch.randn(*X[1, 0, :, :].shape)
    ax1[0].imshow(X[1, 0, :, :], cmap="gray")
    ax2[0].imshow(X[1, 0, :, :] + noise, cmap="gray")
    ax1[1].imshow(X[2, 0, :, :], cmap="gray")
    ax1[1].set_title("Original 8's")
    ax2[1].imshow(X[2, 0, :, :] + noise, cmap="gray")
    ax2[1].set_title("Noisy 8's")
    ax1[2].imshow(X[3, 0, :, :], cmap="gray")
    ax2[2].imshow(X[3, 0, :, :] + noise, cmap="gray")
    plt.tight_layout();
    
    
def plot_eight_pair(input_8, output_8):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))
    ax1.imshow(input_8.squeeze().detach(), cmap="gray")
    ax1.set_title("Input")
    ax2.imshow(output_8.squeeze().detach(), cmap="gray")
    ax2.set_title("Output")
    plt.tight_layout();
    

def plot_gan_loss(dis_loss, gen_loss):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=np.arange(1, 1+len(dis_loss)), y=dis_loss, mode="lines", line=dict(width=2), name="Discriminator loss")
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, 1+len(gen_loss)),
            y=gen_loss,
            mode="lines",
            line=dict(width=2),
            name="Generator loss",
        )
    )
    fig.update_layout(
        width=550,
        height=400,
        title_y=0.93,
        xaxis_title="Epochs",
        yaxis_title="Loss",
        margin=dict(t=60),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.434
        )
    )
    return fig