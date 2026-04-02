"""Metric comparison plots and heatmaps."""

from __future__ import annotations

from typing import Any

import numpy as np


class MetricPlotter:
    """Create comparison plots for benchmark metrics."""

    METRIC_LABELS = {
        "ttc_min": "Min TTC (s)",
        "pet_min": "Min PET (s)",
        "collision_count": "Collision Count",
        "collision_rate": "Collision Rate",
        "max_jerk": "Max Jerk (m/s³)",
        "mean_jerk": "Mean Jerk (m/s³)",
        "max_lateral_deviation": "Max Lat. Dev. (m)",
        "mean_lateral_deviation": "Mean Lat. Dev. (m)",
        "max_heading_error": "Max Head. Err. (rad)",
        "mean_heading_error": "Mean Head. Err. (rad)",
        "overall_score": "Overall Score",
    }

    def bar_comparison(
        self,
        results: dict[str, dict[str, float]],
        metrics: list[str] | None = None,
        title: str = "Planner Comparison",
        save_path: str | None = None,
    ) -> Any:
        """Create a grouped bar chart comparing planners across metrics.

        Args:
            results: {planner_name: {metric_name: value}}.
            metrics: Which metrics to plot. Defaults to all.
            save_path: If provided, save the figure.
        """
        import matplotlib.pyplot as plt

        planner_names = list(results.keys())
        if metrics is None:
            metrics = list(next(iter(results.values())).keys())

        n_planners = len(planner_names)
        n_metrics = len(metrics)
        x = np.arange(n_metrics)
        width = 0.8 / n_planners

        fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.5), 6))

        for i, planner in enumerate(planner_names):
            values = [results[planner].get(m, 0) for m in metrics]
            ax.bar(x + i * width, values, width, label=planner, alpha=0.85)

        labels = [self.METRIC_LABELS.get(m, m) for m in metrics]
        ax.set_xticks(x + width * (n_planners - 1) / 2)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def radar_chart(
        self,
        results: dict[str, dict[str, float]],
        metrics: list[str] | None = None,
        title: str = "Planner Radar Chart",
        save_path: str | None = None,
    ) -> Any:
        """Create a radar chart comparing planners."""
        import matplotlib.pyplot as plt

        planner_names = list(results.keys())
        if metrics is None:
            metrics = list(next(iter(results.values())).keys())

        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for planner in planner_names:
            values = [results[planner].get(m, 0) for m in metrics]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=planner)
            ax.fill(angles, values, alpha=0.1)

        labels = [self.METRIC_LABELS.get(m, m) for m in metrics]
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title(title, y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def heatmap(
        self,
        data: dict[str, dict[str, float]],
        title: str = "Metric Heatmap",
        save_path: str | None = None,
    ) -> Any:
        """Create a heatmap of planner vs scenario metrics.

        Args:
            data: {planner_name: {scenario_name: score}}.
        """
        import matplotlib.pyplot as plt

        planners = list(data.keys())
        scenarios = sorted(set(s for d in data.values() for s in d))

        matrix = np.zeros((len(planners), len(scenarios)))
        for i, planner in enumerate(planners):
            for j, scenario in enumerate(scenarios):
                matrix[i, j] = data[planner].get(scenario, 0)

        fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.2), max(5, len(planners) * 0.8)))
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")

        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_yticks(np.arange(len(planners)))
        ax.set_xticklabels(scenarios, rotation=45, ha="right")
        ax.set_yticklabels(planners)

        for i in range(len(planners)):
            for j in range(len(scenarios)):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=9)

        plt.colorbar(im, ax=ax, label="Score")
        ax.set_title(title)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def metric_over_time(
        self,
        time_series: dict[str, list[float]],
        times: list[float],
        ylabel: str = "Value",
        title: str = "Metric Over Time",
        save_path: str | None = None,
    ) -> Any:
        """Plot metric evolution over simulation time."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))

        for label, values in time_series.items():
            ax.plot(times[: len(values)], values, linewidth=1.5, label=label)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
