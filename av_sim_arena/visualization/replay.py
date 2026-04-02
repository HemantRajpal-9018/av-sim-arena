"""Scenario replay visualization using matplotlib animation."""

from __future__ import annotations

from typing import Any

import numpy as np

from av_sim_arena.metrics.safety import TrajectoryPoint


class ScenarioReplay:
    """Replay a simulation scenario as a matplotlib animation."""

    def __init__(
        self,
        ego_trajectory: list[TrajectoryPoint],
        npc_trajectories: list[list[TrajectoryPoint]] | None = None,
        road_width: float = 3.7,
        num_lanes: int = 2,
        title: str = "Scenario Replay",
    ):
        self.ego_trajectory = ego_trajectory
        self.npc_trajectories = npc_trajectories or []
        self.road_width = road_width
        self.num_lanes = num_lanes
        self.title = title

    def create_animation(
        self,
        interval: int = 50,
        save_path: str | None = None,
    ) -> Any:
        """Create a matplotlib animation of the scenario.

        Args:
            interval: Milliseconds between frames.
            save_path: If provided, save animation to this path.

        Returns:
            matplotlib.animation.FuncAnimation object.
        """
        import matplotlib.animation as animation
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

        all_x = [p.x for p in self.ego_trajectory]
        all_y = [p.y for p in self.ego_trajectory]
        for npc_traj in self.npc_trajectories:
            all_x.extend(p.x for p in npc_traj)
            all_y.extend(p.y for p in npc_traj)

        x_min, x_max = min(all_x) - 10, max(all_x) + 10
        y_min = -self.road_width * self.num_lanes - 5
        y_max = self.road_width * self.num_lanes + 5

        ego_rect = patches.Rectangle(
            (0, 0), 4.5, 2.0, angle=0, fc="blue", ec="darkblue", alpha=0.8
        )
        ax.add_patch(ego_rect)

        npc_rects = []
        colors = ["red", "orange", "green", "purple", "brown"]
        for i in range(len(self.npc_trajectories)):
            color = colors[i % len(colors)]
            rect = patches.Rectangle(
                (0, 0), 4.0, 1.8, angle=0, fc=color, ec="black", alpha=0.7
            )
            ax.add_patch(rect)
            npc_rects.append(rect)

        (ego_trail,) = ax.plot([], [], "b-", linewidth=1, alpha=0.5)

        def init():
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(self.title)
            ax.grid(True, alpha=0.3)

            for lane in range(self.num_lanes + 1):
                y = -self.road_width * self.num_lanes / 2 + lane * self.road_width
                style = "-" if lane in (0, self.num_lanes) else "--"
                ax.axhline(y=y, color="gray", linestyle=style, linewidth=1)

            return [ego_rect, ego_trail] + npc_rects

        def update(frame):
            if frame < len(self.ego_trajectory):
                pt = self.ego_trajectory[frame]
                ego_rect.set_xy((pt.x - 2.25, pt.y - 1.0))
                ego_rect.angle = np.degrees(pt.heading)

                trail_x = [p.x for p in self.ego_trajectory[: frame + 1]]
                trail_y = [p.y for p in self.ego_trajectory[: frame + 1]]
                ego_trail.set_data(trail_x, trail_y)

            for i, npc_traj in enumerate(self.npc_trajectories):
                if frame < len(npc_traj) and i < len(npc_rects):
                    npc_pt = npc_traj[frame]
                    npc_rects[i].set_xy((npc_pt.x - 2.0, npc_pt.y - 0.9))
                    npc_rects[i].angle = np.degrees(npc_pt.heading)

            return [ego_rect, ego_trail] + npc_rects

        num_frames = len(self.ego_trajectory)
        anim = animation.FuncAnimation(
            fig, update, init_func=init, frames=num_frames,
            interval=interval, blit=True,
        )

        if save_path:
            anim.save(save_path, writer="pillow", fps=1000 // interval)

        return anim

    def plot_trajectory(self, save_path: str | None = None) -> Any:
        """Plot the full trajectory as a static image."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

        ego_x = [p.x for p in self.ego_trajectory]
        ego_y = [p.y for p in self.ego_trajectory]
        ego_speed = [p.speed for p in self.ego_trajectory]

        sc = ax.scatter(ego_x, ego_y, c=ego_speed, cmap="RdYlGn", s=10, zorder=5)
        plt.colorbar(sc, ax=ax, label="Speed (m/s)")

        colors = ["red", "orange", "green", "purple", "brown"]
        for i, npc_traj in enumerate(self.npc_trajectories):
            npc_x = [p.x for p in npc_traj]
            npc_y = [p.y for p in npc_traj]
            color = colors[i % len(colors)]
            ax.plot(npc_x, npc_y, color=color, linewidth=1, alpha=0.6, label=f"NPC {i}")

        for lane in range(self.num_lanes + 1):
            y = -self.road_width * self.num_lanes / 2 + lane * self.road_width
            style = "-" if lane in (0, self.num_lanes) else "--"
            ax.axhline(y=y, color="gray", linestyle=style, linewidth=1, alpha=0.5)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(self.title)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
