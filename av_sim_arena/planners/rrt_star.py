"""RRT* (Rapidly-exploring Random Tree Star) planner implementation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from av_sim_arena.planners.base import BasePlanner, PlannerOutput
from av_sim_arena.scenarios.models import VehicleState, Waypoint


@dataclass
class RRTNode:
    """Node in the RRT tree."""

    x: float
    y: float
    heading: float = 0.0
    cost: float = 0.0
    parent: RRTNode | None = None


class RRTStarPlanner(BasePlanner):
    """RRT*-based motion planner for AV navigation."""

    def __init__(
        self,
        max_iterations: int = 200,
        step_size: float = 2.0,
        goal_threshold: float = 3.0,
        search_radius: float = 5.0,
        x_range: tuple[float, float] = (-10.0, 150.0),
        y_range: tuple[float, float] = (-15.0, 15.0),
        seed: int | None = None,
    ):
        super().__init__(name="rrt_star")
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.search_radius = search_radius
        self.x_range = x_range
        self.y_range = y_range
        self.rng = random.Random(seed)

    def plan(
        self,
        ego_state: VehicleState,
        reference_path: list[Waypoint],
        obstacles: list[VehicleState],
        dt: float = 0.1,
    ) -> PlannerOutput:
        if not reference_path:
            return PlannerOutput()

        target = self._find_target(ego_state, reference_path)
        root = RRTNode(x=ego_state.x, y=ego_state.y, heading=ego_state.heading)
        tree = [root]

        best_goal_node = None

        for _ in range(self.max_iterations):
            if self.rng.random() < 0.1:
                sample = RRTNode(x=target.x, y=target.y)
            else:
                sample = RRTNode(
                    x=self.rng.uniform(*self.x_range),
                    y=self.rng.uniform(*self.y_range),
                )

            nearest = self._nearest(tree, sample)
            new_node = self._steer(nearest, sample)

            if self._collision_free(new_node, obstacles):
                neighbors = self._near(tree, new_node)
                new_node.parent = nearest
                new_node.cost = nearest.cost + self._dist(nearest, new_node)

                for neighbor in neighbors:
                    candidate_cost = neighbor.cost + self._dist(neighbor, new_node)
                    if candidate_cost < new_node.cost and self._collision_free_edge(
                        neighbor, new_node, obstacles
                    ):
                        new_node.parent = neighbor
                        new_node.cost = candidate_cost

                tree.append(new_node)

                for neighbor in neighbors:
                    candidate_cost = new_node.cost + self._dist(new_node, neighbor)
                    if candidate_cost < neighbor.cost:
                        neighbor.parent = new_node
                        neighbor.cost = candidate_cost

                if self._dist(new_node, RRTNode(x=target.x, y=target.y)) < self.goal_threshold:
                    if best_goal_node is None or new_node.cost < best_goal_node.cost:
                        best_goal_node = new_node

        if best_goal_node is None:
            return self._fallback_plan(ego_state, target, dt)

        path = self._extract_path(best_goal_node)
        if len(path) < 2:
            return PlannerOutput()

        next_pt = path[1]
        dx = next_pt.x - ego_state.x
        dy = next_pt.y - ego_state.y
        desired_heading = math.atan2(dy, dx)
        steer = self._normalize_angle(desired_heading - ego_state.heading)
        steer = max(-0.5, min(0.5, steer))

        desired_speed = min(10.0, ego_state.speed + 1.0)
        accel = (desired_speed - ego_state.speed) / max(dt, 0.01)
        accel = max(-4.0, min(3.0, accel))

        return PlannerOutput(acceleration=accel, steering=steer)

    def _find_target(self, ego: VehicleState, path: list[Waypoint]) -> Waypoint:
        min_dist = float("inf")
        best_idx = 0
        for i, wp in enumerate(path):
            d = math.sqrt((wp.x - ego.x) ** 2 + (wp.y - ego.y) ** 2)
            if d < min_dist:
                min_dist = d
                best_idx = i
        lookahead = min(best_idx + 5, len(path) - 1)
        return path[lookahead]

    def _nearest(self, tree: list[RRTNode], sample: RRTNode) -> RRTNode:
        return min(tree, key=lambda n: self._dist(n, sample))

    def _steer(self, from_node: RRTNode, to_node: RRTNode) -> RRTNode:
        d = self._dist(from_node, to_node)
        if d < self.step_size:
            return RRTNode(x=to_node.x, y=to_node.y)
        ratio = self.step_size / d
        new_x = from_node.x + ratio * (to_node.x - from_node.x)
        new_y = from_node.y + ratio * (to_node.y - from_node.y)
        heading = math.atan2(new_y - from_node.y, new_x - from_node.x)
        return RRTNode(x=new_x, y=new_y, heading=heading)

    def _near(self, tree: list[RRTNode], node: RRTNode) -> list[RRTNode]:
        return [n for n in tree if self._dist(n, node) < self.search_radius]

    def _collision_free(self, node: RRTNode, obstacles: list[VehicleState]) -> bool:
        for obs in obstacles:
            if math.sqrt((node.x - obs.x) ** 2 + (node.y - obs.y) ** 2) < 3.0:
                return False
        return True

    def _collision_free_edge(
        self, from_node: RRTNode, to_node: RRTNode, obstacles: list[VehicleState]
    ) -> bool:
        steps = max(int(self._dist(from_node, to_node) / 0.5), 1)
        for i in range(steps + 1):
            t = i / steps
            x = from_node.x + t * (to_node.x - from_node.x)
            y = from_node.y + t * (to_node.y - from_node.y)
            for obs in obstacles:
                if math.sqrt((x - obs.x) ** 2 + (y - obs.y) ** 2) < 3.0:
                    return False
        return True

    def _extract_path(self, node: RRTNode) -> list[RRTNode]:
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        return path

    def _fallback_plan(
        self, ego: VehicleState, target: Waypoint, dt: float
    ) -> PlannerOutput:
        dx = target.x - ego.x
        dy = target.y - ego.y
        desired_heading = math.atan2(dy, dx)
        steer = self._normalize_angle(desired_heading - ego.heading)
        steer = max(-0.3, min(0.3, steer))
        return PlannerOutput(acceleration=0.5, steering=steer)

    @staticmethod
    def _dist(a: RRTNode, b: RRTNode) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def reset(self) -> None:
        pass
