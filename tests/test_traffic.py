"""Tests for multi-agent traffic simulation."""

import math

import pytest

from av_sim_arena.scenarios.models import VehicleState
from av_sim_arena.traffic.behavior_tree import (
    AggressiveDriver,
    BehaviorTree,
    Condition,
    FollowBehavior,
    LaneChangeBehavior,
    NodeStatus,
    Selector,
    Sequence,
    YieldBehavior,
)
from av_sim_arena.traffic.npc import NPCPedestrian, NPCVehicle, PedestrianState


class TestBehaviorTreeNodes:
    def test_follow_behavior_no_lead(self):
        node = FollowBehavior()
        state = VehicleState(x=0, y=0, speed=10)
        ctx = {}
        status = node.tick(state, ctx)
        assert status == NodeStatus.SUCCESS
        assert ctx["acceleration"] == 0.5

    def test_follow_behavior_with_lead(self):
        node = FollowBehavior(desired_gap=10.0)
        state = VehicleState(x=0, y=0, speed=10)
        lead = VehicleState(x=20, y=0, speed=10)
        ctx = {"lead_vehicle": lead}
        status = node.tick(state, ctx)
        assert status == NodeStatus.SUCCESS
        assert ctx["acceleration"] > 0  # gap > desired

    def test_follow_behavior_too_close(self):
        node = FollowBehavior(desired_gap=10.0)
        state = VehicleState(x=0, y=0, speed=10)
        lead = VehicleState(x=5, y=0, speed=10)
        ctx = {"lead_vehicle": lead}
        node.tick(state, ctx)
        assert ctx["acceleration"] < 0  # gap < desired

    def test_yield_behavior_should_yield(self):
        node = YieldBehavior(yield_distance=15.0)
        state = VehicleState(x=0, y=0, speed=10)
        approaching = [VehicleState(x=10, y=0, speed=15)]
        ctx = {"approaching_vehicles": approaching}
        status = node.tick(state, ctx)
        assert status == NodeStatus.SUCCESS
        assert ctx["acceleration"] < 0

    def test_yield_behavior_no_threat(self):
        node = YieldBehavior(yield_distance=15.0)
        state = VehicleState(x=0, y=0, speed=10)
        ctx = {"approaching_vehicles": [VehicleState(x=100, y=0)]}
        status = node.tick(state, ctx)
        assert status == NodeStatus.FAILURE

    def test_lane_change_behavior(self):
        node = LaneChangeBehavior()
        state = VehicleState(x=0, y=0, speed=10)
        ctx = {"lane_change_direction": 1, "dt": 0.1}
        status = node.tick(state, ctx)
        assert status == NodeStatus.RUNNING

    def test_aggressive_driver_no_lead(self):
        node = AggressiveDriver()
        state = VehicleState(x=0, y=0, speed=10)
        ctx = {}
        status = node.tick(state, ctx)
        assert status == NodeStatus.SUCCESS
        assert ctx["acceleration"] > 0

    def test_sequence_all_success(self):
        n1 = FollowBehavior()
        n2 = FollowBehavior()
        seq = Sequence([n1, n2])
        state = VehicleState()
        ctx = {}
        assert seq.tick(state, ctx) == NodeStatus.SUCCESS

    def test_selector_first_success(self):
        n1 = FollowBehavior()
        n2 = YieldBehavior()
        sel = Selector([n1, n2])
        state = VehicleState()
        ctx = {}
        assert sel.tick(state, ctx) == NodeStatus.SUCCESS

    def test_condition_true(self):
        cond = Condition("flag", lambda v: v is True)
        state = VehicleState()
        assert cond.tick(state, {"flag": True}) == NodeStatus.SUCCESS

    def test_condition_false(self):
        cond = Condition("flag", lambda v: v is True)
        state = VehicleState()
        assert cond.tick(state, {"flag": False}) == NodeStatus.FAILURE


class TestNPCVehicle:
    def test_creation(self):
        state = VehicleState(x=10, y=0, speed=10)
        npc = NPCVehicle("npc_0", state, behavior="follow")
        assert npc.vehicle_id == "npc_0"

    def test_step(self):
        state = VehicleState(x=0, y=0, speed=10, heading=0)
        npc = NPCVehicle("npc_0", state, behavior="follow")
        ctx = {}
        new_state = npc.step(ctx, dt=0.1)
        assert new_state.x > 0
        assert len(npc.history) == 2

    def test_behaviors(self):
        for behavior in ["follow", "yield", "lane_change", "aggressive", "default"]:
            state = VehicleState(x=0, y=0, speed=10)
            npc = NPCVehicle(f"npc_{behavior}", state, behavior=behavior)
            assert npc.behavior_tree is not None


class TestNPCPedestrian:
    def test_creation_normal(self):
        state = PedestrianState(x=0, y=0)
        ped = NPCPedestrian("ped_0", state, behavior="normal")
        assert ped.state.speed == 1.4

    def test_creation_jaywalking(self):
        state = PedestrianState(x=0, y=0)
        ped = NPCPedestrian("ped_0", state, behavior="jaywalking")
        assert ped.state.speed == 1.8
        assert abs(ped.state.heading - math.pi / 2) < 0.01

    def test_creation_running(self):
        state = PedestrianState(x=0, y=0)
        ped = NPCPedestrian("ped_0", state, behavior="running")
        assert ped.state.speed == 3.0

    def test_step(self):
        state = PedestrianState(x=0, y=0, heading=0, speed=1.4)
        ped = NPCPedestrian("ped_0", state)
        ped.step(dt=1.0)
        assert ped.state.x == pytest.approx(1.4, abs=0.01)
        assert len(ped.history) == 2
