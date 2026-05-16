"""Routing engine.

A small Dijkstra implementation that consumes :class:`~roadpulse_routing.graph.Graph`
and a per-mode :class:`~roadpulse_routing.profiles.Profile`. Edge penalties (flood,
congestion, eco) are pulled out of a swappable :class:`EdgePenaltyLookup` so callers
can plug Redis, an in-memory dict, or a feature-store client.

The implementation deliberately mirrors how the production OSRM Lua profile blends
weights, so unit tests that exercise the Python path also validate the formula
applied by the native engine.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from roadpulse_core.geo import Coordinate, encode_polyline, haversine_m
from roadpulse_core.types import RouteMode

from roadpulse_routing.graph import Edge, Graph
from roadpulse_routing.profiles import PROFILES, Profile


class EdgePenaltyLookup(Protocol):
    """Read-only view onto current edge penalty signals.

    Returns values in ``[0, 1]`` where 0 = no contribution and 1 = maximal slowdown.
    """

    def congestion(self, edge: Edge) -> float: ...
    def flood(self, edge: Edge) -> float: ...
    def eco(self, edge: Edge) -> float: ...


class ZeroPenalty:
    """No-op penalty lookup used in tests and offline backtests."""

    def congestion(self, edge: Edge) -> float:  # noqa: ARG002
        return 0.0

    def flood(self, edge: Edge) -> float:  # noqa: ARG002
        return 0.0

    def eco(self, edge: Edge) -> float:  # noqa: ARG002
        return 0.0


@dataclass(slots=True)
class StaticPenalty:
    """Static dict-backed penalty lookup keyed by H3 hex id stored on the edge tags."""

    congestion_by_hex: dict[str, float] = field(default_factory=dict)
    flood_by_hex: dict[str, float] = field(default_factory=dict)
    eco_by_class: dict[str, float] = field(default_factory=dict)

    def congestion(self, edge: Edge) -> float:
        return self.congestion_by_hex.get(edge.tags.get("hex_id", ""), 0.0)

    def flood(self, edge: Edge) -> float:
        return self.flood_by_hex.get(edge.tags.get("hex_id", ""), 0.0)

    def eco(self, edge: Edge) -> float:
        return self.eco_by_class.get(edge.road_class, 0.0)


# --- API surface ----------------------------------------------------------------------


class RouteRequest(BaseModel):
    """Public route request (mirrors OpenAPI ``POST /v1/route``)."""

    model_config = ConfigDict(populate_by_name=True)

    origin: tuple[float, float]
    destination: tuple[float, float]
    mode: RouteMode = RouteMode.MOTORBIKE
    profile: str = "vn"
    flood_aware: bool = True
    eco: bool = False
    depart_at: str | None = None


class Step(BaseModel):
    """A single step in a route response."""

    edge_idx: int
    distance_m: float
    duration_s: float
    road_class: str
    flood_score: float = 0.0
    congestion_score: float = 0.0


class Route(BaseModel):
    """A single computed route candidate."""

    name: str
    distance_m: float
    duration_s: float
    flood_score: float
    congestion_score: float
    eco_score: float
    toll_estimate_vnd: int
    geometry: str  # encoded polyline (precision 5)
    steps: list[Step] = Field(default_factory=list)


class RouteResponse(BaseModel):
    """Public route response — exactly three candidates: fast, safe, eco."""

    routes: list[Route]
    request: RouteRequest


# --- Engine ---------------------------------------------------------------------------


class RoutingEngine:
    """Dijkstra-based routing engine with custom penalty blending."""

    def __init__(self, graph: Graph, penalties: EdgePenaltyLookup | None = None) -> None:
        self._graph = graph
        self._penalties = penalties or ZeroPenalty()

    @property
    def graph(self) -> Graph:
        return self._graph

    def shortest(
        self,
        origin_id: int,
        destination_id: int,
        *,
        profile: Profile,
        flood_aware: bool = True,
        eco: bool = False,
    ) -> Route:
        """Run Dijkstra with the blended-cost edge function."""
        dist: dict[int, float] = {origin_id: 0.0}
        prev: dict[int, tuple[int, int]] = {}  # node_id -> (prev_node_id, edge_idx)
        pq: list[tuple[float, int]] = [(0.0, origin_id)]
        visited: set[int] = set()
        beta = profile.beta_flood if flood_aware else 0.0
        gamma = profile.gamma_eco if eco else 0.0
        alpha = profile.alpha_congestion
        while pq:
            cost, node_id = heapq.heappop(pq)
            if node_id in visited:
                continue
            visited.add(node_id)
            if node_id == destination_id:
                break
            for edge_idx in self._graph.outgoing(node_id):
                edge = self._graph.edge(edge_idx)
                if not profile.is_usable(edge.road_class, edge.tags):
                    continue
                base = edge.free_flow_seconds
                if not isfinite(base) or base <= 0:
                    continue
                penalty = (
                    1.0
                    + alpha * self._penalties.congestion(edge)
                    + beta * self._penalties.flood(edge)
                    + gamma * self._penalties.eco(edge)
                )
                step_cost = base * penalty
                new_cost = cost + step_cost
                if new_cost < dist.get(edge.dst, float("inf")):
                    dist[edge.dst] = new_cost
                    prev[edge.dst] = (node_id, edge_idx)
                    heapq.heappush(pq, (new_cost, edge.dst))
        if destination_id not in dist:
            raise LookupError("destination is unreachable from origin under this profile")
        return self._reconstruct(origin_id, destination_id, prev, profile)

    def three_candidates(
        self,
        origin_id: int,
        destination_id: int,
        *,
        mode: RouteMode = RouteMode.MOTORBIKE,
    ) -> list[Route]:
        """Return the ``fast`` / ``safe`` / ``eco`` candidates side-by-side."""
        profile = PROFILES[mode]
        fast = self.shortest(
            origin_id,
            destination_id,
            profile=profile,
            flood_aware=False,
            eco=False,
        )
        fast.name = "fast"
        safe = self.shortest(
            origin_id,
            destination_id,
            profile=profile,
            flood_aware=True,
            eco=False,
        )
        safe.name = "safe"
        eco = self.shortest(
            origin_id,
            destination_id,
            profile=profile,
            flood_aware=True,
            eco=True,
        )
        eco.name = "eco"
        return [fast, safe, eco]

    # --- helpers ----------------------------------------------------------------------

    def _reconstruct(
        self,
        origin_id: int,
        destination_id: int,
        prev: dict[int, tuple[int, int]],
        profile: Profile,
    ) -> Route:
        steps: list[Step] = []
        node_id = destination_id
        coordinates: list[Coordinate] = []
        total_flood = 0.0
        total_congestion = 0.0
        total_distance = 0.0
        total_duration = 0.0
        toll_vnd = 0
        while node_id != origin_id:
            prev_node, edge_idx = prev[node_id]
            edge = self._graph.edge(edge_idx)
            flood = self._penalties.flood(edge)
            cong = self._penalties.congestion(edge)
            steps.append(
                Step(
                    edge_idx=edge_idx,
                    distance_m=edge.distance_m,
                    duration_s=edge.free_flow_seconds
                    * (1 + profile.alpha_congestion * cong + profile.beta_flood * flood),
                    road_class=edge.road_class,
                    flood_score=flood,
                    congestion_score=cong,
                )
            )
            total_distance += edge.distance_m
            total_duration += steps[-1].duration_s
            total_flood = max(total_flood, flood)
            total_congestion = max(total_congestion, cong)
            toll_vnd += int(edge.tags.get("toll_vnd", "0") or "0")
            coordinates.append(self._graph.node(prev_node).as_coordinate())
            node_id = prev_node
        coordinates.reverse()
        # Append the destination so the polyline closes.
        coordinates.append(self._graph.node(destination_id).as_coordinate())
        eco_score = total_distance / 1_000.0 * profile.eco_factor
        return Route(
            name="route",
            distance_m=total_distance,
            duration_s=total_duration,
            flood_score=round(total_flood, 3),
            congestion_score=round(total_congestion, 3),
            eco_score=round(eco_score, 3),
            toll_estimate_vnd=toll_vnd,
            geometry=encode_polyline(coordinates),
            steps=list(reversed(steps)),
        )


def total_geometry_length(coords: Iterable[Coordinate]) -> float:
    """Sum great-circle distances over a polyline; useful for backtests."""
    coords = list(coords)
    total = 0.0
    for a, b in zip(coords, coords[1:], strict=False):
        total += haversine_m(a, b)
    return total
