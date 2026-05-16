"""Common enums used across the RoadPulse stack."""

from __future__ import annotations

from enum import Enum


class RouteMode(str, Enum):
    """Mode of transport supported by the routing engine."""

    MOTORBIKE = "motorbike"
    CAR = "car"
    TRUCK = "truck"
    BICYCLE = "bicycle"


class VehicleClass(str, Enum):
    """K-anonymised vehicle class bucket used in VETC aggregates."""

    MOTOR = "motor"
    CAR = "car"
    TRUCK = "truck"


class FloodSeverity(str, Enum):
    """Trigger severity used by the parametric insurance trigger feed."""

    LOW = "low"
    MED = "med"
    HIGH = "high"


class EtaConfidence(str, Enum):
    """Categorical confidence label attached to ETA predictions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
