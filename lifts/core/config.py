from dataclasses import dataclass

@dataclass
class LiftsOptions:
    gravity: float = 9.81
    friction: float = 0.1
    max_tension: float = 10000.0
    max_acceleration: float = 1.0
    delta_t: float = 0.1
    state_size: int = 6
    air_density: float = 1.225

@dataclass
class VehicleOptions:
    max_speed: float = 10.0
    max_acceleration: float = 1.0
    mass: float = 1000.0

@dataclass
class PayloadOptions:
    mass: float = 100.0