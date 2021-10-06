class Orbit:
    def __init__(self):
        pass

    def propagate(self, timestamp: int) -> "PropagatedOrbit":
        return PropagatedOrbit(self)


class PropagatedOrbit:
    def __init__(self, orbit: Orbit):
        self.orbit = orbit
