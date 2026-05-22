from dataclasses import dataclass

@dataclass
class EnergyModel:
    initial_j: float = 2.0
    tx_cost_j: float = 0.00042
    rx_cost_j: float = 0.00018
    idle_cost_j: float = 0.000025
    control_cost_j: float = 0.00005
    dead_threshold_j: float = 0.05

    def tx_cost(self, power_dbm: float, link_quality: float) -> float:
        retransmission_factor = 1.0 + max(0.0, 0.7 - link_quality)
        power_factor = 1.0 + max(0.0, power_dbm) / 8.0
        return self.tx_cost_j * retransmission_factor * power_factor

    def rx_cost(self) -> float:
        return self.rx_cost_j

    def idle_cost(self, duty_cycle=1.0):
        import numpy as np
        return self.idle_cost_j * np.clip(duty_cycle, 0.05, 1.0)

    def control_cost(self, n_messages: int = 1) -> float:
        return self.control_cost_j * max(0, n_messages)
