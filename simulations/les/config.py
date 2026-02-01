
from dataclasses import dataclass

@dataclass
class LESConfig:
    """Configuration for Large Eddy Simulation subgrid-scale models."""
    sgs_model: str = "smagorinsky"  # 'smagorinsky', 'dynamic' (future)
    cs: float = 0.12                 # Smagorinsky constant (0.1-0.2)
    wall_damping: bool = True        # Apply Van Driest damping near walls
    a_plus: float = 25.0             # Van Driest constant
    
    # For future extension:
    dynamic_sgs: bool = False
    filter_type: str = "grid"        # 'grid', 'explicit'
