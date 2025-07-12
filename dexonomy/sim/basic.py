from dataclasses import dataclass


@dataclass
class HandCfg:
    xml_path: str
    freejoint: bool = False
    arm_flag: bool = False
    ee_name: str | None = None  # only useful when arm_flag=True
    skip_table_cbody: list[str] | None = None  # only useful when arm_flag=True


@dataclass
class SimCfg:
    timestep: float = 0.002
    obj_margin: float = 0.0
    hand_margin: float = 0.0
    plane_margin: float = 0.0
    hand_prefix: str = "hand-"
    obj_prefix: str = "obj-"
    plane_prefix: str = "plane-"
    obj_freejoint: bool = True
    miu_coef: tuple[float, float] = (0.0, 0.0)
