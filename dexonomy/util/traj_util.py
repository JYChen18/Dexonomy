import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from dexonomy.util.np_util import (
    np_get_relative_pose,
    np_normalize_vector,
    np_interp_hinge,
    np_interp_slide,
    np_multiply_pose,
)


@dataclass
class MoveCfg:
    """Configuration for movement tasks."""

    type: str
    obj_name: str
    part_name: str


@dataclass
class ForceClosureCfg(MoveCfg):
    """Configuration for force closure tasks."""

    pass


@dataclass
class SlideCfg(MoveCfg):
    """Configuration for slide tasks."""

    axis: np.ndarray  # Direction vector for sliding movement, shape=(3,)
    distance: float  # Distance to slide along the axis


@dataclass
class HingeCfg(MoveCfg):
    """Configuration for hinge tasks."""

    pos: np.ndarray  # Position of the hinge point, shape=(3,)
    axis: np.ndarray  # Rotation axis vector, shape=(3,)
    distance: float  # Rotation angle in radians


@dataclass
class KeyframeCfg(MoveCfg):
    """Configuration for keyframe pose tasks."""

    pose: np.ndarray  # Pose of the keyframe, shape=(N, 7)
    # interp: List[str] | None = None  # TODO: Add different interpolation type between each two keyframes, shape=(N-1,)


class TrajectoryPlanner(ABC):
    """Abstract base class for trajectory planning strategies."""

    def __init__(
        self,
        move_cfg: Union[ForceClosureCfg, SlideCfg, HingeCfg, KeyframeCfg],
        move_step: int = 10,
    ):
        """
        Initialize trajectory planner.

        Args:
            move_cfg: Movement configuration dataclass
            move_step: Number of steps for movement interpolation
        """
        self.qpos_lst = []
        self.extdir_lst = []
        self.ctype_lst = []
        self.move_cfg = move_cfg
        self.move_step = move_step

    def plan_trajectory(
        self,
        init_obj_pose: np.ndarray,
        pregrasp_qpos: np.ndarray,
        grasp_qpos: np.ndarray,
        squeeze_qpos: np.ndarray,
        approach_qpos: Optional[np.ndarray],
    ) -> tuple[List[np.ndarray], List[str], List[np.ndarray], np.ndarray]:
        """
        Plan the trajectory for the specific task type.

        Returns:
            qpos_lst: List of qpos
            ctype_lst: List of control type (ee_pose, joint_angle)
            extdir_lst: List of external direction
            target_obj_pose: Target object pose after movement
        """
        self._add_approach_phase(approach_qpos)
        self._add_pregrasp_phase(pregrasp_qpos)
        self._add_grasp_phases(grasp_qpos, squeeze_qpos)

        # Check if movement is needed (for ForceClosure, no movement)
        if self._needs_movement():
            # Generate movement trajectory
            move_pose = self._interp_move_traj(squeeze_qpos[0, :7])
            move_qpos = np.repeat(squeeze_qpos, len(move_pose), axis=0)
            move_qpos[:, :7] = move_pose
            self.qpos_lst.extend(list(move_qpos))
            self.ctype_lst.extend(["ee_pose"] * move_qpos.shape[0])

            # Generate object movement trajectory
            move_obj_lst = self._interp_move_traj(init_obj_pose)
            if move_obj_lst is not None and len(move_obj_lst) > 0:
                target_obj_pose = move_obj_lst[-1]
            else:
                target_obj_pose = init_obj_pose

            # Add task-specific external directions
            self.extdir_lst.extend(self._get_external_directions(move_obj_lst))

            # Fill None values in extdir_lst
            self._fill_none_extdir()
        else:
            # No movement needed (e.g., ForceClosure)
            target_obj_pose = init_obj_pose

        return (
            self.qpos_lst,
            self.ctype_lst,
            self.extdir_lst,
            target_obj_pose,
        )

    def _add_approach_phase(self, approach_qpos: Optional[np.ndarray]):
        """Add approach phase to trajectory if provided."""
        if approach_qpos is not None:
            self.qpos_lst.extend(list(approach_qpos))
            approach_num = approach_qpos.shape[0] - 1
            self.extdir_lst.extend([None] * approach_num)
            self.ctype_lst.extend(["joint_angle"] * approach_num)

    def _add_pregrasp_phase(self, pregrasp_qpos: np.ndarray):
        """Add pregrasp phase to trajectory."""
        self.extdir_lst.extend([None] * pregrasp_qpos.shape[0])
        self.ctype_lst.extend(["ee_pose"] * pregrasp_qpos.shape[0])
        self.qpos_lst.extend(list(pregrasp_qpos))

    def _add_grasp_phases(self, grasp_qpos: np.ndarray, squeeze_qpos: np.ndarray):
        """Add grasp and squeeze phases to trajectory."""
        # Grasp phase
        self.extdir_lst.extend([None])
        self.ctype_lst.extend(["ee_pose"])
        self.qpos_lst.extend(list(grasp_qpos))

        # Squeeze phase
        self.extdir_lst.extend([None])
        self.ctype_lst.extend(["ee_pose"])
        self.qpos_lst.extend(list(squeeze_qpos))

    def _fill_none_extdir(self):
        """Fill None values in extdir_lst with the next non-None value."""
        i = len(self.extdir_lst) - 1
        for extdir in reversed(self.extdir_lst):
            if extdir is None:
                self.extdir_lst[i] = self.extdir_lst[i + 1]
            i -= 1

    def _needs_movement(self) -> bool:
        """Check if the task type requires movement after grasp."""
        return not isinstance(self.move_cfg, ForceClosureCfg)

    @abstractmethod
    def _interp_move_traj(self, init_pose: np.ndarray) -> np.ndarray:
        """Interpolate movement trajectory for specific task type."""
        pass

    @abstractmethod
    def _get_external_directions(self, move_obj_lst: np.ndarray) -> List[Any]:
        """Get external directions for the specific task type."""
        pass


class ForceClosurePlanner(TrajectoryPlanner):
    """Trajectory planner for force closure tasks."""

    def __init__(self, move_cfg: ForceClosureCfg, move_step: int = 10):
        super().__init__(move_cfg, move_step)

    def _interp_move_traj(self, init_pose: np.ndarray) -> np.ndarray:
        """No movement for force closure."""
        return np.array([])

    def _get_external_directions(self, move_obj_lst: np.ndarray) -> List[Any]:
        return []


class SlidePlanner(TrajectoryPlanner):
    """Trajectory planner for slide tasks."""

    def __init__(self, move_cfg: SlideCfg, move_step: int = 10):
        super().__init__(move_cfg, move_step)

    def _interp_move_traj(self, init_pose: np.ndarray) -> np.ndarray:
        target_pose = np.copy(init_pose)
        target_pose[:3] += self.move_cfg.axis * self.move_cfg.distance
        return np_interp_slide(init_pose, target_pose, self.move_step)

    def _get_external_directions(self, move_obj_lst: np.ndarray) -> List[Any]:
        return [-self.move_cfg.axis] * len(move_obj_lst)


class HingePlanner(TrajectoryPlanner):
    """Trajectory planner for hinge tasks."""

    def __init__(self, move_cfg: HingeCfg, move_step: int = 10):
        super().__init__(move_cfg, move_step)

    def _interp_move_traj(self, init_pose: np.ndarray) -> np.ndarray:
        return np_interp_hinge(
            pose1=init_pose,
            hinge_pos=self.move_cfg.pos,
            hinge_axis=self.move_cfg.axis,
            move_angle=self.move_cfg.distance,
            step=self.move_step,
        )

    def _get_external_directions(self, move_obj_lst: np.ndarray) -> List[Any]:
        return [
            -np.cross(
                self.move_cfg.axis, np_normalize_vector(p[:3] - self.move_cfg.pos)
            )
            for p in move_obj_lst
        ]


class KeyframePlanner(TrajectoryPlanner):
    """Trajectory planner for keyframe tasks."""

    def __init__(self, move_cfg: KeyframeCfg, move_step: int = 10):
        super().__init__(move_cfg, move_step)

    def _interp_move_traj(self, init_pose: np.ndarray) -> np.ndarray:
        relative_pose = np_get_relative_pose(self.move_cfg.pose[0], init_pose)
        prev_hand_pose = init_pose
        pose_lst = [init_pose[None]]
        for obj_pose in self.move_cfg.pose[1:]:
            hand_pose = np_multiply_pose(obj_pose, relative_pose)
            pose_lst.append(np_interp_slide(prev_hand_pose, hand_pose, self.move_step))
            prev_hand_pose = hand_pose
        return np.concatenate(pose_lst, axis=0)

    def _get_external_directions(self, move_obj_lst: np.ndarray) -> List[Any]:
        return [np.array([0, 0, -1.0])] * len(move_obj_lst)


TASK_TYPE_TO_CFG = {
    "force_closure": ForceClosureCfg,
    "slide": SlideCfg,
    "hinge": HingeCfg,
    "keyframe": KeyframeCfg,
}

TASK_TYPE_TO_PLANNER = {
    "force_closure": ForceClosurePlanner,
    "slide": SlidePlanner,
    "hinge": HingePlanner,
    "keyframe": KeyframePlanner,
}


def get_planner(move_cfg: Dict[str, Any], move_step: int = 10) -> TrajectoryPlanner:
    """Get the appropriate planner for the given move configuration."""
    task_type = move_cfg["type"]
    move_cfg["part_name"] = move_cfg.get("part_name", None)
    move_cfg = TASK_TYPE_TO_CFG[task_type](**move_cfg)
    planner = TASK_TYPE_TO_PLANNER[task_type](move_cfg, move_step)
    return planner
