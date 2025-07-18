from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass
import numpy as np

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
    # interp: list[str] | None = None  # TODO: Add different interpolation type between each two keyframes, shape=(N-1,)


class TrajectoryPlanner(ABC):
    """Abstract base class for trajectory planning strategies."""

    def __init__(
        self,
        move_cfg: ForceClosureCfg | SlideCfg | HingeCfg | KeyframeCfg,
        move_step: int = 10,
    ):
        """
        Initialize trajectory planner.

        Args:
            move_cfg: Movement configuration dataclass
            move_step: Number of steps for movement interpolation
        """
        self.ctrl_qpos = []
        self.ext_fdir = []
        self.ctrl_type = []
        self.move_cfg = move_cfg
        self.move_step = move_step

    def plan_trajectory(
        self,
        init_obj_pose: np.ndarray,
        pregrasp_qpos: np.ndarray,
        grasp_qpos: np.ndarray,
        squeeze_qpos: np.ndarray,
        approach_qpos: np.ndarray | None,
    ) -> tuple[list[np.ndarray], list[str], list[np.ndarray], np.ndarray]:
        """
        Plan the trajectory for the specific task type.

        Returns:
            ctrl_qpos: list of qpos
            ctrl_type: list of control type (ee_pose, joint_angle)
            ext_fdir: list of external direction
            target_obj_pose: Target object pose after movement
        """
        if len(pregrasp_qpos.shape) == 1:
            pregrasp_qpos = pregrasp_qpos[None]
        if len(grasp_qpos.shape) == 1:
            grasp_qpos = grasp_qpos[None]
        if len(squeeze_qpos.shape) == 1:
            squeeze_qpos = squeeze_qpos[None]
        if approach_qpos is not None and len(approach_qpos.shape) == 1:
            approach_qpos = approach_qpos[None]
        self._add_approach_phase(approach_qpos)
        self._add_pregrasp_phase(pregrasp_qpos)
        self._add_grasp_phases(grasp_qpos, squeeze_qpos)

        # Check if movement is needed (for ForceClosure, no movement)
        if self._needs_movement():
            # Generate movement trajectory
            move_pose = self._interp_move_traj(squeeze_qpos[0, :7])
            move_qpos = np.repeat(squeeze_qpos, len(move_pose), axis=0)
            move_qpos[:, :7] = move_pose
            self.ctrl_qpos.extend(list(move_qpos))
            self.ctrl_type.extend(["ee_pose"] * move_qpos.shape[0])

            # Generate object movement trajectory
            move_obj_lst = self._interp_move_traj(init_obj_pose)
            if move_obj_lst is not None and len(move_obj_lst) > 0:
                target_obj_pose = move_obj_lst[-1]
            else:
                target_obj_pose = init_obj_pose

            # Add task-specific external directions
            self.ext_fdir.extend(self._get_external_directions(move_obj_lst))

            # Fill None values in ext_fdir
            self._fill_none_ext_fdir()
        else:
            # No movement needed (e.g., ForceClosure)
            target_obj_pose = init_obj_pose

        return self.ctrl_qpos, self.ctrl_type, self.ext_fdir, target_obj_pose

    def _add_approach_phase(self, approach_qpos: np.ndarray | None):
        """Add approach phase to trajectory if provided."""
        if approach_qpos is not None:
            self.ctrl_qpos.extend(list(approach_qpos))
            n_approach = approach_qpos.shape[0] - 1
            self.ext_fdir.extend([None] * n_approach)
            self.ctrl_type.extend(["joint_angle"] * n_approach)

    def _add_pregrasp_phase(self, pregrasp_qpos: np.ndarray):
        """Add pregrasp phase to trajectory."""
        self.ext_fdir.extend([None] * pregrasp_qpos.shape[0])
        self.ctrl_type.extend(["ee_pose"] * pregrasp_qpos.shape[0])
        self.ctrl_qpos.extend(list(pregrasp_qpos))

    def _add_grasp_phases(self, grasp_qpos: np.ndarray, squeeze_qpos: np.ndarray):
        """Add grasp and squeeze phases to trajectory."""
        # Grasp phase
        self.ext_fdir.extend([None])
        self.ctrl_type.extend(["ee_pose"])
        self.ctrl_qpos.extend(list(grasp_qpos))

        # Squeeze phase
        self.ext_fdir.extend([None])
        self.ctrl_type.extend(["ee_pose"])
        self.ctrl_qpos.extend(list(squeeze_qpos))

    def _fill_none_ext_fdir(self):
        """Fill None values in ext_fdir with the next non-None value."""
        i = len(self.ext_fdir) - 1
        for ext_fdir in reversed(self.ext_fdir):
            if ext_fdir is None:
                self.ext_fdir[i] = self.ext_fdir[i + 1]
            i -= 1

    def _needs_movement(self) -> bool:
        """Check if the task type requires movement after grasp."""
        return not isinstance(self.move_cfg, ForceClosureCfg)

    @abstractmethod
    def _interp_move_traj(self, init_pose: np.ndarray) -> np.ndarray:
        """Interpolate movement trajectory for specific task type."""
        pass

    @abstractmethod
    def _get_external_directions(self, move_obj_lst: np.ndarray) -> list[Any]:
        """Get external directions for the specific task type."""
        pass


class ForceClosurePlanner(TrajectoryPlanner):
    """Trajectory planner for force closure tasks."""

    def __init__(self, move_cfg: ForceClosureCfg, move_step: int = 10):
        super().__init__(move_cfg, move_step)

    def _interp_move_traj(self, init_pose: np.ndarray) -> np.ndarray:
        """No movement for force closure."""
        return np.array([])

    def _get_external_directions(self, move_obj_lst: np.ndarray) -> list[Any]:
        return []


class SlidePlanner(TrajectoryPlanner):
    """Trajectory planner for slide tasks."""

    def __init__(self, move_cfg: SlideCfg, move_step: int = 10):
        super().__init__(move_cfg, move_step)

    def _interp_move_traj(self, init_pose: np.ndarray) -> np.ndarray:
        target_pose = np.copy(init_pose)
        target_pose[:3] += self.move_cfg.axis * self.move_cfg.distance
        return np_interp_slide(init_pose, target_pose, self.move_step)

    def _get_external_directions(self, move_obj_lst: np.ndarray) -> list[Any]:
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

    def _get_external_directions(self, move_obj_lst: np.ndarray) -> list[Any]:
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

    def _get_external_directions(self, move_obj_lst: np.ndarray) -> list[Any]:
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


def get_planner(move_cfg: dict[str, Any], move_step: int = 10) -> TrajectoryPlanner:
    """Get the appropriate planner for the given move configuration."""
    task_type = move_cfg["type"]
    move_cfg["part_name"] = move_cfg.get("part_name", None)
    move_cfg = TASK_TYPE_TO_CFG[task_type](**move_cfg)
    planner = TASK_TYPE_TO_PLANNER[task_type](move_cfg, move_step)
    return planner
