from __future__ import annotations

import math
import numpy as np
import re
import sys
import vtk_convenience as conv

from csv import DictWriter
from copy import copy
from dataclasses import dataclass
from enum import IntEnum, auto
from functools import reduce
from typing import Dict, Tuple

from scipy.interpolate import PchipInterpolator
from vtk import vtkPolyData


class Endplate(IntEnum):
    LOWER = 0
    UPPER = 1

    @classmethod
    def options(cls):
        return cls.LOWER, cls.UPPER


class Spine:
    VERTEBRAE = (
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
        "T9",
        "T10",
        "T11",
        "T12",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
    )

    def __init__(
        self,
        geomemtries: vtkPolyData,
        lateral_axis: np.ndarray,
        slice_thickness: float,
        max_angle: float,
    ) -> None:
        local_up = UpApproximator(geomemtries)
        self.vertebrae = [
            Vertebra(
                g,
                lateral_axis=lateral_axis,
                up_approximator=local_up,
                slice_thickness=slice_thickness,
                max_angle=max_angle,
            )
            for g in geomemtries
        ]

    def name_vertebrae(self, offset_to_c1: int) -> None:
        for name, data in zip(self.VERTEBRAE[offset_to_c1:], self.vertebrae):
            setattr(self, name, data)

    @property
    def angles(self) -> Tuple[float]:
        return [v1.angle(v2) for v1, v2 in zip(self.vertebrae, self.vertebrae[1:])]

    @property
    def named_angles(self) -> Dict[str, float]:
        return {
            f"{first}/{second}": getattr(self, first).angle(getattr(self, second))
            for first, second in zip(self.VERTEBRAE, self.VERTEBRAE[1:])
            if hasattr(self, first) and hasattr(self, second)
        }

    @property
    def names(self) -> List[str]:
        return [name for name in cls.VERTEBRAE if hasattr(self, name)]

    def __getitem__(self, index: int) -> Vertebra:
        return self.vertebrae[index]

    def __len__(self) -> int:
        return len(self.vertebrae)

    @staticmethod
    def write(spine: Spine, filename: str) -> None:
        angles = spine.named_angles
        if not angles:
            angles = dict(enumerate(spine.angles))

        with open(filename, "w", newline="") as csv_file:
            writer = DictWriter(csv_file, fieldnames=angles.keys())

            writer.writeheader()
            writer.writerow(angles)

    @classmethod
    def offset_from_filename(cls, filename: str):
        vertebra_regex = re.compile("[CTL]\d{1,2}", re.IGNORECASE)
        match = vertebra_regex.search(filename)
        if not match:
            return None

        return cls.VERTEBRAE.index(match.group().upper())

    @classmethod
    def generate_headers(cls) -> List[str]:
        return [
            f"{first}/{second}"
            for first, second in zip(cls.VERTEBRAE, cls.VERTEBRAE[1:])
        ]


@dataclass
class Orientation:
    up: np.ndarray
    down: np.ndarray
    left: np.ndarray
    right: np.ndarray
    front: np.ndarray
    back: np.ndarray

    center: np.ndarray
    width: float


@dataclass
class Body:
    center_portion: vtkPolyData
    endplates: vtkPolyData
    curves: Tuple[vtkPolyData, vtkPolyData]
    regressions: Tuple[np.ndarray, np.ndarray]

    @dataclass
    class Height:
        near: float
        far: float

    @property
    def minmax(self):
        """
        Return tuple
        (
         (
          min_coordinate_lower_endplate,
          max_coordinate_lower_endplate,
         ),
         (
          min_coordinate_upper_endplate,
          max_coordinate_upper_endplate,
         ),
        )
        """
        return tuple(self._minmax(e) for e in Endplate.options())

    @property
    def height(self) -> Tuple[Body.Height, Body.Height]:
        " Return tuple of posterior and anterior vertebra body height, respectively. "
        minmax = self.minmax

        near_height = minmax[Endplate.UPPER][0]- minmax[Endplate.LOWER][0]
        far_height = minmax[Endplate.UPPER][1]- minmax[Endplate.LOWER][1]

        return self.Height(np.linalg.norm(near_height), np.linalg.norm(far_height))

    def _minmax(self, endplate: Endplate):
        curve = self.curves[endplate]
        curve = np.array([
            curve.GetPoint(id_)
            for id_ in range(curve.GetNumberOfPoints())
        ])
        distances = curve.dot(self.regressions[endplate])
        return (
            curve[distances.argmin()],
            curve[distances.argmax()],
        )
        
class Vertebra:
    def __init__(
        self,
        geometry: vtkPolyData,
        lateral_axis: np.ndarray,
        up_approximator: UpApproximator,
        slice_thickness: float,
        max_angle: float,
    ) -> None:
        self.geometry = geometry
        self.orientation = Vertebra._calc_orientation(
            geometry,
            up_approximator=up_approximator,
            approx_lateral_axis=lateral_axis,
        )

        # TODO clip_plane "plane_normal" param seems inverted
        vertebra_without_appendix = conv.clip_plane(
            geometry,
            plane_origin=self.orientation.center,
            plane_normal=self.orientation.front,
        )

        self.body = Vertebra._extract_body(
            vertebra_without_appendix,
            orientation=self.orientation,
            width=slice_thickness,
            max_angle=max_angle,
        )

        self.body_laterally = Vertebra._extract_body(
            vertebra_without_appendix,
            orientation=self.orientation,
            width=slice_thickness,
            max_angle=max_angle,
            laterally=True,
            center=np.array(
                conv.calc_center_of_mass(vertebra_without_appendix)
            ),
        )

        lower_minmax, upper_minmax = self.body.minmax
        frontal_plane_points = (
            (lower_minmax[0] + lower_minmax[1]) / 2.0,
            (upper_minmax[0] + upper_minmax[1]) / 2.0,
        )

        self.center = (
            np.array(conv.cut_plane(
                self.body.curves[Endplate.LOWER],
                plane_origin=frontal_plane_points[Endplate.LOWER],
                plane_normal=self.orientation.front,
            ).GetPoint(0)),
            np.array(conv.cut_plane(
                self.body.curves[Endplate.UPPER],
                plane_origin=frontal_plane_points[Endplate.UPPER],
                plane_normal=self.orientation.front,
            ).GetPoint(0)),
        ) 

    def angle(self, other: Vertebra):
        rotation_axis = conv.normalize(self.orientation.right)
        this_regression = conv.normalize(self.body.regressions[Endplate.UPPER])
        other_regression = conv.normalize(other.body.regressions[Endplate.UPPER])

        return math.degrees(
            np.arctan2(
                np.cross(other_regression, this_regression).dot(rotation_axis),
                this_regression.dot(other_regression),
            )
        )

    @staticmethod
    def _calc_orientation(
        geometry: vtkPolyData,
        up_approximator: UpApproximator,
        approx_lateral_axis: np.ndarray,
    ) -> Orientation:
        oriented_bounding_box = conv.calc_obb(geometry)[1:]
        center_of_mass = np.array(
            conv.calc_center_of_mass(geometry)
        )  # TODO make calc_center_of_mass return ndarray

        right, flip = conv.closest_vector(oriented_bounding_box, approx_lateral_axis)
        width = np.linalg.norm(right)
        right = conv.normalize(flip * np.array(right))
        up = up_approximator(center_of_mass)
        front = conv.normalize(np.cross(up, right))

        return Orientation(
            up=up,
            down=-up,
            right=right,
            left=-right,
            front=front,
            back=-front,
            center=center_of_mass,
            width=width,
        )

    @staticmethod
    def _extract_body(
        body: vtkPolyData,
        orientation: Orientation,
        width: float,
        max_angle: float,
        laterally: bool=False,
        center: np.ndarray=None,
    ) -> Body:
        if not isinstance(center, np.ndarray):
            center = orientation.center
        else:
            orientation = copy(orientation)
            orientation.center = center

        center_portion = Vertebra._extract_center(
            body, orientation=orientation, width=width, laterally=laterally
        )
        endplates = conv.eliminate_misaligned_faces(
            center_portion, direction=orientation.up, max_angle=max_angle
        )

        if laterally:
            cut_direction = orientation.front
        else:
            cut_direction = orientation.right
        curves = conv.cut_plane(
            endplates,
            plane_origin=center,
            plane_normal=cut_direction,
        )
        curves = (
            conv.clip_plane(
                curves,
                plane_origin=center,
                plane_normal=orientation.down,
            ),
            conv.clip_plane(
                curves,
                plane_origin=center,
                plane_normal=orientation.up,
            ),
        )
        if laterally:
            direction_of_interest = orientation.right
        else:
            direction_of_interest = orientation.front
        regressions = [conv.normalize(calc_main_component(s)) for s in curves]
        regressions = [
            -direction if direction.dot(direction_of_interest) < 0 else direction
            for direction in regressions
        ]

        return Body(
            center_portion=center_portion,
            endplates=endplates,
            curves=curves,
            regressions=regressions,
        )

    @staticmethod
    def _extract_center(
        body: vtkPolyData, orientation: Orientation, width: float, laterally: bool=False
    ) -> vtkPolyData:
        width = width * orientation.width / 2.0
        if laterally:
            first_cut_direction = orientation.front
            second_cut_direction = orientation.back
        else:
            first_cut_direction = orientation.right
            second_cut_direction = orientation.left
        body_wo_right = Vertebra._clip_offset(
            body, center=orientation.center, direction=first_cut_direction, offset=width
        )
        body_wo_right_and_left = Vertebra._clip_offset(
            body_wo_right,
            center=orientation.center,
            direction=second_cut_direction,
            offset=width,
        )
        return body_wo_right_and_left

        if laterally:
            clipping_axes = orientation.front, orientation.back
        else:
            clipping_axes = orientation.right, orientation.left
        return reduce(
            lambda remaining_body, clipping_direction: Vertebra._clip_offset(
                remaining_body,
                center=orientation.center,
                direction=clipping_direction,
                offset=width,
            ),
            clipping_axes,
            body,
        )

    @staticmethod
    def _clip_offset(
        geometry: vtkPolyData, center: np.ndarray, direction: np.ndarray, offset: float
    ) -> vtkPolyData:
        offset_origin = center + offset * direction
        return conv.clip_plane(
            geometry, plane_origin=offset_origin, plane_normal=-direction
        )


class UpApproximator:
    """
    For a set of vertebra geoemtries, guess the most probable up-vector.
    This is achieved by calcuating the Pchip interpolation through all vertebra geometries' centers
    of mass (COM). The up-vector is then approximated by a secant along this Pchip curve.

    Usage:
        # create list of vertebra geometries
        vertebrae_stl_file_list = ["L1.stl", "L2.stl", "T12.stl"]
        vertebrae_geo = [load_stl(stl) for stl in vertebrae_stl_file_list]

        # initialize the approximator
        approximator = UpApproximator(vertebrae_geo)

        # query up-vector at any spot (here: center of mass for T12)
        com_T12 = calc_center_of_mass(vertebrae_geo[2])
        local_up = approximator(com_T12)
    """

    def __init__(self, geomemtries: vtkPolyData) -> None:
        centers_of_mass = np.array([conv.calc_center_of_mass(g) for g in geomemtries])
        self.most_significant_column = self.column_with_widest_spread(centers_of_mass)
        centers_of_mass = self.sort_by_column(
            centers_of_mass, column=self.most_significant_column
        )

        interpolator = PchipInterpolator(
            centers_of_mass[:, self.most_significant_column],
            centers_of_mass,
            extrapolate=True,
        )
        self.derivative = interpolator.derivative()

    @classmethod
    def column_with_widest_spread(cls, array: np.ndarray) -> int:
        """
        For a two-dimensional array, find the column with the widest range in value.
        """
        max_positions = np.amax(array, axis=0)
        min_positions = np.amin(array, axis=0)
        spread = max_positions - min_positions
        return spread.tolist().index(max(spread))

    @classmethod
    def sort_by_column(cls, array: np.ndarray, column: int) -> np.ndarray:
        """
        For a two-dimensional array, select a column to sort the array by.
        """
        array = array.copy()
        return array[array[:, column].argsort()]

    def __call__(self, position: np.ndarray) -> np.ndarray:
        """
        Get the most probably up-vector for a 3D position.
        The approximator only takes one axis into consideration, height.
        The height is assumed to be the axis with the widest range in value.

        The result is calculated by a secant, with the two intersections being
        distance 'window_size' apart..

        Keyword arguments:
            position: numpy.ndarray to calculate the local up-vector for.
            window_size: the distance of secant intersections.
        """
        return conv.normalize(
            self.derivative((position[self.most_significant_column],))[0]
        )


def calc_main_component(geometry: vtk.vtkPolyData):
    """
    Return the main component of singular value decomposition through
    all vertices of a vtkPolyData object.
    """
    points = list(conv.iter_points(geometry))
    points = np.array(points)
    mean = points.mean(axis=0)
    _1, _2, eigenvector = np.linalg.svd(points - mean)
    return eigenvector[0]
