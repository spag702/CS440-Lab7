from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from interactive_markers import InteractiveMarkerServer, MenuHandler
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
)


@dataclass  # (Think of Python dataclasses like structs in C/C++)
class Waypoint:
    """Represents a waypoint."""

    x: float
    y: float
    is_key: bool
    friction: float = 0.5
    v: float = 0.0

    @classmethod
    def from_row_dict(cls, d: dict[str, str]) -> Waypoint:
        """Converts from a dictionary mapping names to string representations
        of values."""
        x = float(d["x"])
        y = float(d["y"])
        is_key = convert_str_to_bool(d["is_key"])
        friction = float(d["friction"]) if "friction" in d else 0.5
        v = float(d["v"]) if "v" in d else 0.0
        return cls(x=x, y=y, is_key=is_key, friction=friction, v=v)


def load_waypoints(waypoints_file_path: Path) -> list[Waypoint]:
    """Loads waypoints from a file."""

    waypoints: list[Waypoint] = []
    with open(str(waypoints_file_path)) as f:
        reader = csv.DictReader(f)
        for row_dict in reader:
            waypoints.append(Waypoint.from_row_dict(row_dict))
    return waypoints


def save_waypoints(waypoints: list[Waypoint], waypoints_file_path: Path) -> None:
    """Save waypoints to a file."""
    with open(str(waypoints_file_path), "w") as f:
        writer = csv.DictWriter(f, fieldnames=[fd.name for fd in fields(Waypoint)])
        writer.writeheader()
        for waypoint in waypoints:
            writer.writerow(asdict(waypoint))


def convert_str_to_bool(s: str) -> bool:
    """Helper function to convert a string to a boolean."""
    if s == "True":
        return True
    elif s == "False":
        return False
    raise ValueError(f"Value must either be 'True' or 'False', got: {s!r}")


def optimize_path(waypoints: list[Waypoint], max_v: float) -> None:
    """Computes per-waypoint target speed from curvature and friction, capped at max_v."""
    x = [p.x for p in waypoints]
    y = [p.y for p in waypoints]

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

    with np.errstate(divide='ignore'):
        radius = 1.0 / curvature

    for idx, p in enumerate(waypoints):
        v_corner = np.sqrt(9.81 * p.friction * radius[idx])
        p.v = min(float(v_corner), max_v)


def interpolate_waypoints(
    key_waypoints: list[Waypoint], default_friction: float = 0.5, max_v: float = 5.0
) -> list[Waypoint]:
    """Return a list of waypoints with cubic-spline-interpolated points between key waypoints."""
    n = len(key_waypoints)

    # Need at least 2 distinct key waypoints to fit a spline.
    if n < 2:
        return [Waypoint(x=kw.x, y=kw.y, is_key=True, friction=kw.friction) for kw in key_waypoints]

    # Build closed-loop arrays by appending the first point at the end.
    xs = np.array([kw.x for kw in key_waypoints] + [key_waypoints[0].x])
    ys = np.array([kw.y for kw in key_waypoints] + [key_waypoints[0].y])

    # Parametric t = cumulative chord length so spacing reflects real distances.
    dists = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    t = np.concatenate([[0.0], np.cumsum(dists)])

    # CubicSpline requires strictly increasing t — bail out if any segment has zero length.
    if not np.all(dists > 0):
        return [Waypoint(x=kw.x, y=kw.y, is_key=True, friction=kw.friction) for kw in key_waypoints]

    cs_x = CubicSpline(t, xs)
    cs_y = CubicSpline(t, ys)

    # Per segment: at minimum 5 intermediate points, then ~1 per 0.2 m of chord.
    waypoints: list[Waypoint] = []
    for i in range(n):
        kw_curr = key_waypoints[i]
        waypoints.append(Waypoint(x=kw_curr.x, y=kw_curr.y, is_key=True, friction=kw_curr.friction))

        num_interp = max(5, int(dists[i] / 0.2))
        for ti in np.linspace(t[i], t[i + 1], num_interp + 2)[1:-1]:
            waypoints.append(Waypoint(x=float(cs_x(ti)), y=float(cs_y(ti)), is_key=False, friction=default_friction))

    optimize_path(waypoints, max_v)

    return waypoints


def create_default_waypoints() -> list[Waypoint]:
    """Creates a list of waypoints used by default."""
    return [
        Waypoint(x=0.0, y=0.0, is_key=True),
        Waypoint(x=0.0, y=1.0, is_key=True),
        Waypoint(x=1.0, y=1.0, is_key=True),
        Waypoint(x=1.0, y=0.0, is_key=True),
    ]


class ManageNode(Node):
    """Node for managing waypoints."""

    def __init__(self) -> None:
        """Initializes a new instance."""
        super().__init__("manage_node")

        #
        # Parameters
        #
        self.declare_parameter("waypoints_file_path")
        self.declare_parameter("auto_save_period", 10.0)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("interactive_markers_namespace", "manage")
        self.declare_parameter("default_friction", 0.5)
        self.declare_parameter("max_v", 5.0)

        self.waypoints_file_path: Path = Path(
            self.get_parameter_value_checked("waypoints_file_path", str)
        )
        auto_save_period: float = self.get_parameter_value_checked(
            "auto_save_period", float
        )
        self.map_frame: str = self.get_parameter_value_checked("map_frame", str)
        interactive_markers_namespace = self.get_parameter_value_checked(
            "interactive_markers_namespace", str
        )
        self.default_friction: float = self.get_parameter_value_checked(
            "default_friction", float
        )
        self.max_v: float = self.get_parameter_value_checked("max_v", float)

        #
        # Timer and publisher
        #
        if auto_save_period > 0.0:
            self.create_timer(auto_save_period, self.auto_save_callback)
        self.viz_waypoints_pub = self.create_publisher(Marker, "viz/waypoints", 10)

        #
        # Interactive marker server
        #
        self.interactive_marker_server = InteractiveMarkerServer(
            self, interactive_markers_namespace
        )
        self.int_marker_counter: int = 0

        #
        # Initialize and populate waypoints
        #

        self.key_waypoints: list[Waypoint] = []
        """List of key waypoints."""
        self.key_waypoint_int_markers: list[InteractiveMarker] = []
        """List of interactive markers corresponding to key waypoints."""
        self.waypoints: list[Waypoint] = []
        """List of all waypoints, generated by interpolating between key waypoints."""

        try:
            waypoints = load_waypoints(self.waypoints_file_path)
            if len(waypoints) < 3:
                waypoints = create_default_waypoints()
                self.get_logger().info(
                    "Read less than 3 waypoints, using default waypoints."
                )
            else:
                self.get_logger().info(
                    f"Read {len(waypoints)} waypoints from file {str(self.waypoints_file_path)!r}"
                )
        except FileNotFoundError:  # No file, start with default key
            self.get_logger().info("File does not exist, using default waypoints.")
            waypoints = create_default_waypoints()

        # Append key waypoints
        for wp in waypoints:
            if wp.is_key:
                self.append_key_waypoint(wp)

        self.get_logger().info(f"Initialized node {self.get_name()!r}")
        self.get_logger().info(f"Autosaving every {auto_save_period} seconds")

    def get_parameter_value_checked(
        self, name: str, expected_type: type, check_none: bool = True
    ) -> Any:
        """Helper method to get the value of a parameter, check that it's not
        None, then check that its type is given"""
        value = self.get_parameter(name).value
        if check_none and value is None:
            raise ValueError(f"No value given for parameter {name!r}")
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Given value for parameter {name!r} is not of type "
                f"{expected_type.__name__!r}"
            )
        return value

    def auto_save_callback(self) -> None:
        """Auto saves to the file."""
        save_waypoints(self.waypoints, self.waypoints_file_path)

        self.get_logger().info(
            f"Auto-saved {len(self.waypoints)} waypoints to "
            f"{str(self.waypoints_file_path.absolute())!r}"
        )

    def make_key_waypoint_int_marker(
        self, *, name: str, x: float, y: float
    ) -> InteractiveMarker:
        """Makes an interactive marker corresponding to a key waypoint."""
        marker = Marker(
            type=Marker.CYLINDER,
            scale=Vector3(x=0.5, y=0.5, z=0.1),
            color=ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.25),
        )
        move_control = InteractiveMarkerControl(
            always_visible=True,
            interaction_mode=InteractiveMarkerControl.MOVE_PLANE,
            independent_marker_orientation=True,
            orientation=Quaternion(w=1.0, x=0.0, y=1.0, z=0.0),
            markers=[marker],
        )
        marker = Marker(
            type=Marker.CUBE,
            scale=Vector3(x=0.8, y=0.8, z=0.05),
            color=ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5),
        )

        int_marker = InteractiveMarker(
            header=Header(frame_id=self.map_frame),
            pose=Pose(
                position=Point(x=x, y=y, z=0.0),
                orientation=Quaternion(w=0.0, x=0.0, y=0.0, z=0.0),
            ),
            name=name,
            controls=[move_control],
        )
        return int_marker

    def make_waypoints_marker(
        self,
        waypoints: list[Waypoint],
    ) -> Marker:
        """Displays the given waypoints on the given publisher.

        Args:
            pub: Publisher object
            waypoints: Matrix of waypoints
        """

        points: list[Point] = [Point(x=wp.x, y=wp.y) for wp in waypoints]
        colors: list[ColorRGBA] = [ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)] * len(
            waypoints
        )

        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.action = Marker.ADD
        marker.id = 1
        marker.type = Marker.POINTS
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.points = points
        marker.colors = colors

        return marker

    def update_waypoints(self) -> None:
        """Recalculates waypoints and publishes visualization for them."""
        self.waypoints = interpolate_waypoints(self.key_waypoints, self.default_friction, self.max_v)
        waypoints_marker = self.make_waypoints_marker(self.waypoints)
        self.viz_waypoints_pub.publish(waypoints_marker)

    def insert_key_waypoint(self, index: int, kwp: Waypoint) -> None:
        """Inserts a key waypoint at the given index."""

        # Insert and update waypoints
        self.key_waypoints.insert(index, kwp)
        self.update_waypoints()

        # Add to server
        name = f"int_marker_{self.int_marker_counter}"
        int_marker = self.make_key_waypoint_int_marker(name=name, x=kwp.x, y=kwp.y)
        self.int_marker_counter += 1
        self.key_waypoint_int_markers.insert(index, int_marker)

        self.interactive_marker_server.insert(
            int_marker, feedback_callback=self.make_feedback_callback(kwp)
        )

        menu_handler = MenuHandler()
        menu_handler.insert(
            "Insert After", callback=self.make_menu_insert_after_callback(kwp)
        )
        menu_handler.insert(
            "Insert Before", callback=self.make_menu_insert_before_callback(kwp)
        )
        menu_handler.insert("Remove", callback=self.make_menu_remove_callback(kwp))
        menu_handler.apply(self.interactive_marker_server, int_marker.name)

        self.interactive_marker_server.applyChanges()

    def append_key_waypoint(self, kwp: Waypoint) -> None:
        """Appends a key waypoint at the end of the list of key waypoints."""
        self.insert_key_waypoint(len(self.key_waypoints), kwp)

    def remove_key_waypoint(self, index: int) -> None:
        """Remove a key waypoint using the given index."""
        int_marker = self.key_waypoint_int_markers.pop(index)
        self.key_waypoints.pop(index)
        self.update_waypoints()
        self.interactive_marker_server.erase(int_marker.name)
        self.interactive_marker_server.applyChanges()

    def make_feedback_callback(
        self, kwp: Waypoint
    ) -> Callable[[InteractiveMarkerFeedback], None]:
        """Makes the callback for the feedback callback for an interactive marker."""

        def feedback_callback(feedback: InteractiveMarkerFeedback) -> None:
            if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
                kwp.x = feedback.pose.position.x
                kwp.y = feedback.pose.position.y
                self.update_waypoints()

        return feedback_callback

    def make_menu_remove_callback(
        self, kwp: Waypoint
    ) -> Callable[[InteractiveMarkerFeedback], None]:
        """Make the callback for the "Remove" menu button."""

        def menu_remove_callback(feedback: InteractiveMarkerFeedback) -> None:
            if len(self.key_waypoints) >= 4:
                index = self.key_waypoints.index(kwp)
                self.remove_key_waypoint(index)
            else:
                self.get_logger().info(
                    "Ignored 'Remove' action because there are less than 4 key waypoints."
                )

        return menu_remove_callback

    def make_menu_insert_after_callback(
        self, kwp: Waypoint
    ) -> Callable[[InteractiveMarkerFeedback], None]:
        """Make the callback for the "Insert After" menu button."""

        def menu_insert_after_callback(feedback: InteractiveMarkerFeedback) -> None:
            index = self.key_waypoints.index(kwp)
            index_next = (index + 1) % len(self.key_waypoints)
            kwp_next = self.key_waypoints[index_next]
            kwp_new = Waypoint(
                x=(kwp.x + kwp_next.x) / 2.0,
                y=(kwp.y + kwp_next.y) / 2.0,
                is_key=True,
                friction=self.default_friction,
            )
            self.insert_key_waypoint(index_next, kwp_new)

        return menu_insert_after_callback

    def make_menu_insert_before_callback(
        self, kwp: Waypoint
    ) -> Callable[[InteractiveMarkerFeedback], None]:
        """Make the callback for the "Insert Before" menu button."""

        def menu_insert_before_callback(feedback: InteractiveMarkerFeedback) -> None:
            index = self.key_waypoints.index(kwp)
            index_prev = (index - 1) % len(self.key_waypoints)
            kwp_prev = self.key_waypoints[index_prev]
            kwp_new = Waypoint(
                x=(kwp_prev.x + kwp.x) / 2.0,
                y=(kwp_prev.y + kwp.y) / 2.0,
                is_key=True,
                friction=self.default_friction,
            )
            self.insert_key_waypoint(index, kwp_new)

        return menu_insert_before_callback


def main() -> None:
    """Main entry point function."""
    rclpy.init()
    node = ManageNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
