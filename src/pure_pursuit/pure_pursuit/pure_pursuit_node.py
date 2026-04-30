#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


def load_waypoints_xy(path: Path) -> np.ndarray:
    """Load Nx2 (x, y) array from a waypoints CSV.

    Accepts either lab6/lab7 waypoints_manager format (`x,y,is_key` with header)
    or a plain numeric `x,y[,...]` CSV with no header.
    """
    with open(str(path)) as f:
        sample = f.read(1024)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        reader = csv.reader(f)
        if has_header:
            next(reader)
        rows = [(float(r[0]), float(r[1])) for r in reader if len(r) >= 2]
    if not rows:
        raise ValueError(f"No waypoints loaded from {path}")
    return np.asarray(rows, dtype=np.float64)


class PurePursuit(Node):
    """Implements Pure Pursuit on the car."""

    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.declare_parameter('waypoints_file_path', 'test.csv')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('lookahead', 2.0)
        self.declare_parameter('speed', 5.0)
        self.declare_parameter('max_steering', 0.4)
        self.declare_parameter('search_window', 100)
        self.declare_parameter('reset_dist_thresh', 5.0)

        wp_path = Path(self.get_parameter('waypoints_file_path').value)
        odom_topic = self.get_parameter('odom_topic').value
        drive_topic = self.get_parameter('drive_topic').value

        self.l = float(self.get_parameter('lookahead').value)
        self.speed = float(self.get_parameter('speed').value)
        self.max_steering = float(self.get_parameter('max_steering').value)
        self.window = int(self.get_parameter('search_window').value)
        self.reset_dist_thresh = float(self.get_parameter('reset_dist_thresh').value)

        self.current_idx = 0

        self.waypoints = load_waypoints_xy(wp_path)
        self.n = len(self.waypoints)
        self.get_logger().info(f"Loaded {self.n} waypoints from {wp_path}")

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10
        )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

    def _get_yaw(self, odom_msg: Odometry) -> float:
        q = odom_msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _full_scan(self, carX: float, carY: float) -> int:
        diffs = self.waypoints[:, :2] - np.array([carX, carY])
        dists_sq = (diffs ** 2).sum(axis=1)
        return int(np.argmin(dists_sq))

    def _windowed_scan(self, carX: float, carY: float) -> tuple[int, float]:
        best_idx = self.current_idx
        best_dist_sq = float('inf')
        for i in range(self.window):
            idx = (self.current_idx + i) % self.n
            dx = carX - self.waypoints[idx, 0]
            dy = carY - self.waypoints[idx, 1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = idx
        return best_idx, best_dist_sq

    def _find_lookahead(self, carX: float, carY: float, from_idx: int) -> int:
        l_sq = self.l ** 2
        for i in range(self.n):
            idx = (from_idx + i) % self.n
            dx = carX - self.waypoints[idx, 0]
            dy = carY - self.waypoints[idx, 1]
            if dx * dx + dy * dy >= l_sq:
                return idx
        return (from_idx + 1) % self.n

    def odom_callback(self, odom_msg: Odometry) -> None:
        carX = odom_msg.pose.pose.position.x
        carY = odom_msg.pose.pose.position.y
        yaw = self._get_yaw(odom_msg)

        nearest_idx, best_dist_sq = self._windowed_scan(carX, carY)
        if best_dist_sq > self.reset_dist_thresh ** 2:
            self.get_logger().warn("Large position jump detected, running full scan.")
            nearest_idx = self._full_scan(carX, carY)
        self.current_idx = nearest_idx

        lookahead_idx = self._find_lookahead(carX, carY, nearest_idx)
        goal = self.waypoints[lookahead_idx, :2]

        dx = goal[0] - carX
        dy = goal[1] - carY
        local_y = dx * np.sin(-yaw) + dy * np.cos(-yaw)

        steering_angle = np.arctan2(2.0 * local_y, self.l ** 2)
        steering_angle = float(np.clip(steering_angle, -self.max_steering, self.max_steering))

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.speed
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
