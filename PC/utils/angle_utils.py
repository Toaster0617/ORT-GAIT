"""
angle_utils.py
---------------
Angle normalization and visible camera selection logic.
"""

import math

def normalize_angle(a):
    while a < -math.pi:
        a += 2*math.pi
    while a > math.pi:
        a -= 2*math.pi
    return a

def determine_visible_cams(yaw, fov, cam_ranges):
    fov_r = math.radians(fov)
    vmin = (normalize_angle(yaw - fov_r/2)) % (2*math.pi)
    vmax = (normalize_angle(yaw + fov_r/2)) % (2*math.pi)

    intervals = [(vmin, vmax)] if vmin <= vmax else [(vmin, 2*math.pi), (0, vmax)]
    visible = []

    for cam, (cmin, cmax) in cam_ranges.items():
        cmin %= 2*math.pi
        cmax %= 2*math.pi
        camints = [(cmin, cmax)] if cmin <= cmax else [(cmin,2*math.pi),(0,cmax)]

        for (a,b) in intervals:
            for (c,d) in camints:
                if max(a,c) < min(b,d):
                    visible.append(cam)
                    break

    return visible
