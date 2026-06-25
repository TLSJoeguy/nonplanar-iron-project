import numpy as np
from numpy import ndarray as Array

class NDimObject:
    @staticmethod
    def evaluate_plane_line_intersect(line, plane) -> bool:
        n = plane.n
        D = plane.D
        d = line.d
        p = line.p

        if line.bounds is None:  # infinite line
            if np.dot(p, n) + D == 0:  # line intersects plane everywhere (technically True, but for my purposes, False)
                return False
            elif np.dot(n, d) == 0:  # line does not intersect
                return False
            else:
                return True  # line intersects at exactly one point

        else:  # line segment
            p1 = line.endpoints()[0]
            p2 = line.endpoints()[1]

            p1_check = (np.dot(p1, n) - D) / abs(np.dot(p1, n) - D)  # +-1 depending on which side of the plane it's on
            p2_check = (np.dot(p2, n) - D) / abs(np.dot(p2, n) - D)  # +-1 depending on which side of the plane it's on

            if p1_check == 0 or p2_check == 0:  # one or more points lie on the plane
                return True
            elif p1_check == p2_check:  # same sign means they're on the same side
                return False
            else:
                return True

    @staticmethod
    def get_plane_line_intersect(line, plane) -> Array | None:
        if not NDimObject.evaluate_plane_line_intersect(line, plane):
            return None
        else:
            t = (np.dot(plane.n, line.p-plane.r)) / (np.dot(-line.d, plane.n))
            point = line.p + line.d * t
        return point

    @staticmethod
    def transform(matrix: Array, object):
        assert matrix.shape == (4,4), "Error! Transformation matrix must be a 4x4 matrix!"


class Line(NDimObject):
    def __init__(self, point: Array | list, direction: Array | list, bounds=None):
        """Takes either two 1D lists or 1D arrays. Bounds are an optional tuple"""
        self.p = np.array(point)
        self.d = np.array(direction)
        self.bounds = bounds

    def __str__(self):
        if self.bounds is not None:
            return f"<Line-{self.p}+t{self.d}; {self.bounds[0]}<t<{self.bounds[1]}>"
        elif self.bounds is not None and self.bounds == (1,0):
            return f"<Line-{self.p}—{self.d}>"
        else:
            return f"<Line-{self.p}+t{self.d}>"

    def parametric(self):
        return (f"x(t)= {self.p[0]} + {self.d[0]}t\n"
                f"y(t)= {self.p[1]} + {self.d[1]}t\n"
                f"z(t)= {self.p[2]} + {self.d[2]}t\n")

    def endpoints(self) -> tuple[Array, Array] | None:
        if self.bounds is None:
            return None
        else:
            return self.p, self.p + self.d*self.bounds[1]

    @staticmethod
    def make_segment(point_1, point_2):
        point_1 = np.array(point_1)
        point_2 = np.array(point_2)
        direction = point_2 - point_1
        new_segment = Line(point_1, direction, (0, 1))
        return new_segment


class Plane(NDimObject):
    def __init__(self, point: Array | list, normal: Array | list):
        point = np.array(point)
        normal = np.array(normal)
        # normal = normal / np.linalg.norm(normal)
        self.r = point
        self.n = normal
        self.d1 = np.array([0, -normal[2], normal[1]])
        self.d2 = np.linalg.cross(np.array([0, -normal[2], normal[1]]), normal)
        self.D = (normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2])

    def __str__(self):
        return f"<Plane-n:{self.n}-r:{self.r}>"

    def equation(self):
        return f"<Plane-{self.n[0]}x + {self.n[1]}y + {self.n[2]}z = {self.D}>"