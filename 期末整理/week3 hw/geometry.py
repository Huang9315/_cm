import math

# --------------------
# Point
# --------------------
class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def translate(self, dx, dy):
        return Point(self.x + dx, self.y + dy)

    def scale(self, k, center=None):
        center = center or Point(0, 0)
        return Point(
            center.x + k * (self.x - center.x),
            center.y + k * (self.y - center.y)
        )

    def rotate(self, theta, center=None):
        center = center or Point(0, 0)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x, y = self.x - center.x, self.y - center.y
        return Point(
            center.x + x * cos_t - y * sin_t,
            center.y + x * sin_t + y * cos_t
        )

    def __repr__(self):
        return f"Point({self.x:.3f}, {self.y:.3f})"


# --------------------
# Line: ax + by + c = 0
# --------------------
class Line:
    def __init__(self, a, b, c):
        self.a, self.b, self.c = float(a), float(b), float(c)

    @classmethod
    def from_points(cls, p1, p2):
        a = p2.y - p1.y
        b = p1.x - p2.x
        c = -(a * p1.x + b * p1.y)
        return cls(a, b, c)

    def intersection(self, other):
        d = self.a * other.b - other.a * self.b
        if abs(d) < 1e-9:
            return None
        x = (self.b * other.c - other.b * self.c) / d
        y = (other.a * self.c - self.a * other.c) / d
        return Point(x, y)

    def perpendicular_through(self, p):
        return Line(self.b, -self.a, self.a * p.y - self.b * p.x)


# --------------------
# Circle
# --------------------
class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = float(radius)

    def intersect_circle(self, other):
        d = self.center.distance(other.center)
        if d > self.radius + other.radius or d < abs(self.radius - other.radius):
            return []

        a = (self.radius**2 - other.radius**2 + d**2) / (2 * d)
        h = math.sqrt(max(self.radius**2 - a**2, 0))

        x0, y0 = self.center.x, self.center.y
        x1, y1 = other.center.x, other.center.y

        xm = x0 + a * (x1 - x0) / d
        ym = y0 + a * (y1 - y0) / d

        rx = -(y1 - y0) * (h / d)
        ry = (x1 - x0) * (h / d)

        return [
            Point(xm + rx, ym + ry),
            Point(xm - rx, ym - ry)
        ]

    def intersect_line(self, line):
        a, b, c = line.a, line.b, line.c
        x0, y0 = self.center.x, self.center.y

        d = abs(a*x0 + b*y0 + c) / math.hypot(a, b)
        if d > self.radius:
            return []

        t = -(a*x0 + b*y0 + c) / (a*a + b*b)
        xh = x0 + a * t
        yh = y0 + b * t

        h = math.sqrt(max(self.radius**2 - d**2, 0))
        dx = -b / math.hypot(a, b)
        dy = a / math.hypot(a, b)

        return [
            Point(xh + dx*h, yh + dy*h),
            Point(xh - dx*h, yh - dy*h)
        ]


# --------------------
# Triangle
# --------------------
class Triangle:
    def __init__(self, p1, p2, p3):
        self.points = [p1, p2, p3]

    def translate(self, dx, dy):
        return Triangle(*[p.translate(dx, dy) for p in self.points])

    def scale(self, k, center=None):
        return Triangle(*[p.scale(k, center) for p in self.points])

    def rotate(self, theta, center=None):
        return Triangle(*[p.rotate(theta, center) for p in self.points])

    def __repr__(self):
        return f"Triangle{tuple(self.points)}"


# --------------------
# Utility Functions
# --------------------
def foot_of_perpendicular(p, line):
    perp = line.perpendicular_through(p)
    return line.intersection(perp)


def verify_pythagorean(line, p):
    foot = foot_of_perpendicular(p, line)
    base = Point(0, -line.c / line.b) if abs(line.b) > 1e-9 else Point(-line.c / line.a, 0)

    a = p.distance(foot)
    b = foot.distance(base)
    c = p.distance(base)

    return abs(a*a + b*b - c*c) < 1e-6
