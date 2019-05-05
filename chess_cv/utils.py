import cv2
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from collections import defaultdict

"""
There are 9 horizontal and 9 vertical lines on a chessboard.
We are solving for the inner 7 (opencv's getChessboardCorners doesn't
consider the outer corners)
"""
NUM_LINES = 7
MIN_CORNER_DIST = 8

"""
Subclass of defaultdict that returns you the key for the default_factory
if it is missing from the dict
"""
class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def imshow_lg(plt, im, im2=None, size=(24,12), **kwargs):
    """
    matplotlib's default imshow is too small. This will show a large figure
    and supports two images in the same plot.
    """
    plt.figure(figsize=size)
    if im2 is not None:
        plt.subplot(121)
        plt.imshow(im, **kwargs)
        plt.subplot(122)
        plt.imshow(im2, **kwargs)
    else:
        plt.imshow(im, **kwargs)

### IMG conversions
def bgr2gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def gray2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def gray2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def draw_points(img, points, size=25, color=(0, 0, 255)):
    for x, y in points:
        cv2.circle(img, (int(x),int(y)), size, color, -1)

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def draw_houghlines(img, lines):
    """
    Draws lines returned from opencv's Houghlines function
    """
    def calc_line(img, r, t):
        a = np.cos(t)
        b = np.sin(t)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),5)

    for rho, theta in lines[:, 0]:
        calc_line(img, rho, theta)

def filter_houghlines(houghlines, horiz_low=(85,265), horiz_high=(96,276), vert_low=(-1, 175), vert_high=(6, 186)):
    """
    Filters lines returned from opencv's Houghlines outside some theta ranges
    """
    _and = np.logical_and
    _or = np.logical_or
    angles = np.round(houghlines[:, 0, 1] * (180/np.pi))
    horiz = houghlines[np.where(
        _or(_and(angles>horiz_low[0], angles<horiz_high[0]),
            _and(angles>horiz_low[1], angles<horiz_high[1])))]
    vert = houghlines[np.where(
        _or(_and(angles>vert_low[0], angles<vert_high[0]),
            _and(angles>vert_low[1], angles<vert_high[1])))]
    return horiz, vert

# TODO: instead of merging lines in a caveman fashion, trying using a sophisticated
# clustering/unsupervised learning alg (k-means)
def merge_lines(lines, rho_accuracy):
    """
    'lines' is an array of 1x2 vectors (same as above)
    Returns an array of 1x2 vectors (same shape returned by opencv's houghLines)
    """
    pos_lines = np.absolute(lines)
    sorted_indices = pos_lines[:, 0, 0].argsort()
    sorted_lines = pos_lines[sorted_indices][:, 0]
    true_lines = lines[sorted_indices][:, 0]
    merged_lines = []
    rs, ts, c = 0, 0, 0
    lr = None
    for (rho, theta), (true_rho, _) in zip(list(sorted_lines), list(true_lines)):
        if true_rho < 0:
            theta -= np.pi
        if c == 0:
            rs, ts, lr = rho, theta, rho
            c = 1
            continue
        if abs(rho - lr) <= rho_accuracy:
            rs += rho
            ts += theta
            c += 1
        else:
            merged_lines.append(np.array([[rs / c, ts / c]])) # TODO: instead of a fair average, try to weight each line by how
                                                              # many corners lie on it
            rs, ts, c = rho, theta, 1
        lr = rho
    merged_lines.append(np.array([[rs / c, ts / c]]))
    return np.array(merged_lines)

def x_intersect(lines, y=0):
    """
    {lines} is an array of 1x2 vectors in polar form (rho, theta)
        i.e. {lines.shape == (n, 1, 2)}
    """
    return (lines[:, 0, 0] - y * np.sin(lines[:, 0, 1])) / np.cos(lines[:, 0, 1])

def y_intersect(lines, x=0):
    """
    {lines} is an array of 1x2 vectors in polar form (rho, theta)
        i.e. {lines.shape == (n, 1, 2)}
    """
    return (lines[:, 0, 0] - x * np.cos(lines[:, 0, 1])) / np.sin(lines[:, 0, 1])

def sorted_perms(set_size, sample_size):
    mem = keydefaultdict(lambda s: sorted_perms_helper(s[0], s[1]))

    def sorted_perms_helper(sample_size, start):
        if sample_size == 0:
            return [tuple()]
        if sample_size == 1:
            return [(i,) for i in range(start, set_size + 1)]
        perms = []
        for start in range(start, set_size - sample_size + 2):
            perms.extend([(start,) + p for p in mem[(sample_size - 1, start + 1)]])
        return perms
    return sorted_perms_helper(sample_size, 1)


def get_best_assignment(coords, degree=1, alg=linear_model.LinearRegression):
    """
    coords is a flat array of x or y intercepts of chessboard lines
    """
    coords_shaped = coords.reshape((coords.shape[0], 1))
    possible_indices = sorted_perms(7, coords.shape[0])
    best_score = 0
    best_assignment = None

    if degree > 1:
        poly = PolynomialFeatures(degree=degree)
        coords_shaped = poly.fit_transform(coords_shaped)

    for assignment in possible_indices:
        linreg = alg()
        indices = np.array(assignment).reshape((coords.shape[0],))
        linreg.fit(coords_shaped, indices)
        score = linreg.score(coords_shaped, indices)
        if score > best_score:
            best_score = score
            best_assignment = assignment

    return best_assignment


def get_point(i, j, h):
    """
    Converts points using a homography matrix {h}
    """
    p = np.matmul(h, np.array([i,j,1], dtype=np.float32).reshape(3,1))
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float32).reshape(1,2)

def index_points(indices1, indices2, off1=0, off2=0):
    return [np.array([i+off1, j+off2]) for i in indices1 for j in indices2]

def gen_board_indices():
    for i in range(0, 8):
        for j in range(0, 8):
            yield (i, j)

def score_bih(h, corners):
    return sum([np.linalg.norm(corners-get_point(i, j, h), axis=2).min() for i,j in gen_board_indices() if i > 0 and j > 0])

def detect_bad_lines(lines1, lines2, corners, min_distance=MIN_CORNER_DIST):
    detector = np.zeros((lines1.shape[0], lines2.shape[0]))
    for i, h in enumerate(lines1[:]):
        for j, v in enumerate(lines2[:]):
            b = np.array([v[0, 0], h[0, 0]]).reshape((2,1))
            A = np.array([np.cos(v[0,1]), np.sin(v[0,1]),
                          np.cos(h[0,1]), np.sin(h[0,1])]).reshape((2,2))
            A_inv = np.linalg.inv(A)
            intersection = np.matmul(A_inv, b).reshape((1,2))
            distances = np.linalg.norm(corners - intersection, axis=2)
            if distances.min() < min_distance:
                detector[i, j] = 1
    return detector

def get_intersections(mlh, mlv, corners, min_distance=MIN_CORNER_DIST):
    points = []
    for h in mlh[:]:
        for v in mlv[:]:
            b = np.array([v[0, 0], h[0, 0]]).reshape((2,1))
            A = np.array([np.cos(v[0,1]), np.sin(v[0,1]),
                          np.cos(h[0,1]), np.sin(h[0,1])]).reshape((2,2))
            A_inv = np.linalg.inv(A)
            intersection = np.matmul(A_inv, b).reshape((1,2))
            distances = np.linalg.norm(corners - intersection, axis=2)
            points.append(corners[np.isclose(distances, distances.min())])
    return points

def get_best_bih(horiz_lines, vert_lines, horiz_assignments, vert_assignments, corners, min_distance=MIN_CORNER_DIST, alg=cv2.RANSAC):
    best_bih = None
    best_score = np.inf
    intersections = get_intersections(horiz_lines, vert_lines, corners, min_distance)
    pts1 = np.array(intersections, dtype=np.float32).reshape(-1, 1, 2)

    for i_off in range(0, NUM_LINES - horiz_assignments[-1] + 1):
        for j_off in range(0, NUM_LINES - vert_assignments[-1] + 1):
            b_indices = index_points(horiz_assignments, vert_assignments, i_off, j_off)
            pts2 = np.array(b_indices, dtype=np.float32).reshape(-1, 1, 2)
            h, status = cv2.findHomography(pts1, pts2, alg)
            score = score_bih(np.linalg.inv(h), corners)
            if score < best_score:
                best_score = score
                best_bih = h
    return best_bih
