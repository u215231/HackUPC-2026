import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

def get_translation_2d(tx: int, ty: int) -> np.ndarray:
    """Return a 2x1 array of a translation vector [tx, ty].T."""
    return np.array([[tx, ty]]).T

def get_rotation_2d(radians: float) -> np.ndarray:
    """Return a 2x1 array 2x2 of a rotation matrix based on a radians angle."""
    return np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)]])

def get_scaling_2d(sx: int, sy: int) -> np.ndarray:
    """Return a 2x2 array of a scaling matrix, i.e., a diagonal matrix of 
    entries sx and sy."""
    return np.array([
        [sx, 0],
        [0, sy]])

def get_translation_3d(tx: int, ty: int, tz: int) -> np.ndarray:
    return np.array([[tx, ty, tz]]).T

def get_yaw_rotation_3d(radians: float) -> np.ndarray:
    """Return a 2x1 array 2x2 of a rotation matrix based on a radians angle."""
    return np.array([
        [np.cos(radians), -np.sin(radians), 0],
        [np.sin(radians), np.cos(radians),  0],
        [0,               0,                1]])

def transform_points(
        points: np.ndarray, 
        translation: np.ndarray = None, 
        rotation: np.ndarray = None,
        scaling: np.ndarray = None):
    """
    Apply a scaling, rotation and translation over points organized as 2xn
    array, i.e, and array of n-columns with (x, y) coordinates.
    """
    n, m = points.shape
    translation = np.zeros((n, 1)) if translation is None else translation
    rotation =  np.identity(n) if rotation is None else rotation
    scaling = np.identity(n) if scaling is None else scaling
    points = points.astype(float)
    points = scaling @ points
    points = rotation @ points
    points = points + translation @ np.ones((1, m)) 
    return points

def display_rectangle_2d(rectangle: np.ndarray):
    plt.plot(rectangle[0, :], rectangle[1, :], color='gray')
    plt.fill(rectangle[0, :], rectangle[1, :], alpha=0.2, color='blue')
    plt.axis("equal")

def display_prism_3d(
        vertices: np.ndarray, 
        edges: list[tuple], 
        color='blue', 
        linewidth=2):
    plt.subplot(projection='3d')
    for u, v in edges:
        x = [vertices[0, u], vertices[0, v]]
        y = [vertices[1, u], vertices[1, v]]
        z = [vertices[2, u], vertices[2, v]]
        plt.plot(x, y, z, color=color, linewidth=linewidth)

def fix_yaw_scrolling_3d():
    fig = plt.gcf()
    ax = plt.gca()
    fixed_elevation = 30 
    ax.view_init(elev=fixed_elevation)
    def on_move(event):
        if event.button == 1:
            ax.view_init(elev=fixed_elevation, azim=ax.azim)
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect('motion_notify_event', on_move)

def fix_pitch_scrolling_3d():
    fig = plt.gcf()
    ax = plt.gca()
    fixed_azim = 180 
    fixed_roll = 0
    def on_move(event):
        if event.button == 1:
            ax.view_init(elev=ax.elev, azim=fixed_azim, roll=fixed_roll)
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect('motion_notify_event', on_move)

def fix_roll_scrolling_3d():
    fig = plt.gcf()
    ax = plt.gca()
    fixed_azim = -90 
    fixed_roll = 0
    def on_move(event):
        if event.button == 1:
            ax.view_init(elev=ax.elev, azim=fixed_azim, roll=fixed_roll)
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect('motion_notify_event', on_move)

class Mesh(ABC):
    vertices: np.ndarray
    edges: list[tuple]

    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges

    def transform(       
            self, 
            translation: np.ndarray = None, 
            rotation: np.ndarray = None,
            scaling: np.ndarray = None):
        self.vertices = transform_points(
            self.vertices,
            translation,
            rotation,
            scaling)

class Mesh3D(Mesh):
    def __init__(self, vertices, edges):
        super().__init__(vertices, edges)

    def display(self, color='blue', linewidth=2):
        plt.subplot(projection='3d')
        for u, v in self.edges:
            x = [self.vertices[0, u], self.vertices[0, v]]
            y = [self.vertices[1, u], self.vertices[1, v]]
            z = [self.vertices[2, u], self.vertices[2, v]]
            plt.plot(x, y, z, color=color, linewidth=linewidth)

class Mesh2D(Mesh):
    def __init__(self, vertices, edges):
        super().__init__(vertices, edges)

    def plot(self, color='gray'):
        for u, v in self.edges:
            x = [self.vertices[0, u], self.vertices[0, v]]
            y = [self.vertices[1, u], self.vertices[1, v]]
            plt.plot(x, y, color=color)
    
    def fill(self, alpha=0.2, color='blue'):
        plt.fill(self.vertices[0, :], self.vertices[1, :], alpha=alpha, color=color)

def get_rectangle_2d(width: int, depth: int) -> np.ndarray:
    """
    Return a 2x5 array of (x, y) coords on each column representing a 
    rectangle.
    """
    return np.array([
        [0, width, width, 0, 0], 
        [0, 0, depth, depth, 0]])

def get_rectangle_2d_mesh(width: int, depth: int) -> Mesh2D:
    vertices = np.array([
        [0, width, width, 0], 
        [0, 0, depth, depth]])
    edges = [(0,1), (1,2), (2,3), (3,0)]
    return Mesh2D(vertices, edges)

def get_rectangle_3d_mesh(
        x1: int, x2: int, 
        y1: int, y2: int, 
        z1: int, z2: int) -> Mesh3D:
    vertices = np.array([
        [x1, x1, x2, x2], 
        [y1, y1, y2, y2],
        [z1, z2, z2, z1]])
    edges = [(0,1), (1,2), (2,3), (3,0)]
    return Mesh3D(vertices, edges)

def get_prism_3d_mesh(width: int, depth: int, height: int) -> Mesh3D:
    vertices = np.array([
        [0, width, width, 0,     0,      width,  width,  0     ], 
        [0, 0,     depth, depth, 0,      0,      depth,  depth ],
        [0, 0,     0,     0,     height, height, height, height]])
    edges = [
        (0,1), (1,2), (2,3), (3,0), 
        (4,5), (5,6), (6,7), (7,4), 
        (0,4), (1,5), (2,6), (3,7)]
    return Mesh3D(vertices, edges)

# def get_prism_faces(vertices: np.ndarray):
#     v = vertices.T
#     return [
#         [v[0], v[1], v[2], v[3]], 
#         [v[4], v[5], v[6], v[7]], 
#         [v[0], v[1], v[5], v[4]],
#         [v[1], v[2], v[6], v[5]],
#         [v[2], v[3], v[7], v[6]],
#         [v[3], v[0], v[4], v[7]] 
#     ]

def calculate_poligon_area(x_coords, y_coords):
    """
    Calculate the area of an irregular polygon from his vertexs using 
    Gauss (Shoelace) formula
    """
    x_y_next = np.dot(x_coords, np.roll(y_coords, -1))
    y_x_next = np.dot(y_coords, np.roll(x_coords, -1))
    area = 0.5 * np.abs(x_y_next - y_x_next)
    return area