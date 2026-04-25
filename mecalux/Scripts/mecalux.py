from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys

BASE_DIR = Path(__file__).parent.parent
DEG_RAD = np.pi / 180.0
RAD_DEG = 180.0 / np.pi 

def get_rectangle_2d(width: int, depth: int) -> np.ndarray:
    """
    Return a 2x5 array of (x, y) coords on each column representing a 
    rectangle.
    """
    return np.array([
        [0, width, width, 0, 0], 
        [0, 0, depth, depth, 0]])

def get_rectangle_3d(width: int, depth: int, height: int) -> np.ndarray:
    """
    Return a 3x5 array of (x, y) coords on each column representing a 
    rectangle.
    """
    return np.array([
        [0, width, width, 0, 0], 
        [0, 0, depth, depth, 0]])

def get_translation(tx: int, ty: int) -> np.ndarray:
    """Return a 2x1 array of a translation vector [tx, ty].T."""
    return np.array([[tx, ty]]).T

def get_rotation(radians: float) -> np.ndarray:
    """Return a 2x1 array 2x2 of a rotation matrix based on a radians angle."""
    return np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)]])

def get_scaling(sx: int, sy: int) -> np.ndarray:
    """Return a 2x2 array of a scaling matrix, i.e., a diagonal matrix of 
    entries sx and sy."""
    return np.array([
        [sx, 0],
        [0, sy]])

def transform_points(
        points: np.ndarray, 
        translation: np.ndarray = np.zeros((2, 1)), 
        rotation: np.ndarray = np.identity(2),
        scaling: np.ndarray = np.identity(2)):
    """
    Apply a scaling, rotation and translation over points organized as 2xn
    array, i.e, and array of n-columns with (x, y) coordinates.
    """
    _, m = points.shape
    points = points.astype(float)
    points = scaling @ points
    points = rotation @ points
    points += translation @ np.ones((1, m)) 
    return points

def display_rectangle(rectangle: np.ndarray):
    plt.plot(rectangle[0, :], rectangle[1, :], color='gray')
    plt.fill(rectangle[0, :], rectangle[1, :], alpha=0.2, color='blue')
    plt.axis("equal")

class Data:
    """
    Class to pares Mecalux files into Numpy matrices stored in attributes 
    as the same name of the files.

    Attributes:
        ceiling (np.array): Array of columns being [Width, Height].
        obstacles (np.array): Array of columns being [Coord X, Coord Y, Width, 
            Depth].
        types_of_bays (np.array): Array of columns being [Id, Width, Depth, 
            Height, Gap, nLoads, Price].
        warehouse (np.array): Array of columns being [Coord X, Coord Y].
        output (np.array): Array of columns being [Id, Coord X, Coord Y, 
            Rotation].
    """
    CEILING: str = "ceiling.csv"
    OBSTACLES: str = "obstacles.csv"
    TYPES_OF_BAYS: str = "types_of_bays.csv"
    WAREHOUSE: str = "warehouse.csv"
    OUTPUT: str = "solution.csv"

    ceiling: np.ndarray
    obstacles: np.ndarray
    types_of_bays: np.ndarray
    warehouse: np.ndarray
    output: np.ndarray | None = None

    def __init__(self, data: dict, data_dir: Path | str = ""):
        self.ceiling = np.array(data[self.CEILING])
        self.obstacles = np.array(data[self.OBSTACLES])
        self.types_of_bays = np.array(data[self.TYPES_OF_BAYS])
        self.warehouse = np.array(data[self.WAREHOUSE])
        if self.OUTPUT in data.keys():
            self.output = np.array(data[self.OUTPUT])
        self.data_dir = Path(data_dir)
    
    def display_warehouse(self):
        warehouse = self.warehouse
        if warehouse.ndim <= 1:
            return
        n, m = warehouse.shape
        assert m >= 2
        for i in range(n):
            i1, i2 = i % n, (i + 1) % n
            x_coords = [warehouse[i1, 0], warehouse[i2, 0]]
            y_coords = [warehouse[i1, 1], warehouse[i2, 1]]
            plt.plot(x_coords, y_coords, color='k')
        plt.scatter(warehouse[:, 0], warehouse[:, 1], color='k')
        plt.axis('equal')

    def display_warehouse_3d(self, z_level=0):
        warehouse = self.warehouse
        if warehouse.ndim <= 1:
            return
        n, m = warehouse.shape
        assert m >= 2
        fig = plt.gcf()
        if not fig.axes:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.gca()
        for i in range(n):
            i1, i2 = i % n, (i + 1) % n
            x_coords = [
                warehouse[i1, 0], 
                warehouse[i1, 0],
                warehouse[i2, 0], 
                warehouse[i2, 0], 
                warehouse[i1, 0]
            ]
            y_coords = [
                warehouse[i1, 1], 
                warehouse[i1, 1],
                warehouse[i2, 1], 
                warehouse[i2, 1], 
                warehouse[i1, 1]
            ]     
            z_coords = [0, 1000, 1000, 0, 0]
            ax.plot(x_coords, y_coords, z_coords, color='k')
        ax.scatter(
            self.warehouse[:, 0], 
            self.warehouse[:, 1], 
            0, 
            color='k', 
            marker='s')
        ax.scatter(
            self.warehouse[:, 0], 
            self.warehouse[:, 1], 
            1000, 
            color='k', 
            marker='s')
        ax.set_box_aspect((1, 1, 0.5))

    def display_obstacles(self):
        obstacles = self.obstacles
        if obstacles.ndim <= 1:
            return
        n, m = obstacles.shape
        assert m >= 4
        for i in range(n):
            translation = get_translation(obstacles[i, 0], obstacles[i, 1])
            obstacle = get_rectangle_2d(obstacles[i, 2], obstacles[i, 3])
            obstacle = transform_points(obstacle, translation)
            plt.plot(obstacle[0, :], obstacle[1, :], color='gray')
            plt.fill(obstacle[0, :], obstacle[1, :], alpha=0.2, color='blue')
            plt.axis("equal")
    
    def display_ceiling(self):
        if self.ceiling.ndim <= 1:
            return
        ceiling = np.vstack((
            self.ceiling, 
            [self.warehouse[:, 0].max(), self.ceiling[-1, 1]]))
        n, m = ceiling.shape
        assert m >= 2
        for i in range(n - 1):
            i1, i2 = i, i + 1
            x_coords = [ceiling[i1, 0], ceiling[i2, 0]]
            y_coords = [ceiling[i1, 1], ceiling[i1, 1]]
            plt.plot(x_coords, y_coords, color='k')
        for i in range(n):
            i1, i2 = i, max(i - 1, 0)
            plt.vlines(
                x=ceiling[i1, 0], 
                ymin=0, 
                ymax=max(ceiling[i1, 1], ceiling[i2, 1]), 
                color='0.7', linestyle='--')
        plt.ylim(bottom=0)
        plt.title(self.data_dir.name)
        plt.xlabel("Width [x]")
        plt.ylabel("Height [z]")
    
    def display_warehouse_obstacles(self):
        self.display_warehouse()
        self.display_obstacles()
        plt.title(self.data_dir.name)
        plt.xlabel("Width [x]")
        plt.ylabel("Depth [y]")   

    def display_bay(output, bay):
        pass

    def display_soultion(self): 
        output = self.output
        types_of_bays = self.types_of_bays
        if output is None:
            return
        for i in range(len(output)):
            bay_id = int(output[i, 0])
            translation = get_translation(output[i, 1], output[i, 2])
            rotation = get_rotation(output[i, 3] * DEG_RAD)
            bay = get_rectangle_2d(types_of_bays[bay_id, 1], types_of_bays[bay_id, 2])
            bay = transform_points(bay, translation, rotation)
            plt.plot(bay[0, 0:2], bay[1, 0:2], color='gray', linewidth=2)
            plt.plot(bay[0, 2:4], bay[1, 2:4], color='red', linewidth=3)
            plt.plot(bay[0, 1:3], bay[1, 1:3], color='gray', linewidth=2)
            plt.plot(bay[0, 3:5], bay[1, 3:5], color='gray', linewidth=2)
            center_x = np.mean(bay[0, :4])
            center_y = np.mean(bay[1, :4])
            plt.text(
                center_x, 
                center_y, 
                str(bay_id), 
                color='black', 
                fontsize=9, 
                #fontweight='bold',
                ha='center', 
                va='center', 
                zorder=10)
            plt.scatter(
                x=bay[0, :], 
                y=bay[1, :], 
                color='blue', 
                marker=(4, 0, 45 + output[i, 3]), 
                zorder=5)
            plt.axis("equal")
            
def read_data(data_dir: Path | str) -> Data:
    """
    Read the CSV files from a Mecalux data directory into a data object.
    """
    data = {}
    for path in data_dir.iterdir():
        with open(path, "r") as f:
            matrix = []
            for line in f:
                if line != "\n":                    
                    array = [float(v.rstrip("\n")) for v in line.split(",")]
                    matrix.append(array)
            data.update({path.name: matrix})
    return Data(data, data_dir)

if __name__ == "__main__":
    DATA_DIR = BASE_DIR / "PublicTestCases/Case0"
    # print("Running Mecalux Solver...")
    # cmd = f"./solver.out {DATA_DIR} {Data.OUTPUT} 500 10 32 8 20000 1"
    # process = subprocess.run(
    #     cmd, 
    #     shell=True,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.STDOUT,
    #     text=True,           
    #     check=True           
    # )
    # for line in process.stdout:
    #     print(line, end="") 
    #     sys.stdout.flush()
    print("Displaying solution (press CTRL + W)...")
    data = read_data(DATA_DIR)
    data.display_ceiling()
    plt.show()
    data.display_warehouse_obstacles()
    data.display_soultion()
    plt.show()
    