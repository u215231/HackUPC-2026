from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "PublicTestCases/Case0"

class Data:
    """
    Class to pares Mecalux files into Numpy matrices stored in attributes 
    as the same name of the files.
    """
    CEILING: str = "ceiling.csv"
    OBSTACLES: str = "obstacles.csv"
    TYPES_OF_BAYS: str = "types_of_bays.csv"
    WAREHOUSE: str = "warehouse.csv"

    ceiling: np.ndarray
    obstacles: np.ndarray
    types_of_bays: np.ndarray
    warehouse: np.ndarray

    def __init__(self, data: dict, data_dir: Path | str = ""):
        self.ceiling = np.array(data[self.CEILING])
        self.obstacles = np.array(data[self.OBSTACLES])
        self.types_of_bays = np.array(data[self.TYPES_OF_BAYS])
        self.warehouse = np.array(data[self.WAREHOUSE])
        self.data_dir = Path(data_dir)
    
    def display_warehouse(self, show=True):
        if self.warehouse.ndim <= 1:
            return
        n, m = self.warehouse.shape
        assert m >= 2
        for i in range(n):
            i1, i2 = i % n, (i+1) % n
            x_coords = [self.warehouse[i1, 0], self.warehouse[i2, 0]]
            y_coords = [self.warehouse[i1, 1], self.warehouse[i2, 1]]
            plt.plot(x_coords, y_coords, color='k')
        plt.scatter(self.warehouse[:, 0], self.warehouse[:, 1], color='k')
        plt.axis('equal')
        if show:
            plt.show()
    
    def display_obstacles(self, show=True):
        if self.obstacles.ndim <= 1:
            return
        n, m = self.obstacles.shape
        assert m >= 4
        for i in range(n):
            x_coords = [
                self.obstacles[i, 0], 
                self.obstacles[i, 0] + self.obstacles[i, 2],
                self.obstacles[i, 0] + self.obstacles[i, 2],
                self.obstacles[i, 0],
                self.obstacles[i, 0]
            ]
            y_coords = [
                self.obstacles[i, 1], 
                self.obstacles[i, 1],
                self.obstacles[i, 1] + self.obstacles[i, 3],
                self.obstacles[i, 1] + self.obstacles[i, 3],
                self.obstacles[i, 1]
            ]
            plt.plot(x_coords, y_coords, color='gray')
            plt.fill(x_coords, y_coords, alpha=0.2, color='blue')
            plt.axis("equal")
        if show:
            plt.show()
    
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
        plt.show()
    
    def display_warehouse_obstacles(self):
        self.display_warehouse(show=False)
        self.display_obstacles(show=False)
        plt.title(self.data_dir.name)
        plt.xlabel("Width [x]")
        plt.ylabel("Depth [y]")
        plt.show()

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
                    array = [int(v.rstrip("\n")) for v in line.split(",")]
                    matrix.append(array)
            data.update({path.name: matrix})
    return Data(data, data_dir)

if __name__ == "__main__":
    data = read_data(DATA_DIR)
    data.display_warehouse_obstacles()
    data.display_ceiling()
    