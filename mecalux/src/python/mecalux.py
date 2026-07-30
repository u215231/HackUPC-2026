from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
from utils import *
import time

BASE_DIR = Path(__file__).parent.parent.parent
DEG_RAD = np.pi / 180.0
RAD_DEG = 180.0 / np.pi 

class Mecalux:
    """
    Class to pares Mecalux files into Numpy matrices stored in attributes 
    as the same name of the files.

    Our goal: Place bays in a warehouse the cheapest way using the largest 
    ammount of area.

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

    def compute_price_load_ratio(self) -> float:
        bays_prices = self.types_of_bays[:, 6]
        bays_loads = self.types_of_bays[:, 5]
        bays_ids = self.output[:, 0].astype(int)
        return sum(bays_prices[bay] / bays_loads[bay] for bay in bays_ids)

    def compute_price_sum(self) -> float:
        bays_prices = self.types_of_bays[:, 6]
        bays_ids = self.output[:, 0].astype(int)
        return sum(bays_prices[bay] for bay in bays_ids)
    
    def compute_load_sum(self) -> float:
        bays_loads = self.types_of_bays[:, 5]
        bays_ids = self.output[:, 0].astype(int)
        return sum(bays_loads[bay] for bay in bays_ids)

    def compute_bays_area(self) -> float:
        bays_widths = self.types_of_bays[:, 1]
        bays_heights = self.types_of_bays[:, 2]
        bays_areas = bays_widths * bays_heights
        bays_ids = self.output[:, 0].astype(int)
        return sum(bays_areas[bay] for bay in bays_ids)
    
    def compute_warehouse_area(self) -> float:
        return calculate_poligon_area(self.warehouse[:, 0], self.warehouse[:, 1])

    def compute_quality_erroneous(self) -> float:
        """Quality of the solution: the cheapest as possible."""
        prices_loads = self.compute_price_load_ratio()
        bays_area = self.compute_bays_area()
        warehouse_area = self.compute_warehouse_area()
        return prices_loads ** (2 - bays_area / warehouse_area)
    
    def compute_quality_correct(self) -> float:
        """Quality of the solution: the cheapest as possible."""
        prices = self.compute_price_sum()
        loads = self.compute_load_sum()
        bays_area = self.compute_bays_area()
        warehouse_area = self.compute_warehouse_area()
        return (prices / loads) ** (2 - bays_area / warehouse_area)
    
    def display_warehouse_2d(self):
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
        plt.axis("equal")
        plt.title(self.data_dir.name)
        plt.xlabel("Width [x]")
        plt.ylabel("Depth [y]")   

    def display_warehouse_3d(self, height=1000):
        warehouse = self.warehouse
        if warehouse.ndim <= 1:
            return
        n, m = warehouse.shape
        assert m >= 2
        fig = plt.gcf()
        ax = fig.gca() if fig.axes else fig.add_subplot(111, projection='3d') 
        for i in range(n):
            i1, i2 = i % n, (i + 1) % n
            mesh = get_rectangle_3d_mesh(
                warehouse[i1, 0], warehouse[i2, 0],
                warehouse[i1, 1], warehouse[i2, 1], 
                0, height
            )
            mesh.display(color='k')
        ax.scatter(
            xs=self.warehouse[:, 0], 
            ys=self.warehouse[:, 1], 
            zs=0, 
            color='k', 
            marker='o')
        ax.scatter(
            xs=self.warehouse[:, 0], 
            ys=self.warehouse[:, 1], 
            zs=height, 
            color='k', 
            marker='o')
        ax.set_box_aspect((1, 1, 0.5))

    def display_obstacles_2d(self):
        obstacles = self.obstacles
        if obstacles.ndim <= 1:
            return
        n, m = obstacles.shape
        assert m >= 4
        for i in range(n):
            translation = get_translation_2d(obstacles[i, 0], obstacles[i, 1])
            mesh = get_rectangle_2d_mesh(obstacles[i, 2], obstacles[i, 3])
            mesh.transform(translation)
            mesh.plot()
            mesh.fill()
        plt.axis("equal")
        plt.title(self.data_dir.name)
        plt.xlabel("Width [x]")
        plt.ylabel("Depth [y]")   

    def display_obstacles_3d(self):
        obstacles = self.obstacles
        if obstacles.ndim <= 1:
            return
        n, m = obstacles.shape
        assert m >= 4
        for i in range(n):
            translation = np.array([[obstacles[i, 0], obstacles[i, 1], 0]]).T
            mesh = get_prism_3d_mesh(obstacles[i, 2], obstacles[i, 3], 1000)
            mesh.transform(translation)
            mesh.display()          
            # faces = get_prism_faces(vertices)
        
    def display_ceiling_2d(self):
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
    
    def display_bay_2d(self, output: np.ndarray, type_of_bay: np.ndarray):
        translation = get_translation_2d(output[1], output[2])
        rotation = get_rotation_2d(output[3] * DEG_RAD)
        bay = get_rectangle_2d(type_of_bay[1], type_of_bay[2])
        bay = transform_points(bay, translation, rotation)
        plt.plot(bay[0, 0:2], bay[1, 0:2], color='gray', linewidth=2)
        plt.plot(bay[0, 2:4], bay[1, 2:4], color='red', linewidth=3)
        plt.plot(bay[0, 1:3], bay[1, 1:3], color='gray', linewidth=2)
        plt.plot(bay[0, 3:5], bay[1, 3:5], color='gray', linewidth=2)
        center_x = np.mean(bay[0, :4])
        center_y = np.mean(bay[1, :4])
        plt.text(
            x=center_x, 
            y=center_y, 
            s=str(int(type_of_bay[0])), 
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
            marker=(4, 0, 45 + output[3]), 
            zorder=5)
        plt.axis("equal")
        
    def display_soultion_2d(self): 
        output = self.output
        types_of_bays = self.types_of_bays
        if self.output is None:
            return
        for i in range(len(output)):
            bay_id = int(output[i, 0])
            self.display_bay_2d(output[i, :], types_of_bays[bay_id, :])

    def display_bay_3d(self, output: np.ndarray, type_of_bay: np.ndarray, with_base=True):
        translation = get_translation_3d(output[1], output[2], 0)
        rotation = get_yaw_rotation_3d(output[3] * DEG_RAD)
        mesh = get_prism_3d_mesh(type_of_bay[1], type_of_bay[2], type_of_bay[3])
        mesh.transform(translation, rotation)
        if with_base:
            mesh.display_with_base(
                linewidth=1, 
                facecolor=(type_of_bay[0] / self.types_of_bays[:, 0].max(), 0.0, 0.0),
                facealpha=0.4)
        else:
            mesh.display_with_faces(
            linewidth=1, 
            facecolor=(type_of_bay[0] / self.types_of_bays[:, 0].max(), 0.0, 0.0),
            facealpha=0.4)

    def display_soultion_3d(self, with_base=True): 
        output = self.output
        types_of_bays = self.types_of_bays
        if output is None:
            return
        for i in range(len(output)):
            bay_id = int(output[i, 0])
            self.display_bay_3d(output[i, :], types_of_bays[bay_id, :], with_base)
            
def read_data(data_dir: Path | str) -> Mecalux:
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
    return Mecalux(data, data_dir)

if __name__ == "__main__":
    CASE_ID = 3
    SOLVER = "../../bin/solver.out"
    DATA_DIR = BASE_DIR / f"PublicTestCases/Case{CASE_ID}"
    #f"PrivateTestCases/PrivateCase{CASE_ID}_flame"
    OUTPUT_PATH = DATA_DIR / Mecalux.OUTPUT
    
    ABSOLUTE_MIN = False
    GREEDY_MIN = False
    DISPLAY_SOLUTION = True
    DISPLAY_QUALITY = True
    RUN_SA = False # Simulated Annealing

    if GREEDY_MIN:
        print("Running Mecalux Solver...")
        angle_step = 90
        grid_step = 2000
        sa_flag = int(RUN_SA)
        cmd = f"{SOLVER} {DATA_DIR} {OUTPUT_PATH} {sa_flag} {grid_step} {angle_step}"
        process = subprocess.run(
            cmd, 
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,           
            check=True           
        )
        for line in process.stdout:
            print(line, end="") 
            sys.stdout.flush()

    if ABSOLUTE_MIN:
        print("Running Mecalux Solver Absolute Minimizer...")
        start_time = time.perf_counter()
        N = 10
        quality = float("inf")
        for i in range(N):
            cmd = f"{SOLVER} {DATA_DIR} {OUTPUT_PATH} 0"
            process = subprocess.run(
                cmd, 
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,           
                check=True           
            )
            data = read_data(DATA_DIR)
            current_quality = data.compute_quality_correct()
            if current_quality < quality:
                quality = current_quality
                print(f"[{i + 1}] Quality score: {quality}")
            else:
                continue
        end_time = time.perf_counter()
        print(f"Total time: {end_time - start_time} [s]")
        import shutil
        shutil.copy2(
            OUTPUT_PATH,
            OUTPUT_PATH.parent / (OUTPUT_PATH.stem + "_opt" + OUTPUT_PATH.suffix))
       
    data = read_data(DATA_DIR)

    if DISPLAY_QUALITY:
        print(f"Quality score: {data.compute_quality_correct()}")

    if DISPLAY_SOLUTION:
        print("Displaying solution (press CTRL + W)...")
        plt.figure("Solution 2D")
        data.display_warehouse_2d()
        data.display_obstacles_2d()
        data.display_soultion_2d()
        # plt.show()
        plt.figure("Ceiling 2D")
        data.display_ceiling_2d()
        # plt.show()
        plt.figure("View 3D Faces")
        data.display_warehouse_3d(height=0)
        data.display_obstacles_3d()
        data.display_soultion_3d(False)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = plt.gca()
        ax.grid(False)
        fix_yaw_scrolling_3d()
        plt.figure("View 3D Bases")
        data.display_warehouse_3d(height=0)
        data.display_obstacles_3d()
        data.display_soultion_3d(True)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = plt.gca()
        ax.grid(False)
        fix_roll_scrolling_3d()
        plt.show()

        