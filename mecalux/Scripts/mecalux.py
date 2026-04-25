from __future__ import annotations

from pathlib import Path
import argparse
import os
import shutil
import subprocess
import tempfile
from typing import Iterable

import numpy as np

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "PublicTestCases/Case0"


class Data:
    """
    Class to parse Mecalux files into Numpy matrices stored in attributes
    with the same names as the files.
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
        self.ceiling = np.array(data.get(self.CEILING, []), dtype=int)
        self.obstacles = np.array(data.get(self.OBSTACLES, []), dtype=int)
        self.types_of_bays = np.array(data.get(self.TYPES_OF_BAYS, []), dtype=int)
        self.warehouse = np.array(data.get(self.WAREHOUSE, []), dtype=int)
        self.data_dir = Path(data_dir)

    def display_warehouse(self, show=True):
        import matplotlib.pyplot as plt
        if self.warehouse.ndim <= 1:
            return
        n, m = self.warehouse.shape
        assert m >= 2
        for i in range(n):
            i1, i2 = i % n, (i + 1) % n
            x_coords = [self.warehouse[i1, 0], self.warehouse[i2, 0]]
            y_coords = [self.warehouse[i1, 1], self.warehouse[i2, 1]]
            plt.plot(x_coords, y_coords, color="k")
        plt.scatter(self.warehouse[:, 0], self.warehouse[:, 1], color="k")
        plt.axis("equal")
        if show:
            plt.show()

    def display_obstacles(self, show=True):
        import matplotlib.pyplot as plt
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
                self.obstacles[i, 0],
            ]
            y_coords = [
                self.obstacles[i, 1],
                self.obstacles[i, 1],
                self.obstacles[i, 1] + self.obstacles[i, 3],
                self.obstacles[i, 1] + self.obstacles[i, 3],
                self.obstacles[i, 1],
            ]
            plt.plot(x_coords, y_coords, color="gray")
            plt.fill(x_coords, y_coords, alpha=0.2, color="blue")
            plt.axis("equal")
        if show:
            plt.show()

    def display_ceiling(self):
        import matplotlib.pyplot as plt
        if self.ceiling.ndim <= 1:
            return
        ceiling = np.vstack((self.ceiling, [self.warehouse[:, 0].max(), self.ceiling[-1, 1]]))
        n, m = ceiling.shape
        assert m >= 2
        for i in range(n - 1):
            i1, i2 = i, i + 1
            x_coords = [ceiling[i1, 0], ceiling[i2, 0]]
            y_coords = [ceiling[i1, 1], ceiling[i1, 1]]
            plt.plot(x_coords, y_coords, color="k")
        for i in range(n):
            i1, i2 = i, max(i - 1, 0)
            plt.vlines(
                x=ceiling[i1, 0],
                ymin=0,
                ymax=max(ceiling[i1, 1], ceiling[i2, 1]),
                color="0.7",
                linestyle="--",
            )
        plt.ylim(bottom=0)
        plt.title(self.data_dir.name)
        plt.xlabel("Width [x]")
        plt.ylabel("Height [z]")
        plt.show()

    def display_warehouse_obstacles(self):
        import matplotlib.pyplot as plt
        self.display_warehouse(show=False)
        self.display_obstacles(show=False)
        plt.title(self.data_dir.name)
        plt.xlabel("Width [x]")
        plt.ylabel("Depth [y]")
        plt.show()


def read_data(data_dir: Path | str) -> Data:
    """Read the CSV files from a Mecalux data directory into a Data object."""
    data_dir = Path(data_dir)
    data = {}
    for filename in (Data.CEILING, Data.OBSTACLES, Data.TYPES_OF_BAYS, Data.WAREHOUSE):
        path = data_dir / filename
        matrix = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        matrix.append([int(v.strip()) for v in line.split(",") if v.strip()])
        data[filename] = matrix
    return Data(data, data_dir)


def _write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    """Write one of the input matrices in the plain integer CSV format used by the C++ solver."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(matrix)
    if arr.size == 0:
        path.write_text("", encoding="utf-8")
        return
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    with open(path, "w", encoding="utf-8") as f:
        for row in arr:
            f.write(", ".join(str(int(v)) for v in row) + "\n")


def write_data_case_dir(data: Data, case_dir: Path | str) -> Path:
    """
    Materialize a Data object as the four CSV files consumed by the C++ OpenMP solver.

    This lets the Python hackathon entry point own the input, while the fast C++ code
    performs candidate generation and optimization.
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_matrix_csv(case_dir / Data.CEILING, data.ceiling)
    _write_matrix_csv(case_dir / Data.OBSTACLES, data.obstacles)
    _write_matrix_csv(case_dir / Data.TYPES_OF_BAYS, data.types_of_bays)
    _write_matrix_csv(case_dir / Data.WAREHOUSE, data.warehouse)
    return case_dir


def _newer_than(path: Path, sources: Iterable[Path]) -> bool:
    if not path.exists():
        return False
    target_time = path.stat().st_mtime
    return all(target_time >= src.stat().st_mtime for src in sources if src.exists())


def build_cpp_solver(source_dir: Path | str | None = None, force: bool = False) -> Path:
    """
    Build the C++ OpenMP solver and return the executable path.

    Expected files next to this Python script:
      - main.cpp
      - helpers.hpp
    """
    source_dir = Path(source_dir) if source_dir is not None else Path(__file__).resolve().parent
    main_cpp = source_dir / "main.cpp"
    helpers_hpp = source_dir / "helpers.hpp"
    if not main_cpp.exists() or not helpers_hpp.exists():
        raise FileNotFoundError(
            f"Could not find main.cpp and helpers.hpp in {source_dir}. Keep the three files together."
        )

    build_dir = Path(tempfile.gettempdir()) / "mecalux_openmp_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / ("mecalux_solver.exe" if os.name == "nt" else "mecalux_solver")

    if not force and _newer_than(exe, [main_cpp, helpers_hpp]):
        return exe

    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("No C++ compiler found. Install g++ or c++ to build the OpenMP solver.")

    cmd = [compiler, "-O3", "-std=c++17", "-fopenmp", str(main_cpp), "-o", str(exe)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # If OpenMP is unavailable, still build a single-threaded executable rather than failing completely.
    if proc.returncode != 0:
        fallback_cmd = [compiler, "-O3", "-std=c++17", str(main_cpp), "-o", str(exe)]
        fallback = subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if fallback.returncode != 0:
            raise RuntimeError(
                "Failed to compile C++ solver.\n"
                f"OpenMP command: {' '.join(cmd)}\n"
                f"OpenMP stderr:\n{proc.stderr}\n"
                f"Fallback command: {' '.join(fallback_cmd)}\n"
                f"Fallback stderr:\n{fallback.stderr}"
            )
    return exe


def read_solution_csv(path: Path | str) -> np.ndarray:
    """Read solver output as an N x 4 integer array: bay_id, x, y, rotation."""
    rows = []
    path = Path(path)
    if not path.exists():
        return np.empty((0, 4), dtype=int)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("id"):
                continue
            rows.append([int(round(float(v.strip()))) for v in line.split(",") if v.strip()])
    if not rows:
        return np.empty((0, 4), dtype=int)
    return np.array(rows, dtype=int)


def solve_data(
    data: Data,
    output_path: Path | str = "output.csv",
    *,
    grid: int = 500,
    angle_step: int = 90,
    restarts: int = 64,
    seed: int = 1234567,
    area_with_gap: bool = False,
    threads: int | None = None,
    force_rebuild: bool = False,
) -> np.ndarray:
    """
    Run the OpenMP C++ heuristic on a Data object and return the output matrix.

    Parameters tune speed vs. quality:
      - grid: smaller is stronger but slower, e.g. 500 fast, 250 stronger.
      - angle_step: 90 uses orthogonal bays; 10 enables angled placements.
      - restarts: more randomized greedy restarts usually improves the score.
    """
    output_path = Path(output_path)
    solver = build_cpp_solver(force=force_rebuild)

    with tempfile.TemporaryDirectory(prefix="mecalux_case_") as tmp:
        case_dir = write_data_case_dir(data, Path(tmp) / "case")
        cmd = [
            str(solver),
            str(case_dir),
            str(output_path),
            "--grid",
            str(grid),
            "--angle-step",
            str(angle_step),
            "--restarts",
            str(restarts),
            "--seed",
            str(seed),
        ]
        if area_with_gap:
            cmd.append("--area-with-gap")

        env = os.environ.copy()
        if threads is not None:
            env["OMP_NUM_THREADS"] = str(threads)

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                "C++ solver failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

    return read_solution_csv(output_path)


def solve_case(
    data_dir: Path | str,
    output_path: Path | str = "output.csv",
    **kwargs,
) -> np.ndarray:
    """Convenience wrapper: read a case folder, run the solver, and return the output matrix."""
    return solve_data(read_data(data_dir), output_path, **kwargs)


# Use this function if the hackathon runner imports this script and passes a Data object.
def get_output(data: Data, output_path: Path | str = "output.csv") -> np.ndarray:
    return solve_data(data, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mecalux bay placement solver with C++ OpenMP backend")
    parser.add_argument("data_dir", nargs="?", default=str(DATA_DIR), help="Case folder with the four CSV files")
    parser.add_argument("output", nargs="?", default="output.csv", help="Output CSV path")
    parser.add_argument("--grid", type=int, default=500)
    parser.add_argument("--angle-step", type=int, default=90)
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234567)
    parser.add_argument("--area-with-gap", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--display", action="store_true", help="Display input warehouse and ceiling before solving")
    args = parser.parse_args()

    data = read_data(args.data_dir)
    if args.display:
        data.display_warehouse_obstacles()
        data.display_ceiling()

    result = solve_data(
        data,
        args.output,
        grid=args.grid,
        angle_step=args.angle_step,
        restarts=args.restarts,
        seed=args.seed,
        area_with_gap=args.area_with_gap,
        threads=args.threads,
        force_rebuild=args.force_rebuild,
    )
    print(f"Wrote {args.output} with {len(result)} bays")


if __name__ == "__main__":
    main()
