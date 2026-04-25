#include "helpers.hpp"

#include <chrono>
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr
                << "Usage:\n"
                << "  " << argv[0] << " <case_folder> <output.csv> [options]\n\n"
                << "Options:\n"
                << "  --grid <mm>             Anchor grid step. Default: 500\n"
                << "  --angle-step <deg>      Candidate rotations. 90 is fast, 10 is stronger. Default: 90\n"
                << "  --restarts <n>          Randomized greedy restarts. Default: 64\n"
                << "  --seed <n>              Random seed. Default: 1234567\n"
                << "  --area-with-gap         Count bay depth+gap as used area instead of physical bay area\n"
                << "  --header                Write CSV header row\n"
                << "  --bad-accept <p>        Probability of accepting a non-improving bay during construction. Default: 0.015\n\n"
                << "Compile:\n"
                << "  g++ -O3 -std=c++17 -fopenmp main.cpp -o solver\n\n"
                << "Example:\n"
                << "  ./solver PublicTestCases/Case0 output_case0.csv --grid 250 --angle-step 10 --restarts 128\n";
            return 1;
        }

        const std::string caseDir = argv[1];
        const std::string outputPath = argv[2];
        wh::Options opt = wh::parseOptions(argc, argv, 3);

#ifdef _OPENMP
        std::cerr << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
        std::cerr << "OpenMP is not enabled. Compile with -fopenmp for parallel speedup.\n";
#endif

        auto t0 = std::chrono::steady_clock::now();
        wh::Instance instance = wh::readInstance(caseDir);

        std::cerr << "Warehouse area: " << instance.warehouseArea << "\n";
        std::cerr << "Obstacle area:  " << instance.obstacleArea << "\n";
        std::cerr << "Usable area:    " << instance.usableArea << "\n";
        std::cerr << "Bay types:      " << instance.bays.size() << "\n";
        std::cerr << "Grid step:      " << opt.gridStep << "\n";
        std::cerr << "Angle step:     " << opt.angleStep << "\n";
        std::cerr << "Restarts:       " << opt.restarts << "\n";

        std::vector<wh::Candidate> candidates;
        wh::Solution best = wh::solveInstance(instance, opt, candidates);

        auto t1 = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(t1 - t0).count();

        std::cerr << "Candidates:     " << candidates.size() << "\n";
        wh::printSolutionSummary(instance, best, candidates);
        std::cerr << "Runtime:        " << seconds << " s\n";

        wh::writeSolutionCsv(outputPath, best, candidates, opt.writeHeader);
        std::cerr << "Wrote:          " << outputPath << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
