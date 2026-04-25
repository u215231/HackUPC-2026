#include "optimizer_algorithms_v3.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace whopt2;

struct Args {
    std::string caseDir;
    std::string outputCsv;
    OptimizerParams params;
};

void usage(const char* exe) {
    std::cerr
        << "Usage:\n"
        << "  " << exe << " <case_dir> <output.csv> "
        << "[gridStep=250] [angleStep=90] [greedyRestarts=threads*4] "
        << "[saChains=threads] [saIterations=30000] [useGapRules=1] [maxCandidates=800000]\n\n"
        << "Fast test:\n"
        << "  " << exe << " ./PublicTestCases/Case0 solution.csv 500 90 32 8 10000 1\n\n"
        << "Better quality:\n"
        << "  " << exe << " ./PublicTestCases/Case0 solution.csv 250 90 128 16 80000 1\n";
}

Args parse_args(int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
        std::exit(1);
    }
    Args a;
    a.caseDir = argv[1];
    a.outputCsv = argv[2];
    if (argc > 3) a.params.gridStep = std::stod(argv[3]);
    if (argc > 4) a.params.angleStep = std::stod(argv[4]);
    if (argc > 5) a.params.greedyRestarts = std::stoi(argv[5]);
    if (argc > 6) a.params.saChains = std::stoi(argv[6]);
    if (argc > 7) a.params.saIterations = std::stoi(argv[7]);
    if (argc > 8) a.params.useGapRules = (std::stoi(argv[8]) != 0);
    if (argc > 9) a.params.maxCandidates = static_cast<std::size_t>(std::stoull(argv[9]));

    if (a.params.greedyRestarts <= 0) a.params.greedyRestarts = std::max(8, num_threads_available() * 4);
    if (a.params.saChains <= 0) a.params.saChains = std::max(1, num_threads_available());
    return a;
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        std::uint64_t seed = static_cast<std::uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        );
        auto t0 = std::chrono::steady_clock::now();

        Instance ins = read_instance(args.caseDir);
        double ua = usable_area(ins);

        std::cerr << "Threads: " << num_threads_available() << "\n";
        std::cerr << "Warehouse vertices: " << ins.warehouse.size()
                  << ", obstacles: " << ins.obstacles.size()
                  << ", bay types: " << ins.bays.size()
                  << ", usable area: " << ua << "\n";
        std::cerr << "gridStep=" << args.params.gridStep
                  << ", angleStep=" << args.params.angleStep
                  << ", greedyRestarts=" << args.params.greedyRestarts
                  << ", saChains=" << args.params.saChains
                  << ", saIterations=" << args.params.saIterations
                  << ", gapRules=" << (args.params.useGapRules ? "ON" : "OFF") << "\n";

        std::vector<Candidate> candidates;
        Solution best = run_full_optimizer(ins, args.params, candidates, seed);

        if (!validate_solution_geometry(candidates, best, args.params.useGapRules)) {
            std::cerr << "WARNING: final solution has a geometry conflict.\n";
        }

        write_solution_csv(args.outputCsv, candidates, best, false);

        auto t1 = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(t1 - t0).count();
        std::cerr << "Wrote " << args.outputCsv << " in " << seconds << " s\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
