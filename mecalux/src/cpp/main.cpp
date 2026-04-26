#include "optimizer.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace whopt;

struct CliArgs {
    std::string caseDir;
    std::string outputCsv;
    OptimizerParams params;
};

void print_usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " <case_dir> <output.csv> <run_sa_bool> "
        << "[gridStep=500] [angleStep=10]\n\n"
        << "Example fast test:\n"
        << "  " << argv0 << " ./PublicTestCases/Case0 solution.csv 1 500 10\n\n"
        << "Example better quality without SA:\n"
        << "  " << argv0 << " ./PublicTestCases/Case0 solution.csv 0 250 5\n";
}

CliArgs parse_args(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        std::exit(1);
    }

    CliArgs a;
    a.caseDir = argv[1];
    a.outputCsv = argv[2];
    a.params.runSa = (std::string(argv[3]) == "1" || std::string(argv[3]) == "true" || std::string(argv[3]) == "True");

    if (argc > 4) a.params.gridStep = std::stod(argv[4]);
    if (argc > 5) a.params.angleStep = std::stod(argv[5]);

    return a;
}

int main(int argc, char** argv) {
    try {
        CliArgs args = parse_args(argc, argv);
        auto t0 = std::chrono::steady_clock::now();

        std::uint64_t seed = static_cast<std::uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        );

        Instance ins = read_instance(args.caseDir);
        double area = usable_area(ins);

        std::cerr << "Threads: " << thread_count() << "\n";
        std::cerr << "Warehouse vertices: " << ins.warehouse.size()
                  << ", obstacles: " << ins.obstacles.size()
                  << ", bay types: " << ins.bays.size()
                  << ", usable area: " << area << "\n";
        std::cerr << "Front gap rules: " << (args.params.useGapRules ? "ON" : "OFF") << "\n";

        std::vector<Candidate> candidates;
        Solution best = run_full_optimizer(ins, args.params, candidates, seed);

        bool ok = validate_solution_geometry(candidates, best, args.params.useGapRules);
        if (!ok) {
            std::cerr << "WARNING: final solution has a bay/gap geometry conflict.\n";
        }

        // Set the last argument to true if your evaluator expects a CSV header.
        write_solution_csv(args.outputCsv, candidates, best, false);
//args.caseDir + "/" 
        auto t1 = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(t1 - t0).count();
        std::cerr << "Wrote " << args.outputCsv << " in " << seconds << " s\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
