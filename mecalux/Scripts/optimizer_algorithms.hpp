#pragma once

// ================================================================
// optimizer_algorithms.hpp
//
// This file contains ONLY optimization algorithms:
//   PART 1: candidate generation
//   PART 2: greedy construction with OpenMP parallel restarts
//   PART 3: simulated annealing with OpenMP parallel chains
//
// Helper functions for geometry, gaps, score calculation, and I/O are
// located in warehouse_helpers_separated.hpp.
// ================================================================

#include "warehouse_helpers_separated.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

namespace whopt {

struct OptimizerParams {
    double gridStep = 500.0;
    double angleStep = 10.0;
    int greedyRestarts = 0;
    int saChains = 0;
    int saIterations = 25000;
    bool useGapRules = true;
    std::size_t maxCandidates = 250000;
};

// ================================================================
// PART 1: CANDIDATE GENERATION
// ================================================================

inline double candidate_quality_for_trimming(const Candidate& c) {
    // Used only when too many candidates are generated.
    // Higher is better. This does NOT decide the final answer.
    return 0.75 * (c.loads / std::max(1.0, c.price)) + 0.25 * c.bayArea / 10000000.0;
}

inline std::vector<double> build_angles(double angleStep) {
    std::vector<double> angles;
    if (angleStep <= 0.0) angleStep = 90.0;

    for (double a = 0.0; a < 360.0 - EPS; a += angleStep) add_value(angles, a);

    // Always include cardinal angles exactly, even if angleStep misses them.
    add_value(angles, 0.0);
    add_value(angles, 90.0);
    add_value(angles, 180.0);
    add_value(angles, 270.0);

    return unique_sorted(angles);
}

inline void add_anchor_coordinates_from_instance(const Instance& ins,
                                                 double gridStep,
                                                 std::vector<double>& xs,
                                                 std::vector<double>& ys) {
    auto [mn, mx] = polygon_bbox(ins.warehouse);

    double startX = std::floor(mn.x / gridStep) * gridStep;
    double endX   = std::ceil(mx.x / gridStep) * gridStep;
    double startY = std::floor(mn.y / gridStep) * gridStep;
    double endY   = std::ceil(mx.y / gridStep) * gridStep;

    // Regular grid.
    for (double x = startX; x <= endX + EPS; x += gridStep) add_value(xs, x);
    for (double y = startY; y <= endY + EPS; y += gridStep) add_value(ys, y);

    // Warehouse vertices.
    for (const auto& p : ins.warehouse) {
        add_value(xs, p.x);
        add_value(ys, p.y);
    }

    // Obstacle edges.
    for (const auto& o : ins.obstacles) {
        add_value(xs, o.x);
        add_value(xs, o.x + o.width);
        add_value(ys, o.y);
        add_value(ys, o.y + o.depth);
    }

    // Ceiling x transitions. This helps place tall bays exactly before/after a low-ceiling zone.
    for (const auto& c : ins.ceiling) add_value(xs, c.x);

    // Extra anchors that allow bay edges to align with walls/obstacles/ceiling breaks.
    std::vector<double> baseX = xs;
    std::vector<double> baseY = ys;
    for (const BayType& b : ins.bays) {
        for (double bx : baseX) {
            add_value(xs, bx - b.width);
            add_value(xs, bx + b.width);
            add_value(xs, bx - b.depth);
            add_value(xs, bx + b.depth);
            add_value(xs, bx - (b.depth + b.gap));
            add_value(xs, bx + (b.depth + b.gap));
        }
        for (double by : baseY) {
            add_value(ys, by - b.width);
            add_value(ys, by + b.width);
            add_value(ys, by - b.depth);
            add_value(ys, by + b.depth);
            add_value(ys, by - (b.depth + b.gap));
            add_value(ys, by + (b.depth + b.gap));
        }
    }
}

inline std::vector<Candidate> generate_candidates(const Instance& ins,
                                                  double gridStep,
                                                  double angleStep,
                                                  bool useGapRules,
                                                  std::size_t maxCandidates = 250000) {
    if (gridStep <= 0.0) gridStep = 500.0;

    std::vector<double> xs, ys;
    add_anchor_coordinates_from_instance(ins, gridStep, xs, ys);
    xs = unique_sorted(std::move(xs));
    ys = unique_sorted(std::move(ys));
    std::vector<double> angles = build_angles(angleStep);

    const std::int64_t B = static_cast<std::int64_t>(ins.bays.size());
    const std::int64_t A = static_cast<std::int64_t>(angles.size());
    const std::int64_t X = static_cast<std::int64_t>(xs.size());
    const std::int64_t Y = static_cast<std::int64_t>(ys.size());
    const std::int64_t total = B * A * X * Y;

    std::vector<Candidate> candidates;

#pragma omp parallel
    {
        std::vector<Candidate> local;
        local.reserve(1024);

#pragma omp for schedule(dynamic, 2048)
        for (std::int64_t flat = 0; flat < total; ++flat) {
            std::int64_t t = flat;
            int yi = static_cast<int>(t % Y); t /= Y;
            int xi = static_cast<int>(t % X); t /= X;
            int ai = static_cast<int>(t % A); t /= A;
            int bi = static_cast<int>(t);

            Candidate cand = make_candidate(ins, bi, xs[xi], ys[yi], angles[ai]);
            if (candidate_static_feasible(ins, cand, useGapRules)) {
                local.push_back(cand);
            }
        }

#pragma omp critical
        {
            candidates.insert(candidates.end(), local.begin(), local.end());
        }
    }

    if (maxCandidates > 0 && candidates.size() > maxCandidates) {
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return candidate_quality_for_trimming(a) > candidate_quality_for_trimming(b);
        });
        candidates.resize(maxCandidates);
    }

    return candidates;
}

// ================================================================
// PART 2: GREEDY ALGORITHM
// ================================================================

inline Solution greedy_run(const std::vector<Candidate>& cands,
                           double usableArea,
                           bool useGapRules,
                           std::uint64_t seed,
                           double noise = 0.20) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    const int n = static_cast<int>(cands.size());
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);

    double maxEff = 1e-9;
    double maxArea = 1e-9;
    double maxLoads = 1e-9;
    for (const Candidate& c : cands) {
        maxEff = std::max(maxEff, c.loads / std::max(1.0, c.price));
        maxArea = std::max(maxArea, c.bayArea);
        maxLoads = std::max(maxLoads, c.loads);
    }

    // Randomized greedy key.
    // We run many restarts in parallel, so this does not need to be perfect.
    std::vector<double> key(n);
    for (int i = 0; i < n; ++i) {
        const Candidate& c = cands[i];
        double eff   = (c.loads / std::max(1.0, c.price)) / maxEff;
        double area  = c.bayArea / maxArea;
        double loads = c.loads / maxLoads;
        key[i] = 0.70 * eff + 0.22 * area + 0.08 * loads + noise * U(rng);
    }

    std::sort(order.begin(), order.end(), [&](int a, int b) { return key[a] > key[b]; });

    std::vector<int> selected;
    selected.reserve(512);

    // First pass: pack according to greedy priority.
    for (int idx : order) {
        if (compatible_with_selected(cands, idx, selected, useGapRules)) {
            selected.push_back(idx);
        }
    }

    // Important for your formula:
    // More bays is not always better. Remove bad fillers.
    prune_solution(selected, cands, usableArea);

    // Second pass: only add a bay if it improves the true nonlinear objective.
    Solution cur = evaluate_solution(cands, selected, usableArea);
    std::unordered_set<int> inSolution(selected.begin(), selected.end());

    for (int idx : order) {
        if (inSolution.count(idx)) continue;
        if (!compatible_with_selected(cands, idx, selected, useGapRules)) continue;

        selected.push_back(idx);
        Solution trial = evaluate_solution(cands, selected, usableArea);
        if (trial.score + EPS < cur.score) {
            cur = trial;
            inSolution.insert(idx);
        } else {
            selected.pop_back();
        }
    }

    return evaluate_solution(cands, selected, usableArea);
}

inline Solution run_parallel_greedy(const std::vector<Candidate>& cands,
                                    double usableArea,
                                    bool useGapRules,
                                    int restarts,
                                    std::uint64_t baseSeed) {
    if (restarts <= 0) restarts = std::max(8, thread_count() * 4);
    Solution best;

#pragma omp parallel
    {
        Solution localBest;

#pragma omp for schedule(dynamic)
        for (int r = 0; r < restarts; ++r) {
            std::uint64_t seed = baseSeed + 1000003ULL * static_cast<std::uint64_t>(r + 1);
            Solution s = greedy_run(cands, usableArea, useGapRules, seed, 0.25);
            if (s.score < localBest.score) localBest = std::move(s);
        }

#pragma omp critical
        {
            if (localBest.score < best.score) best = std::move(localBest);
        }
    }

    return best;
}

// ================================================================
// PART 3: SIMULATED ANNEALING
// ================================================================

inline bool accept_sa_move(double currentScore,
                           double nextScore,
                           double temperature,
                           std::mt19937_64& rng) {
    if (nextScore + EPS < currentScore) return true;
    if (!std::isfinite(nextScore)) return false;
    if (!std::isfinite(currentScore)) return true;

    std::uniform_real_distribution<double> U(0.0, 1.0);
    double denom = std::max(1e-12, std::abs(currentScore) * temperature);
    double probability = std::exp(-(nextScore - currentScore) / denom);
    return U(rng) < probability;
}

inline Solution anneal_chain(const std::vector<Candidate>& cands,
                             double usableArea,
                             bool useGapRules,
                             Solution start,
                             int iterations,
                             std::uint64_t seed,
                             double startTemp = 0.075,
                             double endTemp = 0.001) {
    if (iterations <= 0) iterations = 10000;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::uniform_int_distribution<int> candDist(0, static_cast<int>(cands.size()) - 1);

    std::vector<int> selected = start.selected;
    Solution current = evaluate_solution(cands, selected, usableArea);
    Solution best = current;

    for (int it = 0; it < iterations; ++it) {
        double frac = static_cast<double>(it) / std::max(1, iterations - 1);
        double temperature = startTemp * std::pow(endTemp / startTemp, frac);

        std::vector<int> next = selected;
        bool generatedMove = false;
        double moveType = U(rng);

        if (moveType < 0.30 && !next.empty()) {
            // Move A: remove one bay.
            // This is necessary because some bays make the objective worse.
            std::uniform_int_distribution<int> posDist(0, static_cast<int>(next.size()) - 1);
            int pos = posDist(rng);
            next.erase(next.begin() + pos);
            generatedMove = true;
        } else if (moveType < 0.62) {
            // Move B: add one compatible bay.
            for (int tries = 0; tries < 50 && !generatedMove; ++tries) {
                int idx = candDist(rng);
                if (compatible_with_selected(cands, idx, next, useGapRules)) {
                    next.push_back(idx);
                    generatedMove = true;
                }
            }
        } else if (!next.empty()) {
            // Move C: replace one bay with another.
            std::uniform_int_distribution<int> posDist(0, static_cast<int>(next.size()) - 1);
            int pos = posDist(rng);
            next.erase(next.begin() + pos);

            for (int tries = 0; tries < 50 && !generatedMove; ++tries) {
                int idx = candDist(rng);
                if (compatible_with_selected(cands, idx, next, useGapRules)) {
                    next.push_back(idx);
                    generatedMove = true;
                }
            }

            // If replacement failed, keep it as a pure removal move.
            if (!generatedMove) generatedMove = true;
        }

        if (!generatedMove) continue;

        Solution trial = evaluate_solution(cands, next, usableArea);
        if (accept_sa_move(current.score, trial.score, temperature, rng)) {
            selected.swap(next);
            current = std::move(trial);
            if (current.score + EPS < best.score) best = current;
        }
    }

    prune_solution(best.selected, cands, usableArea);
    return evaluate_solution(cands, best.selected, usableArea);
}

inline Solution run_parallel_sa(const std::vector<Candidate>& cands,
                                double usableArea,
                                bool useGapRules,
                                const Solution& greedyBest,
                                int chains,
                                int iterations,
                                std::uint64_t baseSeed) {
    if (chains <= 0) chains = std::max(1, thread_count());
    Solution best = greedyBest;

#pragma omp parallel
    {
        Solution localBest;

#pragma omp for schedule(dynamic)
        for (int ch = 0; ch < chains; ++ch) {
            std::uint64_t seed = baseSeed + 9176ULL * static_cast<std::uint64_t>(ch + 1);

            // Start one chain from the best greedy result.
            // Start the others from different randomized greedy solutions.
            Solution start = (ch == 0)
                ? greedyBest
                : greedy_run(cands, usableArea, useGapRules, seed ^ 0xA5A5A5A5ULL, 0.35);

            Solution s = anneal_chain(cands, usableArea, useGapRules, start, iterations, seed);
            if (s.score < localBest.score) localBest = std::move(s);
        }

#pragma omp critical
        {
            if (localBest.score < best.score) best = std::move(localBest);
        }
    }

    return best;
}

inline Solution run_full_optimizer(const Instance& ins,
                                   const OptimizerParams& params,
                                   std::vector<Candidate>& outCandidates,
                                   std::uint64_t seed) {
    double area = usable_area(ins);

    std::cerr << "Generating candidates...\n";
    outCandidates = generate_candidates(
        ins,
        params.gridStep,
        params.angleStep,
        params.useGapRules,
        params.maxCandidates
    );

    if (outCandidates.empty()) {
        throw std::runtime_error("No feasible candidate placements were generated. Try smaller gridStep or disable gap rules temporarily.");
    }

    std::cerr << "Feasible candidates: " << outCandidates.size() << "\n";

    std::cerr << "Running greedy restarts...\n";
    Solution greedy = run_parallel_greedy(
        outCandidates,
        area,
        params.useGapRules,
        params.greedyRestarts,
        seed
    );
    print_solution_summary("Greedy", greedy);

    std::cerr << "Running simulated annealing chains...\n";
    Solution best = run_parallel_sa(
        outCandidates,
        area,
        params.useGapRules,
        greedy,
        params.saChains,
        params.saIterations,
        seed ^ 0x9E3779B97F4A7C15ULL
    );
    print_solution_summary("Best", best);

    return best;
}

} // namespace whopt
