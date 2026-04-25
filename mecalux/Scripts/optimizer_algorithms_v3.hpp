#pragma once

// ================================================================
// optimizer_algorithms_v3.hpp
//
// This file contains ONLY optimization algorithms.
// The helpers are in warehouse_helpers_v2.hpp.
//
// PART 1: candidate generation
// PART 2: greedy algorithm with OpenMP parallel restarts
// PART 3: simulated annealing with OpenMP parallel chains
//
// Main correction versus the previous version:
//   The greedy algorithm builds dense layouts first and only then prunes
//   bays that hurt the real score. The previous version was too sparse
//   because it rejected/removed too early according to the nonlinear score.
// ================================================================

#include "warehouse_helpers_v2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

namespace whopt2 {

struct OptimizerParams {
    double gridStep = 250.0;
    double angleStep = 90.0;
    int greedyRestarts = 0;
    int saChains = 0;
    int saIterations = 30000;
    bool useGapRules = true;
    std::size_t maxCandidates = 800000;
};

inline void push_value(std::vector<double>& v, double x) {
    if (std::isfinite(x)) v.push_back(x);
}

inline void unique_sort_in_place(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    std::vector<double> out;
    out.reserve(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (out.empty() || std::fabs(v[i] - out.back()) > 1e-6) out.push_back(v[i]);
    }
    v.swap(out);
}

// ================================================================
// PART 1: CANDIDATE GENERATION
// ================================================================

inline void build_angles(double angleStep, std::vector<double>& angles) {
    angles.clear();
    if (angleStep <= 0.0) angleStep = 90.0;
    for (double a = 0.0; a < 360.0 - EPS; a += angleStep) push_value(angles, a);
    push_value(angles, 0.0);
    push_value(angles, 90.0);
    push_value(angles, 180.0);
    push_value(angles, 270.0);
    unique_sort_in_place(angles);
}

inline void build_anchor_coordinates(const Instance& ins, double gridStep,
                                     std::vector<double>& xs, std::vector<double>& ys) {
    xs.clear();
    ys.clear();
    std::pair<Point, Point> bb = polygon_bbox(ins.warehouse);
    Point mn = bb.first;
    Point mx = bb.second;

    double step = std::max(50.0, gridStep);
    double startX = std::floor(mn.x / step) * step;
    double endX = std::ceil(mx.x / step) * step;
    double startY = std::floor(mn.y / step) * step;
    double endY = std::ceil(mx.y / step) * step;

    for (double x = startX; x <= endX + EPS; x += step) push_value(xs, x);
    for (double y = startY; y <= endY + EPS; y += step) push_value(ys, y);

    for (std::size_t i = 0; i < ins.warehouse.size(); ++i) {
        push_value(xs, ins.warehouse[i].x);
        push_value(ys, ins.warehouse[i].y);
    }

    for (std::size_t i = 0; i < ins.obstacles.size(); ++i) {
        const Obstacle& o = ins.obstacles[i];
        push_value(xs, o.x);
        push_value(xs, o.x + o.width);
        push_value(ys, o.y);
        push_value(ys, o.y + o.depth);
    }

    for (std::size_t i = 0; i < ins.ceiling.size(); ++i) {
        push_value(xs, ins.ceiling[i].x);
    }

    // Offsets let bay edges align with walls/obstacles/ceiling transitions.
    std::vector<double> baseX = xs;
    std::vector<double> baseY = ys;
    for (std::size_t bi = 0; bi < ins.bays.size(); ++bi) {
        const BayType& b = ins.bays[bi];
        double vals[6] = {b.width, b.depth, b.depth + b.gap, b.width + b.gap,
                          0.5 * b.width, 0.5 * b.depth};
        for (std::size_t i = 0; i < baseX.size(); ++i) {
            for (int k = 0; k < 6; ++k) {
                push_value(xs, baseX[i] - vals[k]);
                push_value(xs, baseX[i] + vals[k]);
            }
        }
        for (std::size_t i = 0; i < baseY.size(); ++i) {
            for (int k = 0; k < 6; ++k) {
                push_value(ys, baseY[i] - vals[k]);
                push_value(ys, baseY[i] + vals[k]);
            }
        }
    }

    unique_sort_in_place(xs);
    unique_sort_in_place(ys);
}

inline double candidate_trim_priority(const Candidate& c, const Instance& ins) {
    const BayType& b = ins.bays[c.bayIndex];
    double efficiency = b.loads / std::max(1.0, b.price);
    double areaTerm = (b.width * b.depth) / 1000000.0;
    double spatial = std::fmod(std::fabs(c.x * 0.00137 + c.y * 0.00211 + c.rotation * 0.017), 1.0);
    return 1000.0 * efficiency + 0.01 * areaTerm + 0.0001 * spatial;
}

inline void generate_candidates(const Instance& ins,
                                double gridStep,
                                double angleStep,
                                bool useGapRules,
                                std::size_t maxCandidates,
                                std::vector<Candidate>& candidates) {
    candidates.clear();

    std::vector<double> xs, ys, angles;
    build_anchor_coordinates(ins, gridStep, xs, ys);
    build_angles(angleStep, angles);

    const long long B = static_cast<long long>(ins.bays.size());
    const long long A = static_cast<long long>(angles.size());
    const long long X = static_cast<long long>(xs.size());
    const long long Y = static_cast<long long>(ys.size());
    const long long total = B * A * X * Y;

#pragma omp parallel
    {
        std::vector<Candidate> local;
        local.reserve(2048);

#pragma omp for schedule(dynamic, 4096)
        for (long long flat = 0; flat < total; ++flat) {
            long long t = flat;
            int yi = static_cast<int>(t % Y); t /= Y;
            int xi = static_cast<int>(t % X); t /= X;
            int ai = static_cast<int>(t % A); t /= A;
            int bi = static_cast<int>(t);

            Candidate c = make_candidate(ins, bi, xs[xi], ys[yi], angles[ai]);
            if (candidate_static_feasible(ins, c, useGapRules)) local.push_back(c);
        }

#pragma omp critical
        {
            candidates.insert(candidates.end(), local.begin(), local.end());
        }
    }

    if (maxCandidates > 0 && candidates.size() > maxCandidates) {
        std::sort(candidates.begin(), candidates.end(), [&](const Candidate& a, const Candidate& b) {
            return candidate_trim_priority(a, ins) > candidate_trim_priority(b, ins);
        });
        candidates.resize(maxCandidates);
    }
}

// ================================================================
// PART 2: GREEDY ALGORITHM
// ================================================================

inline double norm01(double v, double lo, double hi) {
    if (hi - lo < EPS) return 0.0;
    return (v - lo) / (hi - lo);
}

inline void make_greedy_order(const std::vector<Candidate>& cands,
                              int mode,
                              std::uint64_t seed,
                              std::vector<int>& order) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    int n = static_cast<int>(cands.size());
    order.resize(n);
    std::iota(order.begin(), order.end(), 0);

    double maxEff = 1e-9, maxArea = 1e-9, maxLoads = 1e-9;
    double minX = n ? cands[0].footprint.minX : 0.0;
    double maxX = minX;
    double minY = n ? cands[0].footprint.minY : 0.0;
    double maxY = minY;

    for (int i = 0; i < n; ++i) {
        const Candidate& c = cands[i];
        maxEff = std::max(maxEff, c.loads / std::max(1.0, c.price));
        maxArea = std::max(maxArea, c.bayArea);
        maxLoads = std::max(maxLoads, c.loads);
        minX = std::min(minX, c.footprint.minX); maxX = std::max(maxX, c.footprint.maxX);
        minY = std::min(minY, c.footprint.minY); maxY = std::max(maxY, c.footprint.maxY);
    }

    std::vector<double> key(n, 0.0);
    for (int i = 0; i < n; ++i) {
        const Candidate& c = cands[i];
        double eff = (c.loads / std::max(1.0, c.price)) / maxEff;
        double area = c.bayArea / maxArea;
        double loads = c.loads / maxLoads;
        double nx = norm01(c.footprint.minX, minX, maxX);
        double ny = norm01(c.footprint.minY, minY, maxY);
        double noise = 0.05 * U(rng);
        int m = mode % 8;
        if (m == 0) key[i] = 4.0 * eff + 1.0 * area - 0.05 * ny - 0.02 * nx + noise;
        else if (m == 1) key[i] = 2.0 * eff + 3.0 * area - 0.04 * ny + noise;
        else if (m == 2) key[i] = 3.0 * eff + 1.0 * loads - 0.10 * nx + noise;
        else if (m == 3) key[i] = 3.0 * eff + 1.0 * loads + 0.10 * nx + noise;
        else if (m == 4) key[i] = 2.0 * eff + 2.0 * area - 0.10 * ny + noise;
        else if (m == 5) key[i] = 2.0 * eff + 2.0 * area + 0.10 * ny + noise;
        else if (m == 6) key[i] = 1.0 * eff + 4.0 * area + noise;
        else key[i] = 3.0 * eff + 1.0 * area + 0.5 * U(rng);
    }

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (std::fabs(key[a] - key[b]) > 1e-12) return key[a] > key[b];
        return a < b;
    });
}

inline void add_all_compatible_in_order(const std::vector<Candidate>& cands,
                                        std::vector<int>& selected,
                                        const std::vector<int>& order,
                                        bool useGapRules) {
    std::unordered_set<int> in(selected.begin(), selected.end());
    for (std::size_t k = 0; k < order.size(); ++k) {
        int idx = order[k];
        if (in.find(idx) != in.end()) continue;
        if (compatible_with_selected(cands, idx, selected, useGapRules)) {
            selected.push_back(idx);
            in.insert(idx);
        }
    }
}

inline void add_objective_improving_candidates(const std::vector<Candidate>& cands,
                                               std::vector<int>& selected,
                                               const std::vector<int>& order,
                                               bool useGapRules,
                                               double usableArea) {
    std::unordered_set<int> in(selected.begin(), selected.end());
    Solution current = evaluate_solution(cands, selected, usableArea);
    for (std::size_t k = 0; k < order.size(); ++k) {
        int idx = order[k];
        if (in.find(idx) != in.end()) continue;
        if (!compatible_with_selected(cands, idx, selected, useGapRules)) continue;
        selected.push_back(idx);
        Solution trial = evaluate_solution(cands, selected, usableArea);
        if (trial.score + EPS < current.score) {
            current = trial;
            in.insert(idx);
        } else {
            selected.pop_back();
        }
    }
}

inline Solution greedy_run(const std::vector<Candidate>& cands,
                           double usableArea,
                           bool useGapRules,
                           std::uint64_t seed,
                           int mode) {
    std::vector<int> order;
    make_greedy_order(cands, mode, seed, order);

    std::vector<int> selected;
    selected.reserve(1024);

    // Dense construction first.
    add_all_compatible_in_order(cands, selected, order, useGapRules);

    // Then prune with the true objective.
    prune_solution_by_true_objective(selected, cands, usableArea);

    // Finally, add individual score-improving bays.
    add_objective_improving_candidates(cands, selected, order, useGapRules, usableArea);

    return evaluate_solution(cands, selected, usableArea);
}

inline Solution run_parallel_greedy(const std::vector<Candidate>& cands,
                                    double usableArea,
                                    bool useGapRules,
                                    int restarts,
                                    std::uint64_t seed) {
    if (restarts <= 0) restarts = std::max(8, num_threads_available() * 4);
    Solution globalBest;

#pragma omp parallel
    {
        Solution localBest;
#pragma omp for schedule(dynamic)
        for (int r = 0; r < restarts; ++r) {
            std::uint64_t s = seed + 1000003ULL * static_cast<std::uint64_t>(r + 1);
            Solution sol = greedy_run(cands, usableArea, useGapRules, s, r % 8);
            if (sol.score < localBest.score) localBest = sol;
        }
#pragma omp critical
        {
            if (localBest.score < globalBest.score) globalBest = localBest;
        }
    }
    return globalBest;
}

// ================================================================
// PART 3: SIMULATED ANNEALING
// ================================================================

inline bool accept_sa(double curScore, double nextScore, double temp, std::mt19937_64& rng) {
    if (nextScore + EPS < curScore) return true;
    if (!std::isfinite(nextScore)) return false;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double scale = std::max(1e-9, std::fabs(curScore));
    double p = std::exp(-(nextScore - curScore) / (scale * std::max(temp, 1e-9)));
    return U(rng) < p;
}

inline bool try_add_random_compatible(const std::vector<Candidate>& cands,
                                      std::vector<int>& selected,
                                      bool useGapRules,
                                      std::mt19937_64& rng,
                                      int tries) {
    if (cands.empty()) return false;
    std::uniform_int_distribution<int> D(0, static_cast<int>(cands.size()) - 1);
    for (int t = 0; t < tries; ++t) {
        int idx = D(rng);
        if (compatible_with_selected(cands, idx, selected, useGapRules)) {
            selected.push_back(idx);
            return true;
        }
    }
    return false;
}

inline Solution anneal_chain(const std::vector<Candidate>& cands,
                             double usableArea,
                             bool useGapRules,
                             const Solution& start,
                             int iterations,
                             std::uint64_t seed) {
    if (iterations <= 0) iterations = 10000;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    std::vector<int> selected = start.selected;
    Solution current = evaluate_solution(cands, selected, usableArea);
    Solution best = current;

    for (int it = 0; it < iterations; ++it) {
        double f = static_cast<double>(it) / std::max(1, iterations - 1);
        double temp = 0.08 * std::pow(0.001 / 0.08, f);

        std::vector<int> next = selected;
        bool ok = false;
        double move = U(rng);

        if (move < 0.20 && !next.empty()) {
            std::uniform_int_distribution<int> P(0, static_cast<int>(next.size()) - 1);
            next.erase(next.begin() + P(rng));
            ok = true;
        } else if (move < 0.50) {
            ok = try_add_random_compatible(cands, next, useGapRules, rng, 1000);
        } else if (move < 0.80 && !next.empty()) {
            std::uniform_int_distribution<int> P(0, static_cast<int>(next.size()) - 1);
            next.erase(next.begin() + P(rng));
            ok = try_add_random_compatible(cands, next, useGapRules, rng, 1000);
            if (!ok) ok = true;
        } else if (!next.empty()) {
            int maxRemove = std::min(3, static_cast<int>(next.size()));
            std::uniform_int_distribution<int> K(1, maxRemove);
            int removeCount = K(rng);
            for (int k = 0; k < removeCount && !next.empty(); ++k) {
                std::uniform_int_distribution<int> P(0, static_cast<int>(next.size()) - 1);
                next.erase(next.begin() + P(rng));
            }
            int added = 0;
            for (int k = 0; k < 5; ++k) {
                if (try_add_random_compatible(cands, next, useGapRules, rng, 1000)) ++added;
            }
            ok = (added > 0 || removeCount > 0);
        }

        if (!ok) continue;
        Solution trial = evaluate_solution(cands, next, usableArea);
        if (accept_sa(current.score, trial.score, temp, rng)) {
            selected.swap(next);
            current = trial;
            if (current.score + EPS < best.score) best = current;
        }
    }

    prune_solution_by_true_objective(best.selected, cands, usableArea);
    return evaluate_solution(cands, best.selected, usableArea);
}

inline Solution run_parallel_sa(const std::vector<Candidate>& cands,
                                double usableArea,
                                bool useGapRules,
                                const Solution& greedyBest,
                                int chains,
                                int iterations,
                                std::uint64_t seed) {
    if (chains <= 0) chains = std::max(1, num_threads_available());
    Solution globalBest = greedyBest;

#pragma omp parallel
    {
        Solution localBest;
#pragma omp for schedule(dynamic)
        for (int c = 0; c < chains; ++c) {
            std::uint64_t s = seed + 9176ULL * static_cast<std::uint64_t>(c + 1);
            Solution start = greedyBest;
            if (c > 0) start = greedy_run(cands, usableArea, useGapRules, s ^ 0xA5A5A5A5ULL, c % 8);
            Solution sol = anneal_chain(cands, usableArea, useGapRules, start, iterations, s);
            if (sol.score < localBest.score) localBest = sol;
        }
#pragma omp critical
        {
            if (localBest.score < globalBest.score) globalBest = localBest;
        }
    }
    return globalBest;
}

inline Solution run_full_optimizer(const Instance& ins,
                                   const OptimizerParams& params,
                                   std::vector<Candidate>& outCandidates,
                                   std::uint64_t seed) {
    double ua = usable_area(ins);
    std::cerr << "Generating candidates...\n";
    generate_candidates(ins, params.gridStep, params.angleStep, params.useGapRules,
                        params.maxCandidates, outCandidates);
    std::cerr << "Feasible candidates: " << outCandidates.size() << "\n";
    if (outCandidates.empty()) throw std::runtime_error("No feasible candidates. Try smaller gridStep or disable gap rules.");

    std::cerr << "Running greedy restarts...\n";
    Solution greedy = run_parallel_greedy(outCandidates, ua, params.useGapRules,
                                          params.greedyRestarts, seed);
    print_solution_summary("Greedy", greedy);

    std::cerr << "Running simulated annealing chains...\n";
    Solution best = run_parallel_sa(outCandidates, ua, params.useGapRules, greedy,
                                    params.saChains, params.saIterations,
                                    seed ^ 0x9E3779B97F4A7C15ULL);
    print_solution_summary("Best", best);
    return best;
}

} // namespace whopt2
