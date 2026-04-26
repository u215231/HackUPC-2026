#pragma once

// ================================================================
// optimizer_algorithms_gap.hpp
//
// This file contains ONLY optimization algorithms:
//   PART 1: candidate generation
//   PART 2: greedy construction with OpenMP parallel restarts
//   PART 3: simulated annealing with OpenMP parallel chains
//
// Helper functions for geometry, gaps, score calculation, and I/O are
// located in warehouse_helpers_gap.hpp.
// ================================================================

#include "warehouse_helpers.hpp"

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
    bool runSa = true;
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
    // This matters for the FRONT GAP: rotation changes where the aisle/gap points.
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
                           double noise,
                           int inner_threads) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    const int n = static_cast<int>(cands.size());

    double maxEff = 1e-9;
    double maxArea = 1e-9;
    double maxLoads = 1e-9;
    double maxWallDist = 1e-9;
    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();

    for (const Candidate& c : cands) {
        maxEff = std::max(maxEff, c.loads / std::max(1.0, c.price));
        maxArea = std::max(maxArea, c.bayArea);
        maxLoads = std::max(maxLoads, c.loads);
        maxWallDist = std::max(maxWallDist, c.wallDist);
        minX = std::min(minX, c.x);
        minY = std::min(minY, c.y);
        maxX = std::max(maxX, c.x);
        maxY = std::max(maxY, c.y);
    }
    double rangeX = std::max(1.0, maxX - minX);
    double rangeY = std::max(1.0, maxY - minY);

    std::vector<double> base_key(n);
    std::vector<int> cardinal_cands;
    std::vector<int> non_cardinal_cands;

    for (int i = 0; i < n; ++i) {
        const Candidate& c = cands[i];
        double eff   = (c.loads / std::max(1.0, c.price)) / maxEff;
        double area  = c.bayArea / maxArea;
        double loads = c.loads / maxLoads;
        
        double wall_score = 1.0 - (c.wallDist / std::max(1.0, maxWallDist));
        double pos_x = (c.x - minX) / rangeX;
        double pos_y = (c.y - minY) / rangeY;
        double pos_score = 1.0 - ((pos_x + pos_y) / 2.0);
        
        // Bonus for touching walls: 1.0 if it touches 2 walls (corner), 0.5 if it touches 1 wall, 0 otherwise
        double corner_bonus = (c.wallTouches >= 2) ? 1.0 : ((c.wallTouches == 1) ? 0.5 : 0.0);
        
        base_key[i] = 0.50 * eff + 0.15 * area + 0.05 * loads + 0.30 * corner_bonus + 0.10 * wall_score + 0.05 * pos_score;
        if (noise > 0.0) base_key[i] += noise * U(rng);

        double mod = std::fmod(c.rotation, 90.0);
        if (mod < 1e-5 || mod > 90.0 - 1e-5) {
            cardinal_cands.push_back(i);
        } else {
            non_cardinal_cands.push_back(i);
        }
    }

    auto cmp = [&](int a, int b) { return base_key[a] > base_key[b]; };
    std::sort(cardinal_cands.begin(), cardinal_cands.end(), cmp);
    std::sort(non_cardinal_cands.begin(), non_cardinal_cands.end(), cmp);

    std::vector<int> selected;
    selected.reserve(512);

    auto greedy_phase = [&](std::vector<int>& active_cands) {
        while (!active_cands.empty()) {
            int batch_size = std::min<int>(active_cands.size(), std::max(1000, 4000 / inner_threads * inner_threads)); 
            
            struct EvalResult {
                int idx = -1;
                double score = -1.0;
                bool compatible = false;
            };
            std::vector<EvalResult> results(batch_size);

            #pragma omp parallel for num_threads(inner_threads) schedule(dynamic, 32)
            for (int i = 0; i < batch_size; ++i) {
                int cand_idx = active_cands[i];
                results[i].idx = cand_idx;
                
                if (!compatible_with_selected(cands, cand_idx, selected, useGapRules)) {
                    results[i].compatible = false;
                    continue;
                }
                results[i].compatible = true;
                
                double score = base_key[cand_idx];
                double bonus = 0.0;
                bool gap_shared = false;
                bool touching = false;
                
                for (int s_idx : selected) {
                    const Candidate& other = cands[s_idx];
                    
                    if (!gap_shared && has_front_gap(cands[cand_idx]) && has_front_gap(other)) {
                        if (convex_quads_interiors_overlap(cands[cand_idx].gapZone, other.gapZone)) {
                            gap_shared = true;
                        }
                    }
                    
                    if (!touching) {
                        if (convex_quads_close(cands[cand_idx].footprint, other.footprint, 50.0)) {
                            touching = true;
                        }
                    }
                    if (gap_shared && touching) break;
                }
                
                if (gap_shared) bonus += 0.15;
                if (touching) bonus += 0.10;
                
                results[i].score = score + bonus;
            }

            int next_best = -1;
            double max_val = -1.0;
            int next_best_i = -1;
            
            for (int i = 0; i < batch_size; ++i) {
                if (results[i].compatible && results[i].score > max_val) {
                    max_val = results[i].score;
                    next_best = results[i].idx;
                    next_best_i = i;
                }
            }
            
            if (next_best != -1) {
                selected.push_back(next_best);
                active_cands.erase(active_cands.begin() + next_best_i);
                
                std::vector<int> new_active;
                new_active.reserve(active_cands.size());
                
                int n_active = active_cands.size();
                std::vector<uint8_t> keep(n_active, 1);
                
                #pragma omp parallel for num_threads(inner_threads) schedule(static)
                for (int i = 0; i < n_active; ++i) {
                    if (candidates_conflict(cands[active_cands[i]], cands[next_best], useGapRules)) {
                        keep[i] = 0;
                    }
                }
                
                for (int i = 0; i < n_active; ++i) {
                    if (keep[i]) new_active.push_back(active_cands[i]);
                }
                active_cands = std::move(new_active);
            } else {
                active_cands.erase(active_cands.begin(), active_cands.begin() + batch_size);
            }
        }
    };

    // Phase 1: Solo tratar ángulos cardinales (0, 90, 180, 270)
    greedy_phase(cardinal_cands);

    // Tras la fase 1, podamos y filtramos la lista no cardinal
    prune_solution(selected, cands, usableArea);

    std::vector<int> valid_non_cardinal;
    valid_non_cardinal.reserve(non_cardinal_cands.size());
    
    int n_non_card = non_cardinal_cands.size();
    std::vector<uint8_t> keep_nc(n_non_card, 1);
    
    #pragma omp parallel for num_threads(inner_threads) schedule(static)
    for (int i = 0; i < n_non_card; ++i) {
        if (!compatible_with_selected(cands, non_cardinal_cands[i], selected, useGapRules)) {
            keep_nc[i] = 0;
        }
    }
    for (int i = 0; i < n_non_card; ++i) {
        if (keep_nc[i]) valid_non_cardinal.push_back(non_cardinal_cands[i]);
    }

    // Phase 2: Rellenar con ángulos oblicuos si quedan huecos
    greedy_phase(valid_non_cardinal);

    prune_solution(selected, cands, usableArea);

    Solution cur = evaluate_solution(cands, selected, usableArea);
    std::unordered_set<int> inSolution(selected.begin(), selected.end());

    // Second pass final: añadir bays individuales si mejoran la curva no lineal
    auto try_add_remaining = [&](const std::vector<int>& pool) {
        for (int idx : pool) {
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
    };

    try_add_remaining(cardinal_cands);
    try_add_remaining(non_cardinal_cands);

    return evaluate_solution(cands, selected, usableArea);
}

inline Solution run_parallel_greedy(const std::vector<Candidate>& cands,
                                    double usableArea,
                                    bool useGapRules,
                                    int restarts,
                                    std::uint64_t baseSeed) {
#ifdef _OPENMP
    omp_set_max_active_levels(2);
#endif

    int inner_threads = thread_count();

    std::uint64_t seed = baseSeed + 1000003ULL;
    // Solo un Greedy Determinista (noise = 0.0) utilizando todos los hilos
    Solution best = greedy_run(cands, usableArea, useGapRules, seed, 0.0, inner_threads);

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

    int num_threads = thread_count();
    std::vector<std::mt19937_64> rngs(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        rngs[i].seed(seed + 9176ULL * (i + 1));
    }

    std::vector<int> selected = start.selected;
    Solution current = evaluate_solution(cands, selected, usableArea);
    Solution best = current;

    for (int it = 0; it < iterations; ++it) {
        double frac = static_cast<double>(it) / std::max(1, iterations - 1);
        double temperature = startTemp * std::pow(endTemp / startTemp, frac);

        struct MoveResult {
            std::vector<int> next;
            Solution trial;
            bool valid = false;
        };
        std::vector<MoveResult> moves(num_threads);

        #pragma omp parallel for num_threads(num_threads)
        for (int t = 0; t < num_threads; ++t) {
            auto& rng = rngs[t];
            std::uniform_real_distribution<double> U(0.0, 1.0);
            std::uniform_int_distribution<int> candDist(0, static_cast<int>(cands.size()) - 1);

            std::vector<int> next = selected;
            bool generatedMove = false;
            double moveType = U(rng);

            if (moveType < 0.30 && !next.empty()) {
                std::uniform_int_distribution<int> posDist(0, static_cast<int>(next.size()) - 1);
                int pos = posDist(rng);
                next.erase(next.begin() + pos);
                generatedMove = true;
            } else if (moveType < 0.62) {
                for (int tries = 0; tries < 50 && !generatedMove; ++tries) {
                    int idx = candDist(rng);
                    if (compatible_with_selected(cands, idx, next, useGapRules)) {
                        next.push_back(idx);
                        generatedMove = true;
                    }
                }
            } else if (!next.empty()) {
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
                if (!generatedMove) generatedMove = true;
            }

            if (generatedMove) {
                moves[t].next = std::move(next);
                moves[t].trial = evaluate_solution(cands, moves[t].next, usableArea);
                moves[t].valid = true;
            }
        }

        int bestMoveIdx = -1;
        double bestAcceptedScore = std::numeric_limits<double>::infinity();

        for (int t = 0; t < num_threads; ++t) {
            if (!moves[t].valid) continue;
            
            if (accept_sa_move(current.score, moves[t].trial.score, temperature, rngs[0])) {
                if (moves[t].trial.score < bestAcceptedScore) {
                    bestAcceptedScore = moves[t].trial.score;
                    bestMoveIdx = t;
                }
            }
        }

        if (bestMoveIdx != -1) {
            selected = std::move(moves[bestMoveIdx].next);
            current = std::move(moves[bestMoveIdx].trial);
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
                                int iterations,
                                std::uint64_t baseSeed) {
    // Only ONE chain as requested, parallelized internally
    std::uint64_t seed = baseSeed + 9176ULL;
    return anneal_chain(cands, usableArea, useGapRules, greedyBest, iterations, seed);
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

    Solution best = greedy;
    if (params.runSa) {
        std::cerr << "Running simulated annealing chains...\n";
        best = run_parallel_sa(
            outCandidates,
            area,
            params.useGapRules,
            greedy,
            params.saIterations,
            seed ^ 0x9E3779B97F4A7C15ULL
        );
        print_solution_summary("Best", best);
    }

    return best;
}

} // namespace whopt
