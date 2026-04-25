#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace wh {

static constexpr double EPS = 1e-7;
static constexpr double INF = 1e100;
static constexpr double PI = 3.141592653589793238462643383279502884;

struct Point {
    double x = 0.0;
    double y = 0.0;
};

using Poly = std::vector<Point>;

struct BayType {
    int id = 0;
    double width = 0.0;
    double depth = 0.0;
    double height = 0.0;
    double gap = 0.0;
    int loads = 0;
    double price = 0.0;
};

struct Obstacle {
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double depth = 0.0;
};

struct CeilingPoint {
    double x = 0.0;
    double height = 0.0;
};

struct Instance {
    Poly warehouse;
    std::vector<Obstacle> obstacles;
    std::vector<BayType> bays;
    std::vector<CeilingPoint> ceiling;
    double warehouseArea = 0.0;
    double obstacleArea = 0.0;
    double usableArea = 0.0;
};

struct Candidate {
    int candId = -1;
    int bayId = 0;
    double x = 0.0;
    double y = 0.0;
    double rotation = 0.0;
    int loads = 0;
    double price = 0.0;
    double physicalArea = 0.0;
    double accountedArea = 0.0;
    Poly footprint;  // actual bay rectangle: width x depth
    Poly clearance;  // bay rectangle plus frontal gap: width x (depth + gap)
};

struct Solution {
    std::vector<int> candidateIds;
    double totalPrice = 0.0;
    int totalLoads = 0;
    double usedArea = 0.0;
    double logScore = INF;
};

struct Options {
    double gridStep = 500.0;
    int angleStep = 90;              // use 90 for fast orthogonal packing; 10 or 15 for angled search
    int restarts = 64;
    unsigned seed = 1234567u;
    bool accountGapArea = false;     // official score usually uses physical bay area; turn on if gap counts
    bool writeHeader = false;
    double badAcceptProbability = 0.015; // small random acceptance during construction
};

inline std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

inline std::vector<double> parseCsvNumbers(const std::string& line) {
    std::vector<double> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        cell = trim(cell);
        if (!cell.empty()) out.push_back(std::stod(cell));
    }
    return out;
}

inline std::vector<std::vector<double>> readNumberCsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open file: " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) continue;
        rows.push_back(parseCsvNumbers(line));
    }
    return rows;
}

inline std::string joinPath(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    char c = dir.back();
    if (c == '/' || c == '\\') return dir + file;
    return dir + "/" + file;
}

inline Point operator+(const Point& a, const Point& b) { return {a.x + b.x, a.y + b.y}; }
inline Point operator-(const Point& a, const Point& b) { return {a.x - b.x, a.y - b.y}; }
inline Point operator*(const Point& a, double k) { return {a.x * k, a.y * k}; }
inline double dot(Point a, Point b) { return a.x * b.x + a.y * b.y; }
inline double cross(Point a, Point b) { return a.x * b.y - a.y * b.x; }
inline double cross(Point a, Point b, Point c) { return cross(b - a, c - a); }

inline double polygonAreaSigned(const Poly& p) {
    double s = 0.0;
    for (size_t i = 0, n = p.size(); i < n; ++i) {
        const Point& a = p[i];
        const Point& b = p[(i + 1) % n];
        s += a.x * b.y - a.y * b.x;
    }
    return 0.5 * s;
}

inline double polygonArea(const Poly& p) {
    return std::abs(polygonAreaSigned(p));
}

inline Poly obstaclePoly(const Obstacle& o) {
    return {{o.x, o.y}, {o.x + o.width, o.y}, {o.x + o.width, o.y + o.depth}, {o.x, o.y + o.depth}};
}

inline bool onSegment(Point a, Point b, Point p) {
    if (std::abs(cross(a, b, p)) > EPS) return false;
    return std::min(a.x, b.x) - EPS <= p.x && p.x <= std::max(a.x, b.x) + EPS &&
           std::min(a.y, b.y) - EPS <= p.y && p.y <= std::max(a.y, b.y) + EPS;
}

inline int signWithEps(double v) {
    if (v > EPS) return 1;
    if (v < -EPS) return -1;
    return 0;
}

// Proper intersection means the two segments cross through each other.
// Touching at endpoints / collinear contact is allowed for "inside warehouse" checks.
inline bool segmentsProperIntersect(Point a, Point b, Point c, Point d) {
    int s1 = signWithEps(cross(a, b, c));
    int s2 = signWithEps(cross(a, b, d));
    int s3 = signWithEps(cross(c, d, a));
    int s4 = signWithEps(cross(c, d, b));
    return (s1 * s2 < 0) && (s3 * s4 < 0);
}

inline bool pointInPolygonOrOnBoundary(const Poly& poly, Point p) {
    bool inside = false;
    const int n = static_cast<int>(poly.size());
    for (int i = 0, j = n - 1; i < n; j = i++) {
        Point a = poly[j];
        Point b = poly[i];
        if (onSegment(a, b, p)) return true;
        const bool crosses = ((a.y > p.y) != (b.y > p.y));
        if (crosses) {
            double xAtY = a.x + (b.x - a.x) * (p.y - a.y) / (b.y - a.y);
            if (xAtY > p.x + EPS) inside = !inside;
        }
    }
    return inside;
}

inline bool polygonInsideWarehouse(const Poly& poly, const Poly& warehouse) {
    for (Point p : poly) {
        if (!pointInPolygonOrOnBoundary(warehouse, p)) return false;
    }
    // A rectangle can have all corners inside a concave polygon while an edge crosses outside.
    for (size_t i = 0; i < poly.size(); ++i) {
        Point a = poly[i];
        Point b = poly[(i + 1) % poly.size()];
        for (size_t j = 0; j < warehouse.size(); ++j) {
            Point c = warehouse[j];
            Point d = warehouse[(j + 1) % warehouse.size()];
            if (segmentsProperIntersect(a, b, c, d)) return false;
        }
    }
    return true;
}

inline void projectOnAxis(const Poly& p, Point axis, double& lo, double& hi) {
    lo = hi = dot(p[0], axis);
    for (size_t i = 1; i < p.size(); ++i) {
        double v = dot(p[i], axis);
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
}

// Strict convex polygon overlap. Returns false when polygons only touch.
inline bool convexPolygonsOverlapStrict(const Poly& a, const Poly& b) {
    auto separatedOnAnyAxis = [&](const Poly& p, const Poly& q) -> bool {
        for (size_t i = 0; i < p.size(); ++i) {
            Point e = p[(i + 1) % p.size()] - p[i];
            Point axis{-e.y, e.x};
            double alo, ahi, blo, bhi;
            projectOnAxis(a, axis, alo, ahi);
            projectOnAxis(b, axis, blo, bhi);
            if (ahi <= blo + EPS || bhi <= alo + EPS) return true;
        }
        return false;
    };
    return !separatedOnAnyAxis(a, b) && !separatedOnAnyAxis(b, a);
}

inline std::pair<Point, Point> boundingBox(const Poly& p) {
    Point mn{INF, INF}, mx{-INF, -INF};
    for (Point q : p) {
        mn.x = std::min(mn.x, q.x); mn.y = std::min(mn.y, q.y);
        mx.x = std::max(mx.x, q.x); mx.y = std::max(mx.y, q.y);
    }
    return {mn, mx};
}

inline Point rotateLocal(double lx, double ly, double degrees) {
    double r = degrees * PI / 180.0;
    double c = std::cos(r), s = std::sin(r);
    return {c * lx - s * ly, s * lx + c * ly};
}

inline Poly makeRotatedRect(double x, double y, double width, double depth, double degrees) {
    std::array<Point, 4> local = {{{0, 0}, {width, 0}, {width, depth}, {0, depth}}};
    Poly poly;
    poly.reserve(4);
    for (Point p : local) {
        Point r = rotateLocal(p.x, p.y, degrees);
        poly.push_back({x + r.x, y + r.y});
    }
    return poly;
}

inline Candidate makeCandidate(const BayType& b, double x, double y, double rot, bool accountGapArea) {
    Candidate c;
    c.bayId = b.id;
    c.x = x;
    c.y = y;
    c.rotation = rot;
    c.loads = b.loads;
    c.price = b.price;
    c.physicalArea = b.width * b.depth;
    c.accountedArea = accountGapArea ? b.width * (b.depth + b.gap) : c.physicalArea;
    c.footprint = makeRotatedRect(x, y, b.width, b.depth, rot);
    c.clearance = makeRotatedRect(x, y, b.width, b.depth + b.gap, rot);
    return c;
}

inline double minCeilingForXRange(const std::vector<CeilingPoint>& ceiling, double xmin, double xmax) {
    if (ceiling.empty()) return INF;
    if (xmin > xmax) std::swap(xmin, xmax);
    std::vector<CeilingPoint> c = ceiling;
    std::sort(c.begin(), c.end(), [](const CeilingPoint& a, const CeilingPoint& b) { return a.x < b.x; });

    double ans = INF;
    for (size_t i = 0; i < c.size(); ++i) {
        double left = (i == 0 ? -INF : c[i].x);
        double right = (i + 1 < c.size() ? c[i + 1].x : INF);
        if (std::max(left, xmin) <= std::min(right, xmax) + EPS) {
            ans = std::min(ans, c[i].height);
        }
    }
    return ans;
}

inline bool staticFeasible(const Candidate& cand, const BayType& type, const Instance& in) {
    // Keep the actual bay and its required frontal gap inside the warehouse.
    if (!polygonInsideWarehouse(cand.clearance, in.warehouse)) return false;

    auto bb = boundingBox(cand.footprint);
    double minH = minCeilingForXRange(in.ceiling, bb.first.x, bb.second.x);
    if (type.height > minH + EPS) return false;

    for (const Obstacle& o : in.obstacles) {
        Poly op = obstaclePoly(o);
        // Gap is a free/aisle zone, so obstacles cannot be inside it either.
        if (convexPolygonsOverlapStrict(cand.clearance, op)) return false;
    }
    return true;
}

inline bool candidateConflictsWithSelected(const Candidate& cand,
                                           const std::vector<int>& selected,
                                           const std::vector<Candidate>& all) {
    for (int id : selected) {
        const Candidate& other = all[id];
        // Physical bay bodies cannot overlap.
        if (convexPolygonsOverlapStrict(cand.footprint, other.footprint)) return true;
        // No physical bay may enter another bay's frontal gap.
        if (convexPolygonsOverlapStrict(cand.footprint, other.clearance)) return true;
        if (convexPolygonsOverlapStrict(cand.clearance, other.footprint)) return true;
        // clearance-clearance overlap is allowed: two bays may share an aisle.
    }
    return false;
}

inline double solutionLogScore(double price, int loads, double area, double usableArea) {
    if (loads <= 0 || price <= 0.0 || usableArea <= 0.0) return INF;
    double areaFraction = std::max(0.0, std::min(1.0, area / usableArea));
    double ratio = price / static_cast<double>(loads);
    if (ratio <= 0.0) return INF;
    return (2.0 - areaFraction) * std::log(ratio);
}

inline double solutionScore(const Solution& s) {
    if (s.logScore >= 700.0) return INF;
    return std::exp(s.logScore);
}

inline void addCandidateToSolution(Solution& s, const Candidate& c, double usableArea) {
    s.candidateIds.push_back(c.candId);
    s.totalPrice += c.price;
    s.totalLoads += c.loads;
    s.usedArea += c.accountedArea;
    s.logScore = solutionLogScore(s.totalPrice, s.totalLoads, s.usedArea, usableArea);
}

inline void recomputeSolution(Solution& s, const std::vector<Candidate>& candidates, double usableArea) {
    s.totalPrice = 0.0;
    s.totalLoads = 0;
    s.usedArea = 0.0;
    for (int id : s.candidateIds) {
        const Candidate& c = candidates[id];
        s.totalPrice += c.price;
        s.totalLoads += c.loads;
        s.usedArea += c.accountedArea;
    }
    s.logScore = solutionLogScore(s.totalPrice, s.totalLoads, s.usedArea, usableArea);
}

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline double hash01(uint64_t x) {
    return (splitmix64(x) >> 11) * (1.0 / 9007199254740992.0);
}

inline void uniqueSorted(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    std::vector<double> out;
    out.reserve(v.size());
    for (double x : v) {
        if (out.empty() || std::abs(x - out.back()) > 1e-6) out.push_back(x);
    }
    v.swap(out);
}

inline Instance readInstance(const std::string& dir) {
    Instance in;

    for (const auto& r : readNumberCsv(joinPath(dir, "warehouse.csv"))) {
        if (r.size() < 2) continue;
        in.warehouse.push_back({r[0], r[1]});
    }
    if (in.warehouse.size() < 3) throw std::runtime_error("warehouse.csv must contain at least 3 points");

    for (const auto& r : readNumberCsv(joinPath(dir, "obstacles.csv"))) {
        if (r.size() < 4) continue;
        in.obstacles.push_back({r[0], r[1], r[2], r[3]});
    }

    for (const auto& r : readNumberCsv(joinPath(dir, "types_of_bays.csv"))) {
        if (r.size() < 7) continue;
        in.bays.push_back({static_cast<int>(std::llround(r[0])), r[1], r[2], r[3], r[4],
                           static_cast<int>(std::llround(r[5])), r[6]});
    }
    if (in.bays.empty()) throw std::runtime_error("types_of_bays.csv has no bay types");

    for (const auto& r : readNumberCsv(joinPath(dir, "ceiling.csv"))) {
        if (r.size() < 2) continue;
        in.ceiling.push_back({r[0], r[1]});
    }
    std::sort(in.ceiling.begin(), in.ceiling.end(), [](const CeilingPoint& a, const CeilingPoint& b) {
        return a.x < b.x;
    });

    in.warehouseArea = polygonArea(in.warehouse);
    in.obstacleArea = 0.0;
    for (const auto& o : in.obstacles) in.obstacleArea += std::abs(o.width * o.depth);
    in.usableArea = std::max(1.0, in.warehouseArea - in.obstacleArea);
    return in;
}

inline std::vector<double> buildAngles(int angleStep) {
    if (angleStep <= 0 || angleStep > 360) angleStep = 90;
    std::vector<double> a;
    for (int deg = 0; deg < 360; deg += angleStep) a.push_back(static_cast<double>(deg));
    // Always include the four cardinal rotations even if angleStep does not divide them.
    a.push_back(0); a.push_back(90); a.push_back(180); a.push_back(270);
    uniqueSorted(a);
    return a;
}

inline void addRangeGrid(std::vector<double>& values, double lo, double hi, double step) {
    if (step <= 0) return;
    double start = std::floor(lo / step) * step;
    double end = std::ceil(hi / step) * step;
    for (double v = start; v <= end + EPS; v += step) values.push_back(v);
}

inline std::vector<Candidate> generateCandidates(const Instance& in, const Options& opt) {
    auto bb = boundingBox(in.warehouse);
    double minX = bb.first.x, minY = bb.first.y, maxX = bb.second.x, maxY = bb.second.y;

    std::vector<double> xs, ys;
    addRangeGrid(xs, minX, maxX, opt.gridStep);
    addRangeGrid(ys, minY, maxY, opt.gridStep);

    // Geometry feature lines help the search snap bays to walls, obstacles, and ceiling changes.
    for (Point p : in.warehouse) { xs.push_back(p.x); ys.push_back(p.y); }
    for (const auto& o : in.obstacles) {
        xs.push_back(o.x); xs.push_back(o.x + o.width);
        ys.push_back(o.y); ys.push_back(o.y + o.depth);
    }
    for (const auto& c : in.ceiling) xs.push_back(c.x);

    uniqueSorted(xs);
    uniqueSorted(ys);
    const std::vector<double> angles = buildAngles(opt.angleStep);

    std::vector<Candidate> all;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<Candidate> local;
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int bi = 0; bi < static_cast<int>(in.bays.size()); ++bi) {
            const BayType& b = in.bays[bi];
            for (double angle : angles) {
                for (double x : xs) {
                    for (double y : ys) {
                        Candidate cand = makeCandidate(b, x, y, angle, opt.accountGapArea);
                        if (staticFeasible(cand, b, in)) {
                            local.push_back(std::move(cand));
                        }
                    }
                }
            }
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            all.insert(all.end(), local.begin(), local.end());
        }
    }

    for (int i = 0; i < static_cast<int>(all.size()); ++i) all[i].candId = i;
    return all;
}

inline double candidateOrderScore(const Candidate& c, const Instance& in, int restart, uint64_t seed) {
    double efficiency = static_cast<double>(c.loads) / std::max(1.0, c.price);
    double areaFrac = c.accountedArea / in.usableArea;
    double loadScore = static_cast<double>(c.loads) / 100.0;
    double rnd = hash01(seed ^ (static_cast<uint64_t>(restart) << 32) ^ static_cast<uint64_t>(c.candId));

    // Rotate through a few ranking styles to diversify restarts.
    switch (restart % 5) {
        case 0: return 10000.0 * efficiency + 0.10 * rnd;
        case 1: return 10000.0 * efficiency + 20.0 * areaFrac + 0.25 * rnd;
        case 2: return 50.0 * areaFrac + 5.0 * loadScore + rnd;
        case 3: return 10000.0 * efficiency + 5.0 * loadScore + 2.0 * rnd;
        default: return rnd;
    }
}

inline void pruneSolution(Solution& sol, const std::vector<Candidate>& candidates, double usableArea) {
    bool improved = true;
    while (improved) {
        improved = false;
        for (size_t i = 0; i < sol.candidateIds.size(); ++i) {
            const Candidate& c = candidates[sol.candidateIds[i]];
            double np = sol.totalPrice - c.price;
            int nl = sol.totalLoads - c.loads;
            double na = sol.usedArea - c.accountedArea;
            double newLog = solutionLogScore(np, nl, na, usableArea);
            if (newLog + 1e-10 < sol.logScore) {
                sol.candidateIds.erase(sol.candidateIds.begin() + static_cast<long>(i));
                recomputeSolution(sol, candidates, usableArea);
                improved = true;
                break;
            }
        }
    }
}

inline void addImprovingFillers(Solution& sol,
                                const std::vector<int>& order,
                                const std::vector<Candidate>& candidates,
                                double usableArea) {
    bool any = true;
    int passes = 0;
    while (any && passes < 3) {
        any = false;
        ++passes;
        for (int id : order) {
            const Candidate& c = candidates[id];
            if (candidateConflictsWithSelected(c, sol.candidateIds, candidates)) continue;
            double newLog = solutionLogScore(sol.totalPrice + c.price,
                                             sol.totalLoads + c.loads,
                                             sol.usedArea + c.accountedArea,
                                             usableArea);
            if (newLog + 1e-10 < sol.logScore || sol.totalLoads == 0) {
                addCandidateToSolution(sol, c, usableArea);
                any = true;
            }
        }
        pruneSolution(sol, candidates, usableArea);
    }
}

inline Solution constructSolutionForRestart(const Instance& in,
                                            const std::vector<Candidate>& candidates,
                                            const Options& opt,
                                            int restart) {
    std::vector<int> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        double sa = candidateOrderScore(candidates[a], in, restart, opt.seed);
        double sb = candidateOrderScore(candidates[b], in, restart, opt.seed);
        if (std::abs(sa - sb) > 1e-12) return sa > sb;
        return a < b;
    });

    Solution sol;
    const bool fillAllMode = (restart % 4 == 0); // some restarts maximize packing first, then prune by objective
    uint64_t rndState = opt.seed ^ (0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(restart + 1));

    for (int id : order) {
        const Candidate& c = candidates[id];
        if (candidateConflictsWithSelected(c, sol.candidateIds, candidates)) continue;

        if (fillAllMode || sol.totalLoads == 0) {
            addCandidateToSolution(sol, c, in.usableArea);
            continue;
        }

        double newLog = solutionLogScore(sol.totalPrice + c.price,
                                         sol.totalLoads + c.loads,
                                         sol.usedArea + c.accountedArea,
                                         in.usableArea);
        bool improves = newLog + 1e-10 < sol.logScore;
        rndState = splitmix64(rndState);
        bool randomBadAccept = hash01(rndState) < opt.badAcceptProbability;
        if (improves || randomBadAccept) {
            addCandidateToSolution(sol, c, in.usableArea);
        }
    }

    pruneSolution(sol, candidates, in.usableArea);
    addImprovingFillers(sol, order, candidates, in.usableArea);
    pruneSolution(sol, candidates, in.usableArea);
    return sol;
}

inline Solution solveInstance(const Instance& in, const Options& opt, std::vector<Candidate>& candidates) {
    candidates = generateCandidates(in, opt);
    if (candidates.empty()) return Solution{};

    Solution best;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        Solution localBest;
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int r = 0; r < opt.restarts; ++r) {
            Solution s = constructSolutionForRestart(in, candidates, opt, r);
            if (s.totalLoads > 0 && s.logScore < localBest.logScore) localBest = std::move(s);
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            if (localBest.totalLoads > 0 && localBest.logScore < best.logScore) best = std::move(localBest);
        }
    }
    return best;
}

inline void writeSolutionCsv(const std::string& path,
                             const Solution& sol,
                             const std::vector<Candidate>& candidates,
                             bool header) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write output file: " + path);
    if (header) out << "Id, X, Y, Rotation\n";
    out << std::fixed << std::setprecision(0);
    for (int id : sol.candidateIds) {
        const Candidate& c = candidates[id];
        out << c.bayId << ", " << c.x << ", " << c.y << ", " << c.rotation << "\n";
    }
}

inline void printSolutionSummary(const Instance& in, const Solution& sol, const std::vector<Candidate>& candidates) {
    std::cerr << "Selected bays: " << sol.candidateIds.size() << "\n";
    std::cerr << "Total loads:   " << sol.totalLoads << "\n";
    std::cerr << "Total price:   " << sol.totalPrice << "\n";
    std::cerr << "Used area:     " << sol.usedArea << " / " << in.usableArea
              << " = " << (100.0 * sol.usedArea / in.usableArea) << "%\n";
    std::cerr << "Log score:     " << sol.logScore << "\n";
    std::cerr << "Score:         " << solutionScore(sol) << "\n";

    std::vector<int> byType;
    for (int id : sol.candidateIds) byType.push_back(candidates[id].bayId);
    std::sort(byType.begin(), byType.end());
    std::cerr << "By bay id:     ";
    for (size_t i = 0; i < byType.size();) {
        size_t j = i;
        while (j < byType.size() && byType[j] == byType[i]) ++j;
        std::cerr << byType[i] << "x" << (j - i) << " ";
        i = j;
    }
    std::cerr << "\n";
}

inline Options parseOptions(int argc, char** argv, int startIndex) {
    Options opt;
    for (int i = startIndex; i < argc; ++i) {
        std::string a = argv[i];
        auto needValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value after " + name);
            return argv[++i];
        };
        if (a == "--grid") opt.gridStep = std::stod(needValue(a));
        else if (a == "--angle-step") opt.angleStep = std::stoi(needValue(a));
        else if (a == "--restarts") opt.restarts = std::stoi(needValue(a));
        else if (a == "--seed") opt.seed = static_cast<unsigned>(std::stoul(needValue(a)));
        else if (a == "--area-with-gap") opt.accountGapArea = true;
        else if (a == "--header") opt.writeHeader = true;
        else if (a == "--bad-accept") opt.badAcceptProbability = std::stod(needValue(a));
        else throw std::runtime_error("Unknown option: " + a);
    }
    if (opt.gridStep <= 0) throw std::runtime_error("--grid must be positive");
    if (opt.angleStep <= 0 || opt.angleStep > 360) throw std::runtime_error("--angle-step must be in 1..360");
    if (opt.restarts <= 0) throw std::runtime_error("--restarts must be positive");
    return opt;
}

} // namespace wh
