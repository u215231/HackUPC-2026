#pragma once

// ================================================================
// warehouse_helpers_gap.hpp
//
// This file contains ONLY helper code:
//   - data structures
//   - CSV input/output
//   - geometry primitives
//   - bay placement validity checks
//   - corrected FRONT gap validation:
//       * gap is only the rectangle in front of the bay
//       * no bay, obstacle, or wall may occupy a gap
//       * gap-gap overlap is allowed
//   - bay/bay conflict checks with corrected FRONT gap handling
//   - objective function evaluation
//
// The greedy algorithm and simulated annealing are NOT in this file.
// They are implemented in optimizer_algorithms_gap.hpp.
// ================================================================

#include <algorithm>
#include <array>
#include <cmath>
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

namespace whopt {

constexpr double EPS = 1e-7;
constexpr double PI  = 3.141592653589793238462643383279502884;

inline int thread_count() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// -----------------------------
// Basic data structures
// -----------------------------

struct Point {
    double x = 0.0;
    double y = 0.0;
};

inline Point operator+(const Point& a, const Point& b) { return {a.x + b.x, a.y + b.y}; }
inline Point operator-(const Point& a, const Point& b) { return {a.x - b.x, a.y - b.y}; }
inline Point operator*(const Point& a, double k)       { return {a.x * k, a.y * k}; }
inline double dot(const Point& a, const Point& b)      { return a.x * b.x + a.y * b.y; }
inline double cross(const Point& a, const Point& b)    { return a.x * b.y - a.y * b.x; }
inline double cross(const Point& a, const Point& b, const Point& c) { return cross(b - a, c - a); }

struct BayType {
    int id = 0;
    double width = 0.0;
    double depth = 0.0;
    double height = 0.0;
    double gap = 0.0;
    double loads = 0.0;
    double price = 0.0;
};

struct Obstacle {
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double depth = 0.0;
};

struct CeilingBreak {
    // Step function: from this x until the next break, ceiling height = height.
    double x = 0.0;
    double height = 0.0;
};

struct Instance {
    std::vector<Point> warehouse;
    std::vector<Obstacle> obstacles;
    std::vector<CeilingBreak> ceiling;
    std::vector<BayType> bays;
};

struct QuadBox {
    std::array<Point, 4> corners{};
    double minX = 0.0;
    double maxX = 0.0;
    double minY = 0.0;
    double maxY = 0.0;
};

struct Candidate {
    int bayIndex = -1;      // index inside Instance::bays
    int bayId = -1;         // actual id written to output

    // IMPORTANT:
    // x,y is the LOCAL bottom-left corner of the bay.
    // Rotation is applied around this point.
    double x = 0.0;
    double y = 0.0;
    double rotation = 0.0;

    double loads = 0.0;
    double price = 0.0;
    double bayArea = 0.0;   // objective area: width * depth, gap not counted by default
    double gapArea = 0.0;   // width * gap, used only for validation/collision
    double wallDist = 0.0;  // distance to nearest boundary
    int wallTouches = 0;    // number of bay edges touching a boundary

    QuadBox footprint;      // physical bay rectangle: local [0,width] x [0,depth]
    QuadBox gapZone;        // EMPTY FRONT GAP ONLY: local [0,width] x [depth, depth+gap]
};

struct Solution {
    std::vector<int> selected; // indices into vector<Candidate>
    double totalPrice = 0.0;
    double totalLoads = 0.0;
    double usedArea = 0.0;
    double areaPct = 0.0;      // fraction 0..1, not 0..100
    double score = std::numeric_limits<double>::infinity();
};

// -----------------------------
// CSV parsing helpers
// -----------------------------

inline std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

inline std::vector<double> parse_csv_numbers(const std::string& line) {
    std::vector<double> out;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (token.empty()) continue;
        try {
            size_t used = 0;
            double v = std::stod(token, &used);
            if (used > 0) out.push_back(v);
        } catch (...) {
            // Header/non-numeric cells are ignored.
        }
    }
    return out;
}

inline std::string join_path(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    char last = dir.back();
    if (last == '/' || last == '\\') return dir + file;
    return dir + "/" + file;
}

inline std::vector<Point> read_warehouse_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open " + path);
    std::vector<Point> poly;
    std::string line;
    while (std::getline(in, line)) {
        auto nums = parse_csv_numbers(line);
        if (nums.size() >= 2) poly.push_back({nums[0], nums[1]});
    }
    if (poly.size() < 3) throw std::runtime_error("Warehouse polygon needs at least 3 points");
    return poly;
}

inline std::vector<Obstacle> read_obstacles_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open " + path);
    std::vector<Obstacle> obs;
    std::string line;
    while (std::getline(in, line)) {
        auto nums = parse_csv_numbers(line);
        if (nums.size() >= 4) obs.push_back({nums[0], nums[1], nums[2], nums[3]});
    }
    return obs;
}

inline std::vector<CeilingBreak> read_ceiling_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open " + path);
    std::vector<CeilingBreak> c;
    std::string line;
    while (std::getline(in, line)) {
        auto nums = parse_csv_numbers(line);
        if (nums.size() >= 2) c.push_back({nums[0], nums[1]});
    }
    std::sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a.x < b.x; });
    if (c.empty()) throw std::runtime_error("ceiling.csv has no valid rows");
    return c;
}

inline std::vector<BayType> read_bays_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open " + path);
    std::vector<BayType> bays;
    std::string line;
    while (std::getline(in, line)) {
        auto nums = parse_csv_numbers(line);
        if (nums.size() >= 7) {
            bays.push_back({
                static_cast<int>(std::llround(nums[0])),
                nums[1], nums[2], nums[3], nums[4], nums[5], nums[6]
            });
        }
    }
    if (bays.empty()) throw std::runtime_error("No bay types found in " + path);
    return bays;
}

inline Instance read_instance(const std::string& caseDir) {
    Instance ins;
    ins.warehouse = read_warehouse_csv(join_path(caseDir, "warehouse.csv"));
    ins.obstacles = read_obstacles_csv(join_path(caseDir, "obstacles.csv"));
    ins.ceiling   = read_ceiling_csv(join_path(caseDir, "ceiling.csv"));
    ins.bays      = read_bays_csv(join_path(caseDir, "types_of_bays.csv"));
    return ins;
}

// -----------------------------
// Geometry helpers
// -----------------------------

inline double polygon_area_signed(const std::vector<Point>& p) {
    double a = 0.0;
    for (size_t i = 0, n = p.size(); i < n; ++i) {
        const Point& u = p[i];
        const Point& v = p[(i + 1) % n];
        a += u.x * v.y - u.y * v.x;
    }
    return 0.5 * a;
}

inline double polygon_area_abs(const std::vector<Point>& p) {
    return std::abs(polygon_area_signed(p));
}

inline bool on_segment(const Point& a, const Point& b, const Point& p) {
    if (std::abs(cross(a, b, p)) > EPS) return false;
    return p.x >= std::min(a.x, b.x) - EPS && p.x <= std::max(a.x, b.x) + EPS &&
           p.y >= std::min(a.y, b.y) - EPS && p.y <= std::max(a.y, b.y) + EPS;
}

inline bool point_in_polygon_inclusive(const std::vector<Point>& poly, const Point& p) {
    bool inside = false;
    for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
        const Point& a = poly[j];
        const Point& b = poly[i];
        if (on_segment(a, b, p)) return true;
        bool intersect = ((a.y > p.y) != (b.y > p.y)) &&
                         (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x);
        if (intersect) inside = !inside;
    }
    return inside;
}

inline bool segment_proper_intersection(const Point& a, const Point& b, const Point& c, const Point& d) {
    double o1 = cross(a, b, c);
    double o2 = cross(a, b, d);
    double o3 = cross(c, d, a);
    double o4 = cross(c, d, b);
    return ((o1 > EPS && o2 < -EPS) || (o1 < -EPS && o2 > EPS)) &&
           ((o3 > EPS && o4 < -EPS) || (o3 < -EPS && o4 > EPS));
}

inline std::array<Point, 4> rectangle_corners_from_local_bottom_left(
    double x, double y, double width, double depth, double rotationDeg) {
    // Coordinate convention:
    //   (x,y) = local bottom-left corner of the bay.
    //   rotation = counter-clockwise degrees around this exact point.
    double r = rotationDeg * PI / 180.0;
    Point ux{std::cos(r), std::sin(r)};       // local width direction
    Point uy{-std::sin(r), std::cos(r)};      // local depth/front direction
    Point p0{x, y};
    Point p1 = p0 + ux * width;
    Point p2 = p1 + uy * depth;
    Point p3 = p0 + uy * depth;
    return {{p0, p1, p2, p3}};
}

inline std::array<Point, 4> rectangle_corners_from_local_rect(
    double x, double y,
    double localX0, double localY0,
    double localX1, double localY1,
    double rotationDeg) {
    // Creates a rotated rectangle described in the bay's LOCAL coordinates.
    // This is used for the FRONT GAP:
    //   local x: [0, width]
    //   local y: [depth, depth + gap]
    // The same pivot/origin is used: (x,y) = local bottom-left of the bay.
    double r = rotationDeg * PI / 180.0;
    Point ux{std::cos(r), std::sin(r)};       // local width direction
    Point uy{-std::sin(r), std::cos(r)};      // local depth/front direction
    Point p0{x, y};

    Point a = p0 + ux * localX0 + uy * localY0;
    Point b = p0 + ux * localX1 + uy * localY0;
    Point c = p0 + ux * localX1 + uy * localY1;
    Point d = p0 + ux * localX0 + uy * localY1;
    return {{a, b, c, d}};
}

inline QuadBox make_quad_box(const std::array<Point, 4>& corners) {
    QuadBox q;
    q.corners = corners;
    q.minX = q.maxX = corners[0].x;
    q.minY = q.maxY = corners[0].y;
    for (const auto& p : corners) {
        q.minX = std::min(q.minX, p.x);
        q.maxX = std::max(q.maxX, p.x);
        q.minY = std::min(q.minY, p.y);
        q.maxY = std::max(q.maxY, p.y);
    }
    return q;
}

inline QuadBox obstacle_box(const Obstacle& o) {
    return make_quad_box({{{o.x, o.y}, {o.x + o.width, o.y}, {o.x + o.width, o.y + o.depth}, {o.x, o.y + o.depth}}});
}

inline bool bbox_positive_overlap(const QuadBox& a, const QuadBox& b) {
    bool xOverlap = std::min(a.maxX, b.maxX) - std::max(a.minX, b.minX) > EPS;
    bool yOverlap = std::min(a.maxY, b.maxY) - std::max(a.minY, b.minY) > EPS;
    return xOverlap && yOverlap;
}

inline bool convex_quads_interiors_overlap(const QuadBox& qa, const QuadBox& qb) {
    if (!bbox_positive_overlap(qa, qb)) return false;
    const auto& a = qa.corners;
    const auto& b = qb.corners;

    auto separated_on_axis = [&](Point axis) -> bool {
        double amin = dot(a[0], axis), amax = amin;
        double bmin = dot(b[0], axis), bmax = bmin;
        for (int k = 1; k < 4; ++k) {
            double av = dot(a[k], axis);
            double bv = dot(b[k], axis);
            amin = std::min(amin, av); amax = std::max(amax, av);
            bmin = std::min(bmin, bv); bmax = std::max(bmax, bv);
        }

        // CRITICAL RULE FOR YOUR CONTEST:
        // Borders are allowed to overlap/touch.
        // Therefore, zero-area intersection is NOT a conflict.
        return std::min(amax, bmax) - std::max(amin, bmin) <= EPS;
    };

    for (int i = 0; i < 4; ++i) {
        Point e = a[(i + 1) % 4] - a[i];
        if (separated_on_axis({-e.y, e.x})) return false;
    }
    for (int i = 0; i < 4; ++i) {
        Point e = b[(i + 1) % 4] - b[i];
        if (separated_on_axis({-e.y, e.x})) return false;
    }
    return true;
}


inline bool convex_quads_close(const QuadBox& qa, const QuadBox& qb, double tol) {
    if (std::min(qa.maxX, qb.maxX) - std::max(qa.minX, qb.minX) <= -tol) return false;
    if (std::min(qa.maxY, qb.maxY) - std::max(qa.minY, qb.minY) <= -tol) return false;

    const auto& a = qa.corners;
    const auto& b = qb.corners;

    auto separated_by_tol = [&](Point axis) -> bool {
        double len = std::hypot(axis.x, axis.y);
        if (len < EPS) return false;
        Point norm = {axis.x / len, axis.y / len};

        double amin = dot(a[0], norm), amax = amin;
        double bmin = dot(b[0], norm), bmax = bmin;
        for (int k = 1; k < 4; ++k) {
            double av = dot(a[k], norm);
            double bv = dot(b[k], norm);
            amin = std::min(amin, av); amax = std::max(amax, av);
            bmin = std::min(bmin, bv); bmax = std::max(bmax, bv);
        }
        return std::min(amax, bmax) - std::max(amin, bmin) <= -tol;
    };

    for (int i = 0; i < 4; ++i) {
        Point e = a[(i + 1) % 4] - a[i];
        if (separated_by_tol({-e.y, e.x})) return false;
    }
    for (int i = 0; i < 4; ++i) {
        Point e = b[(i + 1) % 4] - b[i];
        if (separated_by_tol({-e.y, e.x})) return false;
    }
    return true;
}

inline double point_segment_sq_dist(const Point& p, const Point& a, const Point& b) {
    double l2 = (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y);
    if (l2 < EPS) return (p.x - a.x)*(p.x - a.x) + (p.y - a.y)*(p.y - a.y);
    double t = std::max(0.0, std::min(1.0, dot(p - a, b - a) / l2));
    Point proj = a + (b - a) * t;
    return (p.x - proj.x)*(p.x - proj.x) + (p.y - proj.y)*(p.y - proj.y);
}

inline double min_dist_to_walls(const QuadBox& q, const Instance& ins) {
    Point center = {(q.corners[0].x + q.corners[2].x) / 2.0, (q.corners[0].y + q.corners[2].y) / 2.0};
    double min_sq_dist = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < ins.warehouse.size(); ++i) {
        Point a = ins.warehouse[i];
        Point b = ins.warehouse[(i + 1) % ins.warehouse.size()];
        min_sq_dist = std::min(min_sq_dist, point_segment_sq_dist(center, a, b));
    }
    for (const auto& obs : ins.obstacles) {
        Point o1 = {obs.x, obs.y};
        Point o2 = {obs.x + obs.width, obs.y};
        Point o3 = {obs.x + obs.width, obs.y + obs.depth};
        Point o4 = {obs.x, obs.y + obs.depth};
        min_sq_dist = std::min({min_sq_dist, 
            point_segment_sq_dist(center, o1, o2),
            point_segment_sq_dist(center, o2, o3),
            point_segment_sq_dist(center, o3, o4),
            point_segment_sq_dist(center, o4, o1)
        });
    }
    return std::sqrt(min_sq_dist);
}

inline int count_touching_walls(const QuadBox& q, const Instance& ins, double tol = 25.0) {
    int touches = 0;
    for (int i = 0; i < 4; ++i) {
        Point a = q.corners[i];
        Point b = q.corners[(i + 1) % 4];
        Point center = {(a.x + b.x) * 0.5, (a.y + b.y) * 0.5};
        
        bool touched = false;
        for (size_t j = 0; j < ins.warehouse.size(); ++j) {
            Point wa = ins.warehouse[j];
            Point wb = ins.warehouse[(j + 1) % ins.warehouse.size()];
            if (point_segment_sq_dist(center, wa, wb) <= tol * tol) {
                touched = true; break;
            }
        }
        if (!touched) {
            for (const auto& obs : ins.obstacles) {
                Point o1 = {obs.x, obs.y};
                Point o2 = {obs.x + obs.width, obs.y};
                Point o3 = {obs.x + obs.width, obs.y + obs.depth};
                Point o4 = {obs.x, obs.y + obs.depth};
                if (point_segment_sq_dist(center, o1, o2) <= tol * tol ||
                    point_segment_sq_dist(center, o2, o3) <= tol * tol ||
                    point_segment_sq_dist(center, o3, o4) <= tol * tol ||
                    point_segment_sq_dist(center, o4, o1) <= tol * tol) {
                    touched = true; break;
                }
            }
        }
        if (touched) touches++;
    }
    return touches;
}

inline bool quad_inside_warehouse(const QuadBox& q, const std::vector<Point>& warehouse) {
    Point center{0.0, 0.0};
    for (const auto& p : q.corners) {
        if (!point_in_polygon_inclusive(warehouse, p)) return false;
        center = center + p;
    }
    center = center * 0.25;
    if (!point_in_polygon_inclusive(warehouse, center)) return false;

    for (int i = 0; i < 4; ++i) {
        Point a = q.corners[i];
        Point b = q.corners[(i + 1) % 4];
        for (size_t j = 0; j < warehouse.size(); ++j) {
            Point c = warehouse[j];
            Point d = warehouse[(j + 1) % warehouse.size()];
            if (segment_proper_intersection(a, b, c, d)) return false;
        }
    }
    return true;
}

inline std::pair<Point, Point> polygon_bbox(const std::vector<Point>& poly) {
    Point mn = poly[0], mx = poly[0];
    for (const auto& p : poly) {
        mn.x = std::min(mn.x, p.x); mn.y = std::min(mn.y, p.y);
        mx.x = std::max(mx.x, p.x); mx.y = std::max(mx.y, p.y);
    }
    return {mn, mx};
}

// -----------------------------
// Ceiling and area helpers
// -----------------------------

inline double min_ceiling_for_x_span(const std::vector<CeilingBreak>& ceiling, double xmin, double xmax) {
    if (ceiling.empty()) return std::numeric_limits<double>::infinity();
    if (xmin > xmax) std::swap(xmin, xmax);

    double best = std::numeric_limits<double>::infinity();

    // If the span starts before the first break, assume the first height.
    if (xmin < ceiling.front().x) {
        double lo = xmin;
        double hi = std::min(xmax, ceiling.front().x);
        if (hi - lo > EPS) best = std::min(best, ceiling.front().height);
    }

    for (size_t i = 0; i < ceiling.size(); ++i) {
        double left = ceiling[i].x;
        double right = (i + 1 < ceiling.size()) ? ceiling[i + 1].x : std::numeric_limits<double>::infinity();
        double lo = std::max(xmin, left);
        double hi = std::min(xmax, right);
        if (hi - lo > EPS) best = std::min(best, ceiling[i].height);
    }

    if (!std::isfinite(best)) best = ceiling.front().height;
    return best;
}

inline double usable_area(const Instance& ins) {
    double a = polygon_area_abs(ins.warehouse);
    for (const auto& o : ins.obstacles) a -= std::abs(o.width * o.depth);
    return std::max(1.0, a);
}

// -----------------------------
// Bay placement and gap helpers
// -----------------------------

inline Candidate make_candidate(const Instance& ins,
                                int bayIndex,
                                double x,
                                double y,
                                double rotationDeg) {
    const BayType& b = ins.bays[bayIndex];

    Candidate c;
    c.bayIndex = bayIndex;
    c.bayId = b.id;
    c.x = x;
    c.y = y;
    c.rotation = rotationDeg;
    c.loads = b.loads;
    c.price = b.price;
    c.bayArea = b.width * b.depth;

    c.footprint = make_quad_box(rectangle_corners_from_local_bottom_left(x, y, b.width, b.depth, rotationDeg));

    // Correct gap interpretation:
    // The gap is NOT the whole bay+gap rectangle. It is only the empty walking zone
    // in FRONT of the bay. In local bay coordinates:
    //     bay footprint = [0,width] x [0,depth]
    //     front gap     = [0,width] x [depth, depth+gap]
    // The front direction rotates with the bay, so rotating the bay can move the
    // gap away from a wall, obstacle, or another bay.
    c.gapArea = b.width * std::max(0.0, b.gap);
    c.gapZone = make_quad_box(rectangle_corners_from_local_rect(
        x, y,
        0.0, b.depth,
        b.width, b.depth + std::max(0.0, b.gap),
        rotationDeg
    ));
    c.wallDist = min_dist_to_walls(c.footprint, ins);
    c.wallTouches = count_touching_walls(c.footprint, ins);

    return c;
}

inline bool has_front_gap(const Candidate& c) {
    return c.gapArea > EPS;
}

inline bool candidate_obstacle_conflict(const Candidate& c, const Obstacle& o, bool useGapRules) {
    QuadBox obs = obstacle_box(o);
    // Obstacles cannot overlap the bay footprint.
    if (convex_quads_interiors_overlap(c.footprint, obs)) return true;

    // Obstacles also cannot occupy the FRONT GAP.
    // Gap-gap overlap is allowed, but obstacle-gap overlap is not.
    if (useGapRules && has_front_gap(c) && convex_quads_interiors_overlap(c.gapZone, obs)) return true;
    return false;
}

inline bool candidate_static_feasible(const Instance& ins, const Candidate& c, bool useGapRules) {
    // 1) The physical bay must be inside the warehouse.
    if (!quad_inside_warehouse(c.footprint, ins.warehouse)) return false;

    // 2) If using gap rules, the FRONT GAP ONLY must also remain inside the warehouse.
    // A wall may touch the boundary of the gap, but no positive-area part of the
    // gap may be outside the warehouse.
    if (useGapRules && has_front_gap(c) && !quad_inside_warehouse(c.gapZone, ins.warehouse)) return false;

    // 3) Height must fit under the worst ceiling touched by the bay footprint's x-span.
    double minCeiling = min_ceiling_for_x_span(ins.ceiling, c.footprint.minX, c.footprint.maxX);
    if (ins.bays[c.bayIndex].height > minCeiling + EPS) return false;

    // 4) Bay/gap cannot overlap obstacle interiors. Border touch is allowed.
    for (const auto& o : ins.obstacles) {
        if (candidate_obstacle_conflict(c, o, useGapRules)) return false;
    }

    return true;
}

inline bool candidates_conflict(const Candidate& a, const Candidate& b, bool useGapRules) {
    // Physical bay interiors cannot overlap. Border touch is legal.
    if (convex_quads_interiors_overlap(a.footprint, b.footprint)) return true;

    if (useGapRules) {
        // Correct gap rule:
        //   - A bay footprint cannot occupy another bay's FRONT GAP.
        //   - A bay gap cannot overlap an obstacle or leave the warehouse; that is checked
        //     during candidate generation.
        //   - Two FRONT GAPS may overlap each other, so we deliberately do NOT check
        //     a.gapZone vs b.gapZone.
        if (has_front_gap(a) && convex_quads_interiors_overlap(a.gapZone, b.footprint)) return true;
        if (has_front_gap(b) && convex_quads_interiors_overlap(b.gapZone, a.footprint)) return true;
    }

    return false;
}

inline bool compatible_with_selected(const std::vector<Candidate>& cands,
                                     int candIdx,
                                     const std::vector<int>& selected,
                                     bool useGapRules,
                                     int skipPos = -1) {
    for (int i = 0; i < static_cast<int>(selected.size()); ++i) {
        if (i == skipPos) continue;
        int other = selected[i];
        if (other == candIdx) return false;
        if (candidates_conflict(cands[candIdx], cands[other], useGapRules)) return false;
    }
    return true;
}

// -----------------------------
// Objective function
// -----------------------------

inline Solution evaluate_solution(const std::vector<Candidate>& cands,
                                  const std::vector<int>& selected,
                                  double usableArea) {
    Solution s;
    s.selected = selected;
    s.totalPrice = 0.0;
    s.totalLoads = 0.0;
    s.usedArea = 0.0;

    for (int idx : selected) {
        const Candidate& c = cands[idx];
        s.totalPrice += c.price;
        s.totalLoads += c.loads;
        s.usedArea += c.bayArea;
    }

    if (s.totalLoads <= EPS || s.totalPrice <= EPS) {
        s.score = std::numeric_limits<double>::infinity();
        return s;
    }

    s.areaPct = std::clamp(s.usedArea / std::max(1.0, usableArea), 0.0, 1.0);
    double pricePerLoad = s.totalPrice / s.totalLoads;
    double exponent = 2.0 - s.areaPct;
    s.score = std::pow(pricePerLoad, exponent);
    return s;
}

inline void prune_solution(std::vector<int>& selected,
                           const std::vector<Candidate>& cands,
                           double usableArea) {
    // The objective is nonlinear, so adding more bays can make the score worse.
    // This helper removes bays one by one when removal improves the real objective.
    Solution cur = evaluate_solution(cands, selected, usableArea);
    bool improved = true;

    while (improved && !selected.empty()) {
        improved = false;
        int bestRemovePos = -1;
        double bestScore = cur.score;

        int n_sel = static_cast<int>(selected.size());
        std::vector<double> evals(n_sel, std::numeric_limits<double>::infinity());

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n_sel; ++i) {
            std::vector<int> tmp = selected;
            tmp.erase(tmp.begin() + i);
            Solution trial = evaluate_solution(cands, tmp, usableArea);
            evals[i] = trial.score;
        }

        for (int i = 0; i < n_sel; ++i) {
            if (evals[i] + EPS < bestScore) {
                bestScore = evals[i];
                bestRemovePos = i;
            }
        }

        if (bestRemovePos >= 0) {
            selected.erase(selected.begin() + bestRemovePos);
            cur = evaluate_solution(cands, selected, usableArea);
            improved = true;
        }
    }
}

inline bool validate_solution_geometry(const std::vector<Candidate>& cands,
                                       const Solution& sol,
                                       bool useGapRules) {
    for (int i = 0; i < static_cast<int>(sol.selected.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(sol.selected.size()); ++j) {
            if (candidates_conflict(cands[sol.selected[i]], cands[sol.selected[j]], useGapRules)) {
                return false;
            }
        }
    }
    return true;
}

inline void write_solution_csv(const std::string& path,
                               const std::vector<Candidate>& cands,
                               const Solution& sol,
                               bool header = false) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Could not write " + path);
    if (header) out << "Id,X,Y,Rotation\n";
    out << std::fixed << std::setprecision(3);
    for (int idx : sol.selected) {
        const Candidate& c = cands[idx];
        out << c.bayId << "," << c.x << "," << c.y << "," << c.rotation << "\n";
    }
}

inline void print_solution_summary(const std::string& label, const Solution& s) {
    std::cerr << label
              << " bays=" << s.selected.size()
              << " loads=" << s.totalLoads
              << " price=" << s.totalPrice
              << " areaPct=" << std::fixed << std::setprecision(4) << s.areaPct
              << " score=" << std::setprecision(8) << s.score << "\n";
}

inline void add_value(std::vector<double>& values, double x) {
    if (std::isfinite(x)) values.push_back(x);
}

inline std::vector<double> unique_sorted(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    std::vector<double> out;
    for (double x : values) {
        if (out.empty() || std::abs(x - out.back()) > EPS) out.push_back(x);
    }
    return out;
}

} // namespace whopt
