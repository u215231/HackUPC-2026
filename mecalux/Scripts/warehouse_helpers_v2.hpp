#pragma once

// ================================================================
// warehouse_helpers_v2.hpp
//
// This file contains ONLY helpers:
//   - CSV reading/writing
//   - geometry primitives
//   - bay footprint and gap geometry
//   - warehouse / obstacle / ceiling validation
//   - bay-vs-bay conflict checks
//   - objective score calculation
//
// The optimization algorithms are in optimizer_algorithms_v2.hpp.
// ================================================================

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace whopt2 {

static const double EPS = 1e-8;
static const double PI  = 3.141592653589793238462643383279502884;

inline int num_threads_available() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

struct Point {
    double x;
    double y;
};

inline Point make_point(double x, double y) { Point p{x, y}; return p; }
inline Point addp(Point a, Point b) { return make_point(a.x + b.x, a.y + b.y); }
inline Point subp(Point a, Point b) { return make_point(a.x - b.x, a.y - b.y); }
inline Point mulp(Point a, double k) { return make_point(a.x * k, a.y * k); }
inline double dotp(Point a, Point b) { return a.x * b.x + a.y * b.y; }
inline double crossp(Point a, Point b) { return a.x * b.y - a.y * b.x; }
inline double cross3(Point a, Point b, Point c) { return crossp(subp(b, a), subp(c, a)); }

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
    double x = 0.0;
    double height = 0.0;
};

struct Instance {
    std::vector<Point> warehouse;
    std::vector<Obstacle> obstacles;
    std::vector<CeilingBreak> ceiling;
    std::vector<BayType> bays;
};

struct Quad {
    Point p[4];
    double minX = 0.0;
    double maxX = 0.0;
    double minY = 0.0;
    double maxY = 0.0;
};

struct Candidate {
    int bayIndex = -1;
    int bayId = -1;

    // Contest convention used here:
    // x,y = LOCAL bottom-left corner of the bay.
    // Rotation is applied around this corner, counter-clockwise.
    double x = 0.0;
    double y = 0.0;
    double rotation = 0.0;

    double loads = 0.0;
    double price = 0.0;
    double bayArea = 0.0;

    Quad footprint;  // physical bay: width x depth
    Quad clearance;  // physical bay plus front gap: width x (depth + gap)
};

struct Solution {
    std::vector<int> selected; // indices into candidates
    double totalPrice = 0.0;
    double totalLoads = 0.0;
    double usedArea = 0.0;
    double areaPct = 0.0; // fraction in [0,1]
    double score = std::numeric_limits<double>::infinity();
};

// -----------------------------
// CSV helpers
// -----------------------------

inline std::string trim(const std::string& s) {
    std::size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    std::size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

inline std::vector<double> parse_numbers_from_csv_line(const std::string& line) {
    std::vector<double> values;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (token.empty()) continue;
        try {
            std::size_t used = 0;
            double v = std::stod(token, &used);
            if (used > 0) values.push_back(v);
        } catch (...) {
            // Ignore headers or text cells.
        }
    }
    return values;
}

inline std::string path_join(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    char c = dir[dir.size() - 1];
    if (c == '/' || c == '\\') return dir + file;
    return dir + "/" + file;
}

inline std::vector<Point> read_warehouse_csv(const std::string& path) {
    std::ifstream in(path.c_str());
    if (!in) throw std::runtime_error("Could not open " + path);
    std::vector<Point> poly;
    std::string line;
    while (std::getline(in, line)) {
        std::vector<double> v = parse_numbers_from_csv_line(line);
        if (v.size() >= 2) poly.push_back(make_point(v[0], v[1]));
    }
    if (poly.size() < 3) throw std::runtime_error("warehouse.csv must contain at least 3 points");
    return poly;
}

inline std::vector<Obstacle> read_obstacles_csv(const std::string& path) {
    std::ifstream in(path.c_str());
    if (!in) throw std::runtime_error("Could not open " + path);
    std::vector<Obstacle> out;
    std::string line;
    while (std::getline(in, line)) {
        std::vector<double> v = parse_numbers_from_csv_line(line);
        if (v.size() >= 4) {
            Obstacle o;
            o.x = v[0]; o.y = v[1]; o.width = v[2]; o.depth = v[3];
            out.push_back(o);
        }
    }
    return out;
}

inline std::vector<CeilingBreak> read_ceiling_csv(const std::string& path) {
    std::ifstream in(path.c_str());
    if (!in) throw std::runtime_error("Could not open " + path);
    std::vector<CeilingBreak> out;
    std::string line;
    while (std::getline(in, line)) {
        std::vector<double> v = parse_numbers_from_csv_line(line);
        if (v.size() >= 2) {
            CeilingBreak c;
            c.x = v[0]; c.height = v[1];
            out.push_back(c);
        }
    }
    std::sort(out.begin(), out.end(), [](const CeilingBreak& a, const CeilingBreak& b) { return a.x < b.x; });
    if (out.empty()) throw std::runtime_error("ceiling.csv has no numeric rows");
    return out;
}

inline std::vector<BayType> read_types_of_bays_csv(const std::string& path) {
    std::ifstream in(path.c_str());
    if (!in) throw std::runtime_error("Could not open " + path);
    std::vector<BayType> out;
    std::string line;
    while (std::getline(in, line)) {
        std::vector<double> v = parse_numbers_from_csv_line(line);
        if (v.size() >= 7) {
            BayType b;
            b.id = static_cast<int>(std::llround(v[0]));
            b.width = v[1]; b.depth = v[2]; b.height = v[3]; b.gap = v[4];
            b.loads = v[5]; b.price = v[6];
            out.push_back(b);
        }
    }
    if (out.empty()) throw std::runtime_error("types_of_bays.csv has no numeric rows");
    return out;
}

inline Instance read_instance(const std::string& caseDir) {
    Instance ins;
    ins.warehouse = read_warehouse_csv(path_join(caseDir, "warehouse.csv"));
    ins.obstacles = read_obstacles_csv(path_join(caseDir, "obstacles.csv"));
    ins.ceiling = read_ceiling_csv(path_join(caseDir, "ceiling.csv"));
    ins.bays = read_types_of_bays_csv(path_join(caseDir, "types_of_bays.csv"));
    return ins;
}

// -----------------------------
// Geometry helpers
// -----------------------------

inline double polygon_signed_area(const std::vector<Point>& poly) {
    double a = 0.0;
    for (std::size_t i = 0; i < poly.size(); ++i) {
        const Point& p = poly[i];
        const Point& q = poly[(i + 1) % poly.size()];
        a += p.x * q.y - q.x * p.y;
    }
    return 0.5 * a;
}

inline double polygon_area(const std::vector<Point>& poly) {
    return std::fabs(polygon_signed_area(poly));
}

inline std::pair<Point, Point> polygon_bbox(const std::vector<Point>& poly) {
    Point mn = poly[0];
    Point mx = poly[0];
    for (std::size_t i = 1; i < poly.size(); ++i) {
        mn.x = std::min(mn.x, poly[i].x); mn.y = std::min(mn.y, poly[i].y);
        mx.x = std::max(mx.x, poly[i].x); mx.y = std::max(mx.y, poly[i].y);
    }
    return std::make_pair(mn, mx);
}

inline bool on_segment(Point a, Point b, Point p) {
    if (std::fabs(cross3(a, b, p)) > EPS) return false;
    if (p.x < std::min(a.x, b.x) - EPS || p.x > std::max(a.x, b.x) + EPS) return false;
    if (p.y < std::min(a.y, b.y) - EPS || p.y > std::max(a.y, b.y) + EPS) return false;
    return true;
}

inline bool point_in_polygon_inclusive(const std::vector<Point>& poly, Point p) {
    bool inside = false;
    for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
        Point a = poly[j];
        Point b = poly[i];
        if (on_segment(a, b, p)) return true;
        bool crosses = ((a.y > p.y) != (b.y > p.y));
        if (crosses) {
            double xAtY = (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x;
            if (p.x < xAtY) inside = !inside;
        }
    }
    return inside;
}

inline bool proper_segment_intersection(Point a, Point b, Point c, Point d) {
    double o1 = cross3(a, b, c);
    double o2 = cross3(a, b, d);
    double o3 = cross3(c, d, a);
    double o4 = cross3(c, d, b);
    bool s1 = (o1 > EPS && o2 < -EPS) || (o1 < -EPS && o2 > EPS);
    bool s2 = (o3 > EPS && o4 < -EPS) || (o3 < -EPS && o4 > EPS);
    return s1 && s2;
}

inline Quad make_quad_from_points(Point p0, Point p1, Point p2, Point p3) {
    Quad q;
    q.p[0] = p0; q.p[1] = p1; q.p[2] = p2; q.p[3] = p3;
    q.minX = q.maxX = p0.x;
    q.minY = q.maxY = p0.y;
    for (int i = 1; i < 4; ++i) {
        q.minX = std::min(q.minX, q.p[i].x); q.maxX = std::max(q.maxX, q.p[i].x);
        q.minY = std::min(q.minY, q.p[i].y); q.maxY = std::max(q.maxY, q.p[i].y);
    }
    return q;
}

inline Quad make_rectangle_from_local_bottom_left(double x, double y, double width, double depth, double rotationDeg) {
    double r = rotationDeg * PI / 180.0;
    Point ux = make_point(std::cos(r), std::sin(r));
    Point uy = make_point(-std::sin(r), std::cos(r));
    Point p0 = make_point(x, y);
    Point p1 = addp(p0, mulp(ux, width));
    Point p3 = addp(p0, mulp(uy, depth));
    Point p2 = addp(p1, mulp(uy, depth));
    return make_quad_from_points(p0, p1, p2, p3);
}

inline Quad obstacle_to_quad(const Obstacle& o) {
    return make_quad_from_points(
        make_point(o.x, o.y),
        make_point(o.x + o.width, o.y),
        make_point(o.x + o.width, o.y + o.depth),
        make_point(o.x, o.y + o.depth)
    );
}

inline bool bbox_positive_overlap(const Quad& a, const Quad& b) {
    double ox = std::min(a.maxX, b.maxX) - std::max(a.minX, b.minX);
    double oy = std::min(a.maxY, b.maxY) - std::max(a.minY, b.minY);
    return ox > EPS && oy > EPS;
}

inline bool separated_on_axis(const Quad& a, const Quad& b, Point axis) {
    double amin = dotp(a.p[0], axis), amax = amin;
    double bmin = dotp(b.p[0], axis), bmax = bmin;
    for (int i = 1; i < 4; ++i) {
        double av = dotp(a.p[i], axis);
        double bv = dotp(b.p[i], axis);
        amin = std::min(amin, av); amax = std::max(amax, av);
        bmin = std::min(bmin, bv); bmax = std::max(bmax, bv);
    }
    double overlap = std::min(amax, bmax) - std::max(amin, bmin);
    // Border touching or exact border overlap is allowed.
    return overlap <= EPS;
}

inline bool quad_interiors_overlap(const Quad& a, const Quad& b) {
    if (!bbox_positive_overlap(a, b)) return false;
    for (int i = 0; i < 4; ++i) {
        Point e = subp(a.p[(i + 1) % 4], a.p[i]);
        Point axis = make_point(-e.y, e.x);
        if (separated_on_axis(a, b, axis)) return false;
    }
    for (int i = 0; i < 4; ++i) {
        Point e = subp(b.p[(i + 1) % 4], b.p[i]);
        Point axis = make_point(-e.y, e.x);
        if (separated_on_axis(a, b, axis)) return false;
    }
    return true;
}

inline bool quad_inside_polygon_inclusive(const Quad& q, const std::vector<Point>& poly) {
    Point center = make_point(0.0, 0.0);
    for (int i = 0; i < 4; ++i) {
        if (!point_in_polygon_inclusive(poly, q.p[i])) return false;
        center = addp(center, q.p[i]);
    }
    center = mulp(center, 0.25);
    if (!point_in_polygon_inclusive(poly, center)) return false;

    for (int i = 0; i < 4; ++i) {
        Point a = q.p[i];
        Point b = q.p[(i + 1) % 4];
        for (std::size_t j = 0; j < poly.size(); ++j) {
            Point c = poly[j];
            Point d = poly[(j + 1) % poly.size()];
            if (proper_segment_intersection(a, b, c, d)) return false;
        }
    }
    return true;
}

inline double usable_area(const Instance& ins) {
    double a = polygon_area(ins.warehouse);
    for (std::size_t i = 0; i < ins.obstacles.size(); ++i) {
        a -= std::fabs(ins.obstacles[i].width * ins.obstacles[i].depth);
    }
    return std::max(1.0, a);
}

inline double min_ceiling_on_x_span(const Instance& ins, double xmin, double xmax) {
    if (xmin > xmax) std::swap(xmin, xmax);
    if (ins.ceiling.empty()) return std::numeric_limits<double>::infinity();

    double ans = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < ins.ceiling.size(); ++i) {
        double left = ins.ceiling[i].x;
        double right = (i + 1 < ins.ceiling.size()) ? ins.ceiling[i + 1].x : std::numeric_limits<double>::infinity();
        double lo = std::max(xmin, left);
        double hi = std::min(xmax, right);
        if (hi - lo > EPS) ans = std::min(ans, ins.ceiling[i].height);
    }

    if (!std::isfinite(ans)) {
        if (xmax <= ins.ceiling.front().x + EPS) return ins.ceiling.front().height;
        return ins.ceiling.back().height;
    }
    return ans;
}

// -----------------------------
// Bay placement and validity helpers
// -----------------------------

inline Candidate make_candidate(const Instance& ins, int bayIndex, double x, double y, double rotationDeg) {
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
    c.footprint = make_rectangle_from_local_bottom_left(x, y, b.width, b.depth, rotationDeg);
    c.clearance = make_rectangle_from_local_bottom_left(x, y, b.width, b.depth + b.gap, rotationDeg);
    return c;
}

inline bool candidate_hits_obstacle(const Candidate& c, const Obstacle& o, bool useGapRules) {
    Quad obs = obstacle_to_quad(o);
    if (quad_interiors_overlap(c.footprint, obs)) return true;
    if (useGapRules && quad_interiors_overlap(c.clearance, obs)) return true;
    return false;
}

inline bool candidate_static_feasible(const Instance& ins, const Candidate& c, bool useGapRules) {
    if (!quad_inside_polygon_inclusive(c.footprint, ins.warehouse)) return false;
    if (useGapRules && !quad_inside_polygon_inclusive(c.clearance, ins.warehouse)) return false;

    double minCeil = min_ceiling_on_x_span(ins, c.footprint.minX, c.footprint.maxX);
    if (ins.bays[c.bayIndex].height > minCeil + EPS) return false;

    for (std::size_t i = 0; i < ins.obstacles.size(); ++i) {
        if (candidate_hits_obstacle(c, ins.obstacles[i], useGapRules)) return false;
    }
    return true;
}

inline bool candidates_conflict(const Candidate& a, const Candidate& b, bool useGapRules) {
    // Physical bay interiors may not overlap. Borders may touch.
    if (quad_interiors_overlap(a.footprint, b.footprint)) return true;

    if (useGapRules) {
        // Gap rule implemented here:
        // A physical bay may not occupy another bay's front clearance zone.
        // Two empty clearance zones are allowed to overlap.
        if (quad_interiors_overlap(a.clearance, b.footprint)) return true;
        if (quad_interiors_overlap(b.clearance, a.footprint)) return true;
    }
    return false;
}

inline bool compatible_with_selected(const std::vector<Candidate>& candidates,
                                     int candidateIndex,
                                     const std::vector<int>& selected,
                                     bool useGapRules) {
    for (std::size_t i = 0; i < selected.size(); ++i) {
        int j = selected[i];
        if (j == candidateIndex) return false;
        if (candidates_conflict(candidates[candidateIndex], candidates[j], useGapRules)) return false;
    }
    return true;
}

inline Solution evaluate_solution(const std::vector<Candidate>& candidates,
                                  const std::vector<int>& selected,
                                  double usableArea) {
    Solution s;
    s.selected = selected;
    for (std::size_t i = 0; i < selected.size(); ++i) {
        const Candidate& c = candidates[selected[i]];
        s.totalPrice += c.price;
        s.totalLoads += c.loads;
        s.usedArea += c.bayArea;
    }
    if (s.totalLoads <= EPS || s.totalPrice <= EPS) {
        s.score = std::numeric_limits<double>::infinity();
        return s;
    }
    s.areaPct = s.usedArea / std::max(1.0, usableArea);
    if (s.areaPct < 0.0) s.areaPct = 0.0;
    if (s.areaPct > 1.0) s.areaPct = 1.0;
    double pricePerLoad = s.totalPrice / s.totalLoads;
    double exponent = 2.0 - s.areaPct;
    s.score = std::pow(pricePerLoad, exponent);
    return s;
}

inline void prune_solution_by_true_objective(std::vector<int>& selected,
                                             const std::vector<Candidate>& candidates,
                                             double usableArea) {
    // Removes bays only when removal improves the real contest objective.
    bool changed = true;
    Solution current = evaluate_solution(candidates, selected, usableArea);
    while (changed && !selected.empty()) {
        changed = false;
        int bestPos = -1;
        double bestScore = current.score;
        for (int i = 0; i < static_cast<int>(selected.size()); ++i) {
            std::vector<int> tmp = selected;
            tmp.erase(tmp.begin() + i);
            Solution trial = evaluate_solution(candidates, tmp, usableArea);
            if (trial.score + EPS < bestScore) {
                bestScore = trial.score;
                bestPos = i;
            }
        }
        if (bestPos >= 0) {
            selected.erase(selected.begin() + bestPos);
            current = evaluate_solution(candidates, selected, usableArea);
            changed = true;
        }
    }
}

inline bool validate_solution_geometry(const std::vector<Candidate>& candidates,
                                       const Solution& sol,
                                       bool useGapRules) {
    for (std::size_t i = 0; i < sol.selected.size(); ++i) {
        for (std::size_t j = i + 1; j < sol.selected.size(); ++j) {
            if (candidates_conflict(candidates[sol.selected[i]], candidates[sol.selected[j]], useGapRules)) return false;
        }
    }
    return true;
}

inline void write_solution_csv(const std::string& path,
                               const std::vector<Candidate>& candidates,
                               const Solution& sol,
                               bool header) {
    std::ofstream out(path.c_str());
    if (!out) throw std::runtime_error("Could not write " + path);
    if (header) out << "Id,X,Y,Rotation\n";
    out << std::fixed << std::setprecision(3);
    for (std::size_t i = 0; i < sol.selected.size(); ++i) {
        const Candidate& c = candidates[sol.selected[i]];
        out << c.bayId << "," << c.x << "," << c.y << "," << c.rotation << "\n";
    }
}

inline void print_solution_summary(const std::string& name, const Solution& s) {
    std::cerr << name
              << " bays=" << s.selected.size()
              << " loads=" << s.totalLoads
              << " price=" << s.totalPrice
              << " areaPct=" << std::fixed << std::setprecision(4) << s.areaPct
              << " score=" << std::setprecision(8) << s.score << "\n";
}

} // namespace whopt2
