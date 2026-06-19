#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include "micromag/ovf_io.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

// Case-insensitive compare
bool iequal(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i]))
            return false;
    return true;
}

// Parse "# key: value" lines; returns true and sets value if found
bool parse_header_line(const std::string& line, const std::string& key,
                        std::string& value) {
    // Strip leading '#' and whitespace
    size_t pos = 0;
    while (pos < line.size() && (line[pos] == '#' || line[pos] == ' ' || line[pos] == '\t'))
        ++pos;
    const std::string rest = line.substr(pos);
    // Find ':'
    size_t colon = rest.find(':');
    if (colon == std::string::npos) return false;
    std::string k = trim(rest.substr(0, colon));
    if (!iequal(k, key)) return false;
    value = trim(rest.substr(colon + 1));
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// save_ovf
// ---------------------------------------------------------------------------

void save_ovf(const std::string& filename,
              const VectorField3D& m,
              const std::string& title,
              OVFFormat fmt) {
    const auto& g = m.grid();
    const Index nx = g.nx(), ny = g.ny(), nz = g.nz();
    const Real  dx = g.dx(), dy = g.dy(), dz = g.dz();

    std::ofstream f(filename, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("save_ovf: cannot open '" + filename + "'");

    // Helper: format a double in scientific notation for the OVF header
    auto fmtd = [](double v) -> std::string {
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(15) << v;
        return ss.str();
    };

    // Write header as text
    auto hdr = [&](const std::string& s){ f << s << "\n"; };
    hdr("# OOMMF OVF 2.0");
    hdr("");
    hdr("# Segment count: 1");
    hdr("");
    hdr("# Begin: Segment");
    hdr("# Begin: Header");
    hdr("# Title: " + title);
    hdr("# meshtype: rectangular");
    hdr("# meshunit: m");
    hdr("# xmin: 0");
    hdr("# ymin: 0");
    hdr("# zmin: 0");
    hdr("# xmax: " + fmtd(nx * dx));
    hdr("# ymax: " + fmtd(ny * dy));
    hdr("# zmax: " + fmtd(nz * dz));
    hdr("# valuedim: 3");
    hdr("# valuelabels: mx my mz");
    hdr("# valueunits: 1 1 1");
    hdr("# xbase: " + fmtd(dx * 0.5));
    hdr("# ybase: " + fmtd(dy * 0.5));
    hdr("# zbase: " + fmtd(dz * 0.5));
    hdr("# xstepsize: " + fmtd(dx));
    hdr("# ystepsize: " + fmtd(dy));
    hdr("# zstepsize: " + fmtd(dz));
    hdr("# xnodes: " + std::to_string(nx));
    hdr("# ynodes: " + std::to_string(ny));
    hdr("# znodes: " + std::to_string(nz));
    hdr("# End: Header");

    if (fmt == OVFFormat::Text) {
        hdr("# Begin: Data Text");
        f << std::scientific;
        f.precision(15);
        // Data order: z outer, y middle, x inner (x-fastest)
        for (Index iz = 0; iz < nz; ++iz)
        for (Index iy = 0; iy < ny; ++iy)
        for (Index ix = 0; ix < nx; ++ix) {
            const Vec3& v = m[g.linear_index(ix, iy, iz)];
            f << v.x << " " << v.y << " " << v.z << "\n";
        }
        hdr("# End: Data Text");
    } else if (fmt == OVFFormat::Binary4) {
        // Binary 4 (IEEE 754 float, little-endian) — half the size of Binary8
        hdr("# Begin: Data Binary 4");
        // Check value: 1234567.0 as float (mumax3 convention for Binary 4)
        const float check4 = 1234567.0f;
        f.write(reinterpret_cast<const char*>(&check4), 4);
        for (Index iz = 0; iz < nz; ++iz)
        for (Index iy = 0; iy < ny; ++iy)
        for (Index ix = 0; ix < nx; ++ix) {
            const Vec3& v = m[g.linear_index(ix, iy, iz)];
            float buf[3] = { static_cast<float>(v.x),
                             static_cast<float>(v.y),
                             static_cast<float>(v.z) };
            f.write(reinterpret_cast<const char*>(buf), 12);
        }
        hdr("# End: Data Binary 4");
    } else {
        // Binary 8 (IEEE 754 double, little-endian)
        hdr("# Begin: Data Binary 8");
        // Write check value first (123456789012345.0 as double)
        const double check = 123456789012345.0;
        f.write(reinterpret_cast<const char*>(&check), 8);
        // Write data
        for (Index iz = 0; iz < nz; ++iz)
        for (Index iy = 0; iy < ny; ++iy)
        for (Index ix = 0; ix < nx; ++ix) {
            const Vec3& v = m[g.linear_index(ix, iy, iz)];
            double buf[3] = { v.x, v.y, v.z };
            f.write(reinterpret_cast<const char*>(buf), 24);
        }
        hdr("# End: Data Binary 8");
    }

    hdr("# End: Segment");
}

// ---------------------------------------------------------------------------
// load_ovf
// ---------------------------------------------------------------------------

VectorField3D load_ovf(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("load_ovf: cannot open '" + filename + "'");

    // Parse header
    Index  nx = 0, ny = 0, nz = 0;
    double dx = 0, dy = 0, dz = 0;
    bool   binary8 = false;
    bool   binary4 = false;
    bool   in_data = false;
    bool   header_done = false;
    int    ovf_version = 2;

    std::string line;
    std::string val;

    while (std::getline(f, line)) {
        // Strip trailing \r (Windows line endings)
        if (!line.empty() && line.back() == '\r') line.pop_back();

        if (line.empty() || line[0] != '#') {
            // Non-header data line (text format)
            if (in_data && !binary8 && !binary4 && nx > 0) {
                // Text data starts here — seek back and break
                // We'll re-read from current position after creating the field
                break;
            }
            continue;
        }

        // Parse known header keys
        if (parse_header_line(line, "OOMMF", val)) {
            if (val.find("1.0") != std::string::npos) ovf_version = 1;
            else ovf_version = 2;
            continue;
        }
        if (parse_header_line(line, "xnodes", val)) { nx = std::stoi(val); continue; }
        if (parse_header_line(line, "ynodes", val)) { ny = std::stoi(val); continue; }
        if (parse_header_line(line, "znodes", val)) { nz = std::stoi(val); continue; }
        if (parse_header_line(line, "xstepsize", val)) { dx = std::stod(val); continue; }
        if (parse_header_line(line, "ystepsize", val)) { dy = std::stod(val); continue; }
        if (parse_header_line(line, "zstepsize", val)) { dz = std::stod(val); continue; }

        // Data section markers
        // OVF1 uses "# Begin: data ..." (lower-case data)
        std::string tl = line;
        std::transform(tl.begin(), tl.end(), tl.begin(),
                        [](unsigned char c){ return std::tolower(c); });

        if (tl.find("begin: data binary 8") != std::string::npos ||
            tl.find("begin: data binary8") != std::string::npos) {
            binary8 = true;
            in_data = true;
            break;
        }
        if (tl.find("begin: data binary 4") != std::string::npos ||
            tl.find("begin: data binary4") != std::string::npos) {
            binary4 = true;
            in_data = true;
            break;
        }
        if (tl.find("begin: data text") != std::string::npos) {
            in_data = true;
            // Text data follows on next lines — break to read below
            break;
        }
    }

    if (nx == 0 || ny == 0 || nz == 0 || dx == 0 || dy == 0 || dz == 0)
        throw std::runtime_error(
            "load_ovf: '" + filename + "': missing or zero grid dimensions in header");

    StructuredGrid g(nx, ny, nz,
                     static_cast<Real>(dx),
                     static_cast<Real>(dy),
                     static_cast<Real>(dz));
    VectorField3D m(g);

    if (binary8) {
        // Read 8-byte check value
        double check = 0;
        f.read(reinterpret_cast<char*>(&check), 8);
        // Read data
        for (Index iz = 0; iz < nz; ++iz)
        for (Index iy = 0; iy < ny; ++iy)
        for (Index ix = 0; ix < nx; ++ix) {
            double buf[3] = {0, 0, 0};
            f.read(reinterpret_cast<char*>(buf), 24);
            m[g.linear_index(ix, iy, iz)] = {
                static_cast<Real>(buf[0]),
                static_cast<Real>(buf[1]),
                static_cast<Real>(buf[2])
            };
        }
    } else if (binary4) {
        // 4-byte float check value
        float check4 = 0;
        f.read(reinterpret_cast<char*>(&check4), 4);
        for (Index iz = 0; iz < nz; ++iz)
        for (Index iy = 0; iy < ny; ++iy)
        for (Index ix = 0; ix < nx; ++ix) {
            float buf[3] = {0, 0, 0};
            f.read(reinterpret_cast<char*>(buf), 12);
            m[g.linear_index(ix, iy, iz)] = {
                static_cast<Real>(buf[0]),
                static_cast<Real>(buf[1]),
                static_cast<Real>(buf[2])
            };
        }
    } else {
        // Text: each remaining non-comment line is "mx my mz"
        for (Index iz = 0; iz < nz; ++iz)
        for (Index iy = 0; iy < ny; ++iy)
        for (Index ix = 0; ix < nx; ++ix) {
            // Skip comment/empty lines
            while (std::getline(f, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty() || line[0] == '#') continue;
                break;
            }
            double x, y, z;
            std::istringstream ss(line);
            ss >> x >> y >> z;
            m[g.linear_index(ix, iy, iz)] = {
                static_cast<Real>(x),
                static_cast<Real>(y),
                static_cast<Real>(z)
            };
        }
    }

    return m;
}

}  // namespace micromag
