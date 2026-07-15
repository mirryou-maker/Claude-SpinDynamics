#include <cstdio>
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

    // Use C stdio (FILE*) rather than std::ofstream: on the CUDA-linked build
    // the C++ filebuf flush/close deadlocks (a runtime interaction specific to
    // the GPU module), while plain stdio writes and closes cleanly. Number
    // formatting still uses std::ostringstream (in-memory, unaffected).
    FILE* f = std::fopen(filename.c_str(), "wb");
    if (!f) throw std::runtime_error("save_ovf: cannot open '" + filename + "'");

    // Helper: format a double in scientific notation for the OVF header
    auto fmtd = [](double v) -> std::string {
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(15) << v;
        return ss.str();
    };

    // Write header as text (each line + newline)
    auto hdr = [&](const std::string& s){ std::fputs(s.c_str(), f); std::fputc('\n', f); };
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
        // Data order: z outer, y middle, x inner (x-fastest)
        for (Index iz = 0; iz < nz; ++iz)
        for (Index iy = 0; iy < ny; ++iy)
        for (Index ix = 0; ix < nx; ++ix) {
            const Vec3& v = m[g.linear_index(ix, iy, iz)];
            std::fprintf(f, "%.15e %.15e %.15e\n",
                         (double)v.x, (double)v.y, (double)v.z);
        }
        hdr("# End: Data Text");
    } else if (fmt == OVFFormat::Binary4) {
        // Binary 4 (IEEE 754 float, little-endian) — half the size of Binary8
        hdr("# Begin: Data Binary 4");
        // Check value: 1234567.0 as float (mumax3 convention for Binary 4)
        const float check4 = 1234567.0f;
        std::fwrite(&check4, 4, 1, f);
        for (Index iz = 0; iz < nz; ++iz)
        for (Index iy = 0; iy < ny; ++iy)
        for (Index ix = 0; ix < nx; ++ix) {
            const Vec3& v = m[g.linear_index(ix, iy, iz)];
            float buf[3] = { static_cast<float>(v.x),
                             static_cast<float>(v.y),
                             static_cast<float>(v.z) };
            std::fwrite(buf, 12, 1, f);
        }
        hdr("# End: Data Binary 4");
    } else {
        // Binary 8 (IEEE 754 double, little-endian)
        hdr("# Begin: Data Binary 8");
        // Write check value first (123456789012345.0 as double)
        const double check = 123456789012345.0;
        std::fwrite(&check, 8, 1, f);
        // Write data
        for (Index iz = 0; iz < nz; ++iz)
        for (Index iy = 0; iy < ny; ++iy)
        for (Index ix = 0; ix < nx; ++ix) {
            const Vec3& v = m[g.linear_index(ix, iy, iz)];
            double buf[3] = { v.x, v.y, v.z };
            std::fwrite(buf, 24, 1, f);
        }
        hdr("# End: Data Binary 8");
    }

    hdr("# End: Segment");
    std::fclose(f);
}

// ---------------------------------------------------------------------------
// load_ovf
// ---------------------------------------------------------------------------

// Read an entire file into a std::string via C stdio. std::ifstream's filebuf
// deadlocks in the CUDA-linked build (see save_ovf), but stdio and in-memory
// std::istringstream both work — so slurp the file, then parse from memory.
static std::string slurp_file(const std::string& filename, const char* who) {
    FILE* fp = std::fopen(filename.c_str(), "rb");
    if (!fp) throw std::runtime_error(std::string(who) + ": cannot open '" + filename + "'");
    std::fseek(fp, 0, SEEK_END);
    long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::string data;
    if (sz > 0) {
        data.resize(static_cast<size_t>(sz));
        size_t got = std::fread(&data[0], 1, static_cast<size_t>(sz), fp);
        data.resize(got);
    }
    std::fclose(fp);
    return data;
}

VectorField3D load_ovf(const std::string& filename) {
    std::string _contents = slurp_file(filename, "load_ovf");
    std::istringstream f(_contents, std::ios::binary);

    // Parse header
    Index  nx = 0, ny = 0, nz = 0;
    double dx = 0, dy = 0, dz = 0;
    bool   binary8 = false;
    bool   binary4 = false;
    bool   in_data = false;

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
            continue;   // OVF 1.0 and 2.0 headers are parsed identically below
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

StructuredGrid load_ovf_grid(const std::string& filename) {
    std::string _contents = slurp_file(filename, "load_ovf_grid");
    std::istringstream f(_contents, std::ios::binary);
    Index  nx = 0, ny = 0, nz = 0;
    double dx = 0, dy = 0, dz = 0;
    std::string line, val;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] != '#') continue;
        if (parse_header_line(line, "xnodes", val)) { nx = std::stoi(val); continue; }
        if (parse_header_line(line, "ynodes", val)) { ny = std::stoi(val); continue; }
        if (parse_header_line(line, "znodes", val)) { nz = std::stoi(val); continue; }
        if (parse_header_line(line, "xstepsize", val)) { dx = std::stod(val); continue; }
        if (parse_header_line(line, "ystepsize", val)) { dy = std::stod(val); continue; }
        if (parse_header_line(line, "zstepsize", val)) { dz = std::stod(val); continue; }
        if (nx && ny && nz && dx && dy && dz) break;
    }
    if (nx == 0 || ny == 0 || nz == 0 || dx == 0 || dy == 0 || dz == 0)
        throw std::runtime_error(
            "load_ovf_grid: '" + filename + "': missing grid dimensions");
    return StructuredGrid(nx, ny, nz,
                          static_cast<Real>(dx),
                          static_cast<Real>(dy),
                          static_cast<Real>(dz));
}

void load_ovf_into(const std::string& filename, VectorField3D& m) {
    VectorField3D tmp = load_ovf(filename);
    const auto& tg = tmp.grid();
    const auto& mg = m.grid();
    if (tg.nx() != mg.nx() || tg.ny() != mg.ny() || tg.nz() != mg.nz())
        throw std::runtime_error(
            "load_ovf_into: grid mismatch");
    for (Index i = 0; i < m.size(); ++i)
        m[i] = tmp[i];
}

}  // namespace micromag
