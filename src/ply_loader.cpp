#include "ply_loader.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#ifndef _WIN32
#define _ftelli64 ftello
#define _fseeki64 fseeko
#endif

namespace {

enum class PType { I8, U8, I16, U16, I32, U32, F32, F64, Invalid };

PType parse_type(const std::string& s) {
    if (s == "char" || s == "int8") return PType::I8;
    if (s == "uchar" || s == "uint8") return PType::U8;
    if (s == "short" || s == "int16") return PType::I16;
    if (s == "ushort" || s == "uint16") return PType::U16;
    if (s == "int" || s == "int32") return PType::I32;
    if (s == "uint" || s == "uint32") return PType::U32;
    if (s == "float" || s == "float32") return PType::F32;
    if (s == "double" || s == "float64") return PType::F64;
    return PType::Invalid;
}

int type_size(PType t) {
    switch (t) {
        case PType::I8: case PType::U8: return 1;
        case PType::I16: case PType::U16: return 2;
        case PType::I32: case PType::U32: case PType::F32: return 4;
        case PType::F64: return 8;
        default: return 0;
    }
}

struct Property {
    std::string name;
    PType type = PType::Invalid;
    bool isList = false;
    PType countType = PType::Invalid;
};

struct Element {
    std::string name;
    size_t count = 0;
    std::vector<Property> props;
};

enum class Format { Ascii, BinaryLE, BinaryBE };

// Reads scalars from the body of the file in either ASCII or binary form.
struct Reader {
    const char* p;
    const char* end;
    Format fmt;
    bool ok = true;

    template <class T>
    T raw() {
        T v;
        if (end - p < (ptrdiff_t)sizeof(T)) { ok = false; return T(0); }
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        if (fmt == Format::BinaryBE && sizeof(T) > 1) {
            char* b = reinterpret_cast<char*>(&v);
            for (size_t i = 0; i < sizeof(T) / 2; ++i) std::swap(b[i], b[sizeof(T) - 1 - i]);
        }
        return v;
    }

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
    }

    double ascii_value() {
        skip_ws();
        if (p < end && *p == '+') ++p;
        double v = 0.0;
        auto res = std::from_chars(p, end, v);
        if (res.ec != std::errc()) { ok = false; return 0.0; }
        p = res.ptr;
        return v;
    }

    double value(PType t) {
        if (fmt == Format::Ascii) return ascii_value();
        switch (t) {
            case PType::I8: return raw<int8_t>();
            case PType::U8: return raw<uint8_t>();
            case PType::I16: return raw<int16_t>();
            case PType::U16: return raw<uint16_t>();
            case PType::I32: return raw<int32_t>();
            case PType::U32: return raw<uint32_t>();
            case PType::F32: return raw<float>();
            case PType::F64: return raw<double>();
            default: ok = false; return 0.0;
        }
    }
};

bool read_file(const std::string& path, std::vector<char>& data) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    _fseeki64(f, 0, SEEK_END);
    long long size = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    data.resize(size_t(size));
    size_t got = std::fread(data.data(), 1, data.size(), f);
    std::fclose(f);
    return got == data.size();
}

}  // namespace

bool load_ply(const std::string& path, Mesh& mesh, std::string& err) {
    std::vector<char> data;
    if (!read_file(path, data)) { err = "cannot read " + path; return false; }

    // ---- header ----
    const char* marker = "end_header";
    auto it = std::search(data.begin(), data.end(), marker, marker + std::strlen(marker));
    if (data.size() < 3 || std::strncmp(data.data(), "ply", 3) != 0 || it == data.end()) {
        err = "not a PLY file: " + path;
        return false;
    }
    const char* body = &*it + std::strlen(marker);
    while (body < data.data() + data.size() && *body != '\n') ++body;
    ++body;  // past the newline that ends the header

    std::istringstream header(std::string(data.data(), &*it));
    std::vector<Element> elements;
    Format fmt = Format::Ascii;
    std::string line;
    while (std::getline(header, line)) {
        std::istringstream ls(line);
        std::string kw;
        ls >> kw;
        if (kw == "format") {
            std::string f;
            ls >> f;
            if (f == "ascii") fmt = Format::Ascii;
            else if (f == "binary_little_endian") fmt = Format::BinaryLE;
            else if (f == "binary_big_endian") fmt = Format::BinaryBE;
            else { err = "unknown PLY format " + f; return false; }
        } else if (kw == "element") {
            Element e;
            ls >> e.name >> e.count;
            elements.push_back(e);
        } else if (kw == "property") {
            if (elements.empty()) { err = "property before element"; return false; }
            Property p;
            std::string t;
            ls >> t;
            if (t == "list") {
                std::string ct, it2;
                ls >> ct >> it2 >> p.name;
                p.isList = true;
                p.countType = parse_type(ct);
                p.type = parse_type(it2);
                if (p.countType == PType::Invalid) { err = "bad list count type " + ct; return false; }
            } else {
                p.type = parse_type(t);
                ls >> p.name;
            }
            if (p.type == PType::Invalid) { err = "bad property type in: " + line; return false; }
            elements.back().props.push_back(p);
        }
    }

    // ---- body ----
    mesh.vertices.clear();
    mesh.faces.clear();
    Reader rd{body, data.data() + data.size(), fmt};
    std::vector<uint32_t> poly;
    for (const Element& e : elements) {
        if (e.name == "vertex") {
            int ix = -1, iy = -1, iz = -1;
            for (int i = 0; i < (int)e.props.size(); ++i) {
                if (e.props[i].name == "x") ix = i;
                if (e.props[i].name == "y") iy = i;
                if (e.props[i].name == "z") iz = i;
            }
            if (ix < 0 || iy < 0 || iz < 0) { err = "vertex element lacks x/y/z"; return false; }
            mesh.vertices.resize(e.count);
            for (size_t v = 0; v < e.count && rd.ok; ++v) {
                float xyzv[3] = {0, 0, 0};
                for (int i = 0; i < (int)e.props.size(); ++i) {
                    const Property& p = e.props[i];
                    if (p.isList) {
                        int n = int(rd.value(p.countType));
                        for (int k = 0; k < n; ++k) rd.value(p.type);
                        continue;
                    }
                    double val = rd.value(p.type);
                    if (i == ix) xyzv[0] = float(val);
                    else if (i == iy) xyzv[1] = float(val);
                    else if (i == iz) xyzv[2] = float(val);
                }
                mesh.vertices[v] = {xyzv[0], xyzv[1], xyzv[2]};
            }
        } else if (e.name == "face") {
            int iv = -1;
            for (int i = 0; i < (int)e.props.size(); ++i)
                if (e.props[i].isList && (e.props[i].name == "vertex_indices" || e.props[i].name == "vertex_index")) iv = i;
            if (iv < 0) { err = "face element lacks vertex_indices"; return false; }
            mesh.faces.reserve(e.count);
            for (size_t f = 0; f < e.count && rd.ok; ++f) {
                for (int i = 0; i < (int)e.props.size(); ++i) {
                    const Property& p = e.props[i];
                    if (!p.isList) { rd.value(p.type); continue; }
                    int n = int(rd.value(p.countType));
                    if (i != iv) {
                        for (int k = 0; k < n; ++k) rd.value(p.type);
                        continue;
                    }
                    poly.resize(n > 0 ? n : 0);
                    for (int k = 0; k < n; ++k) poly[k] = uint32_t(rd.value(p.type));
                    for (int k = 1; k + 1 < n; ++k) mesh.faces.push_back({poly[0], poly[k], poly[k + 1]});
                }
            }
        } else {
            // Unknown element: parse and discard.
            for (size_t r = 0; r < e.count && rd.ok; ++r)
                for (const Property& p : e.props) {
                    int n = p.isList ? int(rd.value(p.countType)) : 1;
                    for (int k = 0; k < n; ++k) rd.value(p.type);
                }
        }
        if (!rd.ok) { err = "unexpected end of data in element " + e.name; return false; }
    }

    // Drop faces with out-of-range or repeated indices.
    size_t nv = mesh.vertices.size(), w = 0;
    for (const Face& f : mesh.faces) {
        if (f.a >= nv || f.b >= nv || f.c >= nv || f.a == f.b || f.b == f.c || f.a == f.c) continue;
        mesh.faces[w++] = f;
    }
    mesh.faces.resize(w);
    if (mesh.faces.empty()) { err = "mesh has no triangles"; return false; }
    return true;
}

bool save_ply_binary(const std::string& path, const Mesh& mesh) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "ply\nformat binary_little_endian 1.0\ncomment test file\n"
        << "element vertex " << mesh.vertices.size() << "\nproperty float x\nproperty float y\n"
        << "property float z\nproperty float confidence\n"
        << "element face " << mesh.faces.size() << "\nproperty list uchar int vertex_indices\nend_header\n";
    for (const vec3& v : mesh.vertices) {
        float c = 1.0f;
        out.write(reinterpret_cast<const char*>(&v), 12);
        out.write(reinterpret_cast<const char*>(&c), 4);
    }
    for (const Face& f : mesh.faces) {
        uint8_t n = 3;
        int32_t idx[3] = {int32_t(f.a), int32_t(f.b), int32_t(f.c)};
        out.write(reinterpret_cast<const char*>(&n), 1);
        out.write(reinterpret_cast<const char*>(idx), 12);
    }
    return bool(out);
}

bool save_ply_ascii(const std::string& path, const Mesh& mesh) {
    std::ofstream out(path);
    if (!out) return false;
    out.precision(9);
    out << "ply\nformat ascii 1.0\nelement vertex " << mesh.vertices.size()
        << "\nproperty float x\nproperty float y\nproperty float z\nproperty float intensity\n"
        << "element face " << mesh.faces.size() << "\nproperty list uchar int vertex_indices\nend_header\n";
    for (const vec3& v : mesh.vertices) out << v.x << " " << v.y << " " << v.z << " 0.5\n";
    for (const Face& f : mesh.faces) out << "3 " << f.a << " " << f.b << " " << f.c << "\n";
    return bool(out);
}
