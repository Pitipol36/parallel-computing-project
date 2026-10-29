// Minimal PLY loader: ASCII, binary little-endian and binary big-endian;
// reads vertex x/y/z and polygon faces (polygons are fan-triangulated).
// Any other element or property is parsed and skipped.
#pragma once

#include <string>

#include "mesh.h"

// Returns false and fills err on failure.
bool load_ply(const std::string& path, Mesh& mesh, std::string& err);

// Writes a binary little-endian PLY (used by the unit tests).
bool save_ply_binary(const std::string& path, const Mesh& mesh);
bool save_ply_ascii(const std::string& path, const Mesh& mesh);
