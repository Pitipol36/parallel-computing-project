#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "render.h"

bool write_png(const std::string& path, int w, int h, const std::vector<uint32_t>& rgba) {
    return stbi_write_png(path.c_str(), w, h, 4, rgba.data(), w * 4) != 0;
}
