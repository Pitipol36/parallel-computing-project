// raytracer: BVH ray casting of large triangle meshes, sequential CPU vs CUDA.
//
//   raytracer --model data/dragon_vrip.ply --width 1920 --height 1080 --ao 4 \
//             --mode cpu,gpu_bvh_opt --out renders/dragon.png --csv results/results.csv
//   raytracer --verify [--model data/bun_zipper.ply]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "bvh.h"
#include "camera.h"
#include "ply_loader.h"
#include "render.h"

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string model;     // PLY path, or empty for the procedural test scene
    std::string label;     // name used in CSV output (default: file stem)
    std::string tag = "run";  // experiment name written to the CSV
    std::vector<std::string> modes = {"cpu", "gpu_bvh_opt"};
    RenderParams rp;
    bool sizeGiven = false;
    bool aoGiven = false;
    GpuLaunch launch;
    BuildParams bp;
    int runs = 5;
    int cpuRuns = -1;      // default: same as runs
    int warmup = 1;
    std::string out;
    std::string csv;
    bool verify = false;
    bool validate = false;
    bool force = false;
    std::string dataDir = "data";
};

void usage() {
    std::printf(
        "Usage: raytracer [options]\n"
        "  --model PATH        PLY mesh (default: procedural sphere-on-ground test scene)\n"
        "  --label NAME        model name written to the CSV (default: file name)\n"
        "  --tag NAME          experiment name written to the CSV (default: run)\n"
        "  --mode LIST         comma list of: cpu, gpu_brute, gpu_bvh_naive, gpu_bvh_opt,\n"
        "                      gpu_bvh_persistent, gpu_all, all   (default: cpu,gpu_bvh_opt)\n"
        "  --width N --height N   image size (default 1920x1080)\n"
        "  --ao N              ambient-occlusion rays per pixel (default 4)\n"
        "  --ao-radius F       AO ray length as a fraction of the scene diagonal (default 0.1)\n"
        "  --block WxH         CUDA block size (default 16x16; threads/block for persistent)\n"
        "  --runs N            timed runs, median reported (default 5)\n"
        "  --cpu-runs N        timed CPU runs (default: --runs)\n"
        "  --warmup N          untimed warm-up runs (default 1)\n"
        "  --out FILE.png      write the image (one file per mode: FILE_<mode>.png)\n"
        "  --csv FILE          append timing rows to a CSV file\n"
        "  --bins N --leaf N   SAH bins per axis (16) and max leaf size (4)\n"
        "  --yaw D --pitch D --fov D   camera orbit angles and field of view (25, 15, 35)\n"
        "  --up y|z            model up axis (default y)\n"
        "  --brute-rows N      rows per gpu_brute launch (default 8; keeps launches under the\n"
        "                      Windows watchdog)\n"
        "  --force             run gpu_brute even on large meshes\n"
        "  --validate          validate the BVH structure after building it\n"
        "  --verify            unit tests + CPU vs GPU correctness check, prints PASS/FAIL\n"
        "  --data-dir DIR      where --verify looks for bunny/dragon (default data)\n");
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep))
        if (!item.empty()) v.push_back(item);
    return v;
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") o.model = next();
        else if (a == "--label") o.label = next();
        else if (a == "--tag") o.tag = next();
        else if (a == "--mode") o.modes = split(next(), ',');
        else if (a == "--width") { o.rp.width = std::stoi(next()); o.sizeGiven = true; }
        else if (a == "--height") { o.rp.height = std::stoi(next()); o.sizeGiven = true; }
        else if (a == "--ao") { o.rp.aoSamples = std::stoi(next()); o.aoGiven = true; }
        else if (a == "--ao-radius") o.rp.aoRadius = std::stof(next());
        else if (a == "--block") {
            auto p = split(next(), 'x');
            if (p.size() != 2) { std::fprintf(stderr, "--block expects WxH\n"); return false; }
            o.launch.blockX = std::stoi(p[0]);
            o.launch.blockY = std::stoi(p[1]);
        }
        else if (a == "--runs") o.runs = std::max(1, std::stoi(next()));
        else if (a == "--cpu-runs") o.cpuRuns = std::max(1, std::stoi(next()));
        else if (a == "--warmup") o.warmup = std::max(0, std::stoi(next()));
        else if (a == "--out") o.out = next();
        else if (a == "--csv") o.csv = next();
        else if (a == "--bins") o.bp.bins = std::stoi(next());
        else if (a == "--leaf") o.bp.maxLeaf = std::stoi(next());
        else if (a == "--yaw") o.rp.yawDeg = std::stof(next());
        else if (a == "--pitch") o.rp.pitchDeg = std::stof(next());
        else if (a == "--fov") o.rp.fovDeg = std::stof(next());
        else if (a == "--up") o.rp.upAxis = next() == "z" ? 2 : 1;
        else if (a == "--brute-rows") o.launch.bruteRowsPerLaunch = std::stoi(next());
        else if (a == "--force") o.force = true;
        else if (a == "--validate") o.validate = true;
        else if (a == "--verify") o.verify = true;
        else if (a == "--data-dir") o.dataDir = next();
        else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return false; }
    }
    std::vector<std::string> expanded;
    for (const std::string& m : o.modes) {
        if (m == "all" || m == "gpu_all") {
            if (m == "all") expanded.push_back("cpu");
            for (const char* g : {"gpu_brute", "gpu_bvh_naive", "gpu_bvh_opt", "gpu_bvh_persistent"}) expanded.push_back(g);
        } else {
            GpuMode gm;
            if (m != "cpu" && !parse_gpu_mode(m, gm)) { std::fprintf(stderr, "unknown mode %s\n", m.c_str()); return false; }
            expanded.push_back(m);
        }
    }
    o.modes = expanded;
    return true;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

using Clock = std::chrono::high_resolution_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

bool load_scene(const std::string& path, Mesh& mesh) {
    if (path.empty()) {
        mesh = make_test_scene(5);
        return true;
    }
    std::string err;
    auto t0 = Clock::now();
    if (!load_ply(path, mesh, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return false;
    }
    std::printf("Loaded %s: %zu vertices, %zu triangles (%.0f ms)\n", path.c_str(), mesh.vertices.size(),
                mesh.faces.size(), ms_since(t0));
    return true;
}

std::string out_path(const std::string& out, const std::string& mode, bool multi) {
    if (!multi) return out;
    fs::path p(out);
    return (p.parent_path() / (p.stem().string() + "_" + mode + p.extension().string())).string();
}

bool run_brute_ok(const Options& o, size_t tris, size_t pixels) {
    // ~tris * pixels triangle tests; refuse absurd runs unless forced.
    return o.force || double(tris) * double(pixels) <= 2.5e10;
}

// ---------------------------------------------------------------- benchmark

int run_benchmark(const Options& o) {
    Mesh mesh;
    if (!load_scene(o.model, mesh)) return 1;
    std::string label = o.label.empty() ? (o.model.empty() ? "testscene" : fs::path(o.model).stem().string()) : o.label;

    BVH bvh = build_bvh(mesh, o.bp);
    std::printf("BVH: %zu nodes, %d leaves, depth %d, built in %.1f ms (binned SAH, %d bins, leaf %d)\n",
                bvh.nodes.size(), bvh.leafCount, bvh.maxDepth, bvh.buildMs, o.bp.bins, o.bp.maxLeaf);
    if (o.validate) {
        std::string rep;
        bool ok = validate_bvh(bvh, rep);
        std::printf("BVH validation: %s\n%s", ok ? "PASS" : "FAIL", rep.c_str());
    }

    ShadeParams sp = make_shade_params(mesh.bounds(), o.rp);
    const size_t px = size_t(o.rp.width) * o.rp.height;
    const double rays = double(px) * (2 + o.rp.aoSamples);  // nominal: primary + shadow + AO
    std::printf("Render %dx%d, %d AO samples, ~%.1f M rays/frame\n", o.rp.width, o.rp.height, o.rp.aoSamples, rays / 1e6);

    std::string gdesc;
    size_t freeB = 0, totalB = 0;
    bool haveGpu = gpu_info(gdesc, freeB, totalB);
    if (haveGpu) std::printf("GPU: %s, %.2f GB free\n", gdesc.c_str(), double(freeB) / (1 << 30));

    std::ofstream csv;
    if (!o.csv.empty()) {
        bool exists = fs::exists(o.csv);
        if (fs::path(o.csv).has_parent_path()) fs::create_directories(fs::path(o.csv).parent_path());
        csv.open(o.csv, std::ios::app);
        if (!exists)
            csv << "experiment,model,tris,width,height,ao,mode,block,runs,build_ms,kernel_ms,e2e_ms,upload_ms,download_ms,mrays_per_s\n";
    }

    std::printf("\n%-20s %12s %12s %12s %10s\n", "mode", "kernel ms", "e2e ms", "Mrays/s", "speedup");
    double cpuMs = 0;
    int failures = 0;
    for (const std::string& m : o.modes) {
        std::vector<uint32_t> pixels;
        std::vector<double> kern, e2e, up, down;
        char block[32] = "-";
        if (m == "cpu") {
            int runs = o.cpuRuns > 0 ? o.cpuRuns : o.runs;
            for (int r = 0; r < o.warmup + runs; ++r) {
                double ms = render_cpu(bvh, sp, pixels, nullptr);
                if (r >= o.warmup) { kern.push_back(ms); e2e.push_back(ms); up.push_back(0); down.push_back(0); }
            }
        } else {
            GpuMode gm;
            parse_gpu_mode(m, gm);
            if (!haveGpu) { std::printf("%-20s skipped: no CUDA device\n", m.c_str()); continue; }
            if (gm == GpuMode::Brute && !run_brute_ok(o, mesh.faces.size(), px)) {
                std::printf("%-20s skipped: too slow for %zu tris at this resolution (use --force)\n", m.c_str(), mesh.faces.size());
                continue;
            }
            size_t need = gpu_bytes_needed(bvh, mesh, gm, o.rp.width, o.rp.height, false);
            if (need > freeB) {
                std::printf("%-20s skipped: needs %.2f GB, only %.2f GB free\n", m.c_str(), double(need) / (1 << 30),
                            double(freeB) / (1 << 30));
                continue;
            }
            std::snprintf(block, sizeof(block), "%dx%d", o.launch.blockX, o.launch.blockY);
            bool ok = true;
            for (int r = 0; r < o.warmup + o.runs && ok; ++r) {
                GpuTimings tm;
                std::string err;
                ok = render_gpu(bvh, mesh, gm, sp, o.launch, pixels, nullptr, tm, err);
                if (!ok) std::printf("%-20s FAILED: %s\n", m.c_str(), err.c_str());
                if (ok && r >= o.warmup) {
                    kern.push_back(tm.kernelMs); e2e.push_back(tm.e2eMs);
                    up.push_back(tm.uploadMs); down.push_back(tm.downloadMs);
                }
            }
            if (!ok) { ++failures; continue; }
        }
        double k = median(kern), e = median(e2e);
        if (m == "cpu") cpuMs = k;
        double mrays = rays / (k * 1e3);
        char speed[32] = "";
        if (cpuMs > 0 && m != "cpu") std::snprintf(speed, sizeof(speed), "%.1fx", cpuMs / k);
        std::printf("%-20s %12.2f %12.2f %12.1f %10s\n", m.c_str(), k, e, mrays, speed);
        if (csv.is_open())
            csv << o.tag << "," << label << "," << mesh.faces.size() << "," << o.rp.width << "," << o.rp.height << ","
                << o.rp.aoSamples << "," << m << "," << block << "," << kern.size() << "," << bvh.buildMs << ","
                << k << "," << e << "," << median(up) << "," << median(down) << "," << mrays << "\n";
        if (!o.out.empty()) {
            std::string p = out_path(o.out, m, o.modes.size() > 1);
            if (fs::path(p).has_parent_path()) fs::create_directories(fs::path(p).parent_path());
            if (!write_png(p, o.rp.width, o.rp.height, pixels)) std::printf("  could not write %s\n", p.c_str());
        }
    }
    return failures ? 1 : 0;
}

// ---------------------------------------------------------------- verify

bool verify_scene(const Options& o, const std::string& path, const std::string& name) {
    Mesh mesh;
    if (!load_scene(path, mesh)) return false;
    std::printf("\n=== %s (%zu triangles) ===\n", name.c_str(), mesh.faces.size());
    BVH bvh = build_bvh(mesh, o.bp);
    std::string rep;
    bool ok = validate_bvh(bvh, rep);
    std::printf("BVH validation (built in %.1f ms): %s\n%s", bvh.buildMs, ok ? "PASS" : "FAIL", rep.c_str());

    RenderParams rp = o.rp;
    if (!o.sizeGiven) { rp.width = 640; rp.height = 480; }
    if (!o.aoGiven) rp.aoSamples = 4;
    ShadeParams sp = make_shade_params(mesh.bounds(), rp);
    const size_t px = size_t(rp.width) * rp.height;

    std::vector<uint32_t> cpuPix;
    std::vector<int> cpuIds;
    double ms = render_cpu(bvh, sp, cpuPix, &cpuIds);
    size_t hitCount = 0;
    for (int id : cpuIds) hitCount += id >= 0;
    std::printf("cpu reference: %dx%d, %d AO, %.0f ms, %zu pixels hit the mesh\n", rp.width, rp.height,
                rp.aoSamples, ms, hitCount);
    fs::create_directories("renders");
    write_png("renders/verify_" + name + "_cpu.png", rp.width, rp.height, cpuPix);

    std::string gdesc;
    size_t freeB, totalB;
    if (!gpu_info(gdesc, freeB, totalB)) {
        std::printf("no CUDA device: GPU checks FAIL\n");
        return false;
    }
    for (GpuMode gm : {GpuMode::Brute, GpuMode::BvhNaive, GpuMode::BvhOpt, GpuMode::Persistent}) {
        const char* mn = gpu_mode_name(gm);
        if (gm == GpuMode::Brute && !run_brute_ok(o, mesh.faces.size(), px)) {
            std::printf("  %-20s skipped (brute force too slow for this mesh)\n", mn);
            continue;
        }
        std::vector<uint32_t> pix;
        std::vector<int> ids;
        GpuTimings tm;
        std::string err;
        if (!render_gpu(bvh, mesh, gm, sp, o.launch, pix, &ids, tm, err)) {
            std::printf("  %-20s FAIL: %s\n", mn, err.c_str());
            ok = false;
            continue;
        }
        Comparison c = compare_renders(cpuPix, cpuIds, pix, ids);
        double rate = double(c.idMismatches) / double(c.pixels);
        bool pass = rate <= 1e-4 && c.psnr >= 40.0;
        ok = ok && pass;
        std::printf("  %-20s hit-ID mismatches %zu/%zu (%.4f%%), pixels differing %zu, max diff %d, PSNR %s dB -> %s\n",
                    mn, c.idMismatches, c.pixels, 100.0 * rate, c.diffPixels, c.maxAbsDiff,
                    std::isinf(c.psnr) ? "inf" : std::to_string(c.psnr).substr(0, 6).c_str(), pass ? "PASS" : "FAIL");
        if (gm == GpuMode::BvhOpt) write_png("renders/verify_" + name + "_gpu.png", rp.width, rp.height, pix);
    }
    return ok;
}

int run_verify(const Options& o) {
    bool ok = run_unit_tests() == 0;
    std::vector<std::pair<std::string, std::string>> scenes = {{"", "testscene"}};
    if (!o.model.empty()) {
        scenes.push_back({o.model, fs::path(o.model).stem().string()});
    } else {
        for (const char* f : {"bun_zipper.ply", "dragon_vrip.ply"}) {
            fs::path p = fs::path(o.dataDir) / f;
            if (fs::exists(p)) scenes.push_back({p.string(), p.stem().string()});
            else std::printf("(%s not found, skipping; run scripts/download_data)\n", p.string().c_str());
        }
    }
    for (auto& s : scenes) ok = verify_scene(o, s.first, s.second) && ok;
    std::printf("\nVERIFY: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) return 2;
    return o.verify ? run_verify(o) : run_benchmark(o);
}
