#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "blocking.hpp"
#include "features.hpp"
#include "tsv_io.hpp"

// C++ half of test-time inference: blocking + feature extraction only. For
// every Source 1 test entity, writes one row per candidate pair (ids + all
// feature columns, no label -- there is no test ground truth) to a plain
// TSV file. The actual match/no-match decision and the two required output
// files (output/matching_results.tsv, output/candidate_pairs.tsv) are
// produced by code/business_entity_resolution/src/predict_model.py,
// which loads the LightGBM model trained by train_model.py and applies its
// tuned threshold. Blocking/feature code itself (blocking.cpp, features.cpp,
// normalize.cpp, tokens.hpp) is unchanged.
//
// Per-entity blocking + feature computation is independent across entities,
// so (mirroring train_main.cpp) the export loop is split across --threads
// worker threads, each writing its assigned entities' rows to its own temp
// file; the main thread concatenates those temp files into the real output
// file once all workers finish.

using namespace er;
namespace fs = std::filesystem;

namespace {

struct Args {
    std::string testDir = "dataset/test";
    std::string featuresOut = "code/business_entity_resolution/artifacts/test_features.tsv";
    double maxDfFraction = 0.01;
    uint32_t maxDfFloor = 500;
    uint32_t maxDfCeiling = 50000;
    uint32_t maxCandidates = 50;
    unsigned int threads = 0; // 0 = auto-detect (hardware_concurrency - 2, floor 1)
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[++i];
        };
        if (arg == "--test-dir") a.testDir = next("--test-dir");
        else if (arg == "--features-out") a.featuresOut = next("--features-out");
        else if (arg == "--max-df-fraction") a.maxDfFraction = std::stod(next("--max-df-fraction"));
        else if (arg == "--max-df-floor") a.maxDfFloor = static_cast<uint32_t>(std::stoul(next("--max-df-floor")));
        else if (arg == "--max-df-ceiling") a.maxDfCeiling = static_cast<uint32_t>(std::stoul(next("--max-df-ceiling")));
        else if (arg == "--max-candidates") a.maxCandidates = static_cast<uint32_t>(std::stoul(next("--max-candidates")));
        else if (arg == "--threads") a.threads = static_cast<unsigned int>(std::stoul(next("--threads")));
        else throw std::runtime_error("unknown argument: " + arg);
    }
    return a;
}

unsigned int resolveThreadCount(unsigned int requested) {
    if (requested > 0) return requested;
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) return 1;
    return hw > 2 ? hw - 2 : 1;
}

std::vector<std::vector<uint32_t>> chunkContiguous(size_t n, unsigned int numThreads) {
    std::vector<std::vector<uint32_t>> chunks(numThreads);
    size_t base = n / numThreads, rem = n % numThreads;
    uint32_t pos = 0;
    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t sz = base + (t < rem ? 1 : 0);
        chunks[t].resize(sz);
        for (size_t k = 0; k < sz; ++k) chunks[t][k] = pos + static_cast<uint32_t>(k);
        pos += static_cast<uint32_t>(sz);
    }
    return chunks;
}

struct PredictPartial {
    uint64_t totalCandidates = 0;
    uint64_t entitiesWithCandidates = 0;
};

PredictPartial predictWorker(const std::vector<uint32_t>& idxChunk, const std::string& partPath,
                              const std::vector<Record>& s1, const std::vector<Record>& s2,
                              const std::vector<Record>& s3, const Blocker& blocker,
                              std::atomic<size_t>& globalProcessed, size_t totalToProcess,
                              std::chrono::steady_clock::time_point t0, std::mutex& coutMutex) {
    PredictPartial partial;
    std::ofstream out(partPath, std::ios::binary);
    out << std::setprecision(6);

    for (uint32_t idx : idxChunk) {
        const Record& rec = s1[idx];
        auto cands = blocker.getCandidates(rec);
        if (!cands.empty()) ++partial.entitiesWithCandidates;
        auto collisions = addrCollisionCounts(cands, s2, s3);

        for (size_t ci = 0; ci < cands.size(); ++ci) {
            const auto& c = cands[ci];
            const Record& other = candidateRecord(c, s2, s3);
            FeatureVec f = computeFeatures(rec, other, c.sharedKeyCount, c.weightedScore, collisions[ci]);
            out << rec.id << '\t' << other.id;
            for (int fi = 0; fi < kFeatureCount; ++fi) out << '\t' << f[fi];
            out << '\n';
            ++partial.totalCandidates;
        }

        size_t done = globalProcessed.fetch_add(1) + 1;
        if (done % 200000 == 0) {
            std::lock_guard<std::mutex> lock(coutMutex);
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cout << "[" << elapsed << "s] processed " << done << "/" << totalToProcess << "\n";
        }
    }
    return partial;
}

void mergePartsInto(const std::string& outPath, const std::vector<std::string>& partPaths) {
    fs::path p(outPath);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream out(outPath, std::ios::binary);
    out << std::setprecision(6);
    out << "source1_entity_id\tcandidate_entity_id";
    for (int i = 0; i < kFeatureCount; ++i) out << '\t' << featureName(i);
    out << '\n';
    for (const auto& part : partPaths) {
        std::ifstream in(part, std::ios::binary);
        out << in.rdbuf();
        in.close();
        fs::remove(part);
    }
}

} // namespace

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf); // flush after every log line, even when piped/redirected
    auto t0 = std::chrono::steady_clock::now();
    Args args;
    try {
        args = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "argument error: " << e.what() << "\n";
        return 1;
    }
    auto elapsed = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    unsigned int numThreads = resolveThreadCount(args.threads);
    std::cout << "[" << elapsed() << "s] using " << numThreads << " worker threads (hardware_concurrency="
              << std::thread::hardware_concurrency() << ")\n";

    std::cout << "[" << elapsed() << "s] loading test sources...\n";
    std::vector<Record> s1 = loadSource(args.testDir + "/test_source1.tsv");
    std::vector<Record> s2 = loadSource(args.testDir + "/test_source2.tsv");
    std::vector<Record> s3 = loadSource(args.testDir + "/test_source3.tsv");
    std::cout << "[" << elapsed() << "s] loaded s1=" << s1.size() << " s2=" << s2.size()
              << " s3=" << s3.size() << "\n";

    std::cout << "[" << elapsed() << "s] building blocker...\n";
    Blocker blocker(s2, s3, args.maxDfFraction, args.maxDfFloor, args.maxDfCeiling, args.maxCandidates);
    std::cout << "[" << elapsed() << "s] blocker ready; scoring " << s1.size()
              << " entities across " << numThreads << " threads...\n";

    auto chunks = chunkContiguous(s1.size(), numThreads);
    std::vector<std::string> partPaths;
    for (unsigned int t = 0; t < numThreads; ++t)
        partPaths.push_back(args.featuresOut + ".part" + std::to_string(t));

    std::atomic<size_t> processed{0};
    std::mutex coutMutex;
    std::vector<std::thread> workers;
    std::vector<PredictPartial> results(numThreads);
    for (unsigned int t = 0; t < numThreads; ++t) {
        workers.emplace_back([&, t]() {
            results[t] = predictWorker(chunks[t], partPaths[t], s1, s2, s3, blocker, processed, s1.size(), t0,
                                        coutMutex);
        });
    }
    for (auto& th : workers) th.join();

    uint64_t totalCandidates = 0, entitiesWithCandidates = 0;
    for (const auto& p : results) {
        totalCandidates += p.totalCandidates;
        entitiesWithCandidates += p.entitiesWithCandidates;
    }
    mergePartsInto(args.featuresOut, partPaths);

    std::cout << "[" << elapsed() << "s] total candidates=" << totalCandidates
              << " entities with >=1 candidate=" << entitiesWithCandidates << "/" << s1.size() << "\n";
    std::cout << "[" << elapsed() << "s] wrote " << args.featuresOut << "\n";
    std::cout << "[" << elapsed() << "s] predict with:\n"
              << "  python3 code/business_entity_resolution/src/predict_model.py \\\n"
              << "    --test-dir " << args.testDir << " \\\n"
              << "    --features " << args.featuresOut << " \\\n"
              << "    --model code/business_entity_resolution/artifacts/model.txt \\\n"
              << "    --matching-out output/matching_results.tsv \\\n"
              << "    --candidate-out output/candidate_pairs.tsv\n";

    return 0;
}
