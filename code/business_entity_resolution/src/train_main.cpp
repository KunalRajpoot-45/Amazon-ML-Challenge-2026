#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "blocking.hpp"
#include "features.hpp"
#include "metrics.hpp"
#include "tsv_io.hpp"

// This program is the C++ half of the pipeline: blocking + feature
// extraction only. It writes one row per candidate pair (ids + all feature
// columns + a label) to plain TSV files for the train-fit and validation
// splits. Model training moved to Python (code/business_entity_resolution/src/train_model.py,
// using pip-installed LightGBM) after diagnosis showed several features only
// help conditionally on others -- an interaction effect plain logistic
// regression couldn't exploit. Blocking and feature computation themselves
// are unchanged C++ (blocking.cpp, features.cpp, normalize.cpp, tokens.hpp).
//
// Per-entity blocking + feature computation is independent across entities
// (Blocker::getCandidates is const/read-only after construction, and
// computeFeatures touches no shared state), so both the train-fit and
// validation export loops are split across --threads worker threads, each
// writing its assigned entities' rows to its own temp file; the main thread
// concatenates those temp files (in the same order entities were sliced) into
// the real output file once all workers finish, so row order within a thread
// is preserved and the only cross-entity reordering is at thread-chunk
// granularity (harmless -- training doesn't care about row order).

using namespace er;
namespace fs = std::filesystem;

namespace {

struct Args {
    std::string trainDir = "dataset/train";
    std::string trainFeaturesOut = "code/business_entity_resolution/artifacts/train_features.tsv";
    std::string valFeaturesOut = "code/business_entity_resolution/artifacts/val_features.tsv";
    std::string valEntitiesOut = "code/business_entity_resolution/artifacts/val_entities.tsv";
    std::string dumpExamples; // empty = disabled
    double valFraction = 0.10;
    double maxDfFraction = 0.01;
    uint32_t maxDfFloor = 500;
    uint32_t maxDfCeiling = 50000;
    uint32_t maxCandidates = 50;
    int negMin = 5;
    int negMultiplier = 3;
    unsigned int seed = 42;
    unsigned int threads = 0; // 0 = auto-detect (hardware_concurrency - 2, floor 1)
    size_t limitEntities = 0; // 0 = unlimited; debug/validation flag to cap train+val entity counts
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[++i];
        };
        if (arg == "--train-dir") a.trainDir = next("--train-dir");
        else if (arg == "--train-features-out") a.trainFeaturesOut = next("--train-features-out");
        else if (arg == "--val-features-out") a.valFeaturesOut = next("--val-features-out");
        else if (arg == "--val-entities-out") a.valEntitiesOut = next("--val-entities-out");
        else if (arg == "--val-fraction") a.valFraction = std::stod(next("--val-fraction"));
        else if (arg == "--max-df-fraction") a.maxDfFraction = std::stod(next("--max-df-fraction"));
        else if (arg == "--max-df-floor") a.maxDfFloor = static_cast<uint32_t>(std::stoul(next("--max-df-floor")));
        else if (arg == "--max-df-ceiling") a.maxDfCeiling = static_cast<uint32_t>(std::stoul(next("--max-df-ceiling")));
        else if (arg == "--max-candidates") a.maxCandidates = static_cast<uint32_t>(std::stoul(next("--max-candidates")));
        else if (arg == "--neg-min") a.negMin = std::stoi(next("--neg-min"));
        else if (arg == "--neg-multiplier") a.negMultiplier = std::stoi(next("--neg-multiplier"));
        else if (arg == "--seed") a.seed = static_cast<unsigned int>(std::stoul(next("--seed")));
        else if (arg == "--threads") a.threads = static_cast<unsigned int>(std::stoul(next("--threads")));
        else if (arg == "--limit-entities") a.limitEntities = static_cast<size_t>(std::stoull(next("--limit-entities")));
        else if (arg == "--dump-examples") a.dumpExamples = next("--dump-examples");
        else throw std::runtime_error("unknown argument: " + arg);
    }
    return a;
}

unsigned int resolveThreadCount(unsigned int requested) {
    if (requested > 0) return requested;
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) return 1; // hardware_concurrency() is allowed to return 0 if undetectable
    return hw > 2 ? hw - 2 : 1;
}

double f0_5FromCounts(int tp, int predCount, int truthCount) {
    if (predCount == 0 && truthCount == 0) return 1.0;
    double precision = predCount ? static_cast<double>(tp) / predCount : 0.0;
    double recall = truthCount ? static_cast<double>(tp) / truthCount : 0.0;
    if (precision == 0.0 && recall == 0.0) return 0.0;
    return (1.25 * precision * recall) / (0.25 * precision + recall);
}

// bucket: 0 = singleton (truth empty), 1 = exactly one true match, 2 = multiple true matches
int bucketOf(size_t truthSize) {
    if (truthSize == 0) return 0;
    if (truthSize == 1) return 1;
    return 2;
}
const char* bucketName(int b) { return b == 0 ? "singleton" : b == 1 ? "single-match" : "multi-match"; }

struct EvalResult {
    double meanF0_5 = 0.0;
    double bucketMean[3] = {0, 0, 0};
    long bucketCount[3] = {0, 0, 0};
};

void printEvalResult(const std::string& label, const EvalResult& r) {
    std::cout << "  " << label << ": overall=" << r.meanF0_5 << "\n";
    for (int b = 0; b < 3; ++b) {
        std::cout << "    " << bucketName(b) << " (n=" << r.bucketCount[b] << "): "
                  << (r.bucketCount[b] ? r.bucketMean[b] : 0.0) << "\n";
    }
}

void writeFeatureHeader(std::ofstream& out) {
    out << "source1_entity_id\tcandidate_entity_id";
    for (int i = 0; i < kFeatureCount; ++i) out << '\t' << featureName(i);
    out << "\tlabel\n";
}

void writeFeatureRow(std::ofstream& out, const std::string& s1Id, const std::string& candId,
                      const FeatureVec& f, int label) {
    out << s1Id << '\t' << candId;
    for (int i = 0; i < kFeatureCount; ++i) out << '\t' << f[i];
    out << '\t' << label << '\n';
}

// Concatenates numbered temp part files (written by worker threads, one per
// thread, no header) into outPath with a single header line, then removes
// the parts. Binary copy so no re-parsing/re-formatting of the huge files.
void mergePartsInto(const std::string& outPath, const std::vector<std::string>& partPaths) {
    fs::path p(outPath);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream out(outPath, std::ios::binary);
    out << std::setprecision(6);
    writeFeatureHeader(out);
    for (const auto& part : partPaths) {
        std::ifstream in(part, std::ios::binary);
        out << in.rdbuf();
        in.close();
        fs::remove(part);
    }
}

std::vector<std::vector<uint32_t>> chunkContiguous(const std::vector<uint32_t>& idx, unsigned int numThreads) {
    std::vector<std::vector<uint32_t>> chunks(numThreads);
    size_t n = idx.size();
    size_t base = n / numThreads, rem = n % numThreads;
    size_t pos = 0;
    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t sz = base + (t < rem ? 1 : 0);
        chunks[t].assign(idx.begin() + pos, idx.begin() + pos + sz);
        pos += sz;
    }
    return chunks;
}

struct TrainPartial {
    uint64_t gtTotal = 0, gtFound = 0, pairs = 0, positives = 0;
};

struct EntityRow {
    std::string candId;
    FeatureVec f;
    int label;
};

// The actual candidate-pair generation + hard-negative-sampling logic for
// one train-fit entity, shared between trainWorker (writes these rows to
// its part file) and selfCheckTrainFit (recomputes the same rows in-memory,
// independently of any thread, to verify the parallel run's output). Pulling
// this out means the self-check calls the exact same code path the workers
// do, not a re-implementation that could drift and give a false PASS.
std::vector<EntityRow> computeCandidateRows(const Record& rec, const std::vector<Record>& s2,
                                             const std::vector<Record>& s3, const Blocker& blocker,
                                             const std::unordered_set<std::string>& truthSet,
                                             int negMin, int negMultiplier) {
    std::vector<EntityRow> rows;
    auto cands = blocker.getCandidates(rec);
    auto collisions = addrCollisionCounts(cands, s2, s3);

    int posCount = 0;
    for (const auto& c : cands) {
        const Record& other = candidateRecord(c, s2, s3);
        if (truthSet.count(other.id)) ++posCount;
    }
    int negCap = std::max(negMin, negMultiplier * std::max(1, posCount));

    int negKept = 0;
    for (size_t ci = 0; ci < cands.size(); ++ci) {
        const auto& c = cands[ci];
        const Record& other = candidateRecord(c, s2, s3);
        bool isPos = truthSet.count(other.id) > 0;
        if (!isPos) {
            if (negKept >= negCap) continue;
            ++negKept;
        }
        EntityRow row;
        row.candId = other.id;
        row.f = computeFeatures(rec, other, c.sharedKeyCount, c.weightedScore, collisions[ci]);
        row.label = isPos ? 1 : 0;
        rows.push_back(std::move(row));
    }
    return rows;
}

// Canonical string form of one row's non-id columns, matching exactly what
// writeFeatureRow puts after the candidate id -- used so the self-check's
// freshly-recomputed rows and the parallel run's on-disk rows are compared
// as identically-formatted text, not floating-point values that could differ
// by formatting alone.
std::string formatRowTail(const EntityRow& row) {
    std::ostringstream oss;
    oss << std::setprecision(6);
    for (int i = 0; i < kFeatureCount; ++i) oss << '\t' << row.f[i];
    oss << '\t' << row.label;
    return oss.str();
}

// Correctness gate for the parallel train-fit export: samples a handful of
// entities, recomputes their rows independently (single call, no threads
// involved), and checks the exact same rows appear in the already-merged
// output file. Catches parallelization bugs (chunk boundaries dropping/
// duplicating entities, a bad merge losing or corrupting part-file data) --
// it would NOT catch a data race inside computeCandidateRows/getCandidates
// itself, since that function is pure/stateless and produces identical
// output regardless of which thread calls it; the risk that check would
// target is covered instead by the chunk-partition being an exact,
// non-overlapping split of trainIdx (guaranteed by chunkContiguous) and by
// mergePartsInto concatenating every part exactly once (checked here via
// the row-count match, which would catch a dropped/duplicated part file).
bool selfCheckTrainFit(const std::string& trainFeaturesOut, const std::vector<Record>& s1,
                       const std::vector<Record>& s2, const std::vector<Record>& s3,
                       const Blocker& blocker,
                       const std::unordered_map<std::string, const std::vector<std::string>*>& gtMap,
                       const std::vector<uint32_t>& trainIdx, int negMin, int negMultiplier,
                       unsigned int seed, size_t sampleSize) {
    static const std::vector<std::string> kEmpty;
    auto truthFor = [&](const std::string& id) -> const std::vector<std::string>& {
        auto it = gtMap.find(id);
        return it == gtMap.end() ? kEmpty : *it->second;
    };

    std::mt19937 rng(seed);
    std::vector<uint32_t> sampled = trainIdx;
    std::shuffle(sampled.begin(), sampled.end(), rng);
    if (sampled.size() > sampleSize) sampled.resize(sampleSize);

    std::unordered_map<std::string, std::vector<std::string>> expected; // s1Id -> sorted formatted row tails
    expected.reserve(sampled.size() * 2);
    for (uint32_t idx : sampled) {
        const Record& rec = s1[idx];
        const auto& truth = truthFor(rec.id);
        std::unordered_set<std::string> truthSet(truth.begin(), truth.end());
        auto rows = computeCandidateRows(rec, s2, s3, blocker, truthSet, negMin, negMultiplier);
        std::vector<std::string> tails;
        tails.reserve(rows.size());
        for (const auto& row : rows) tails.push_back(row.candId + formatRowTail(row));
        std::sort(tails.begin(), tails.end());
        expected[rec.id] = std::move(tails);
    }

    std::unordered_map<std::string, std::vector<std::string>> actual;
    actual.reserve(expected.size() * 2);
    {
        std::ifstream in(trainFeaturesOut, std::ios::binary);
        std::string line;
        std::getline(in, line); // header
        while (std::getline(in, line)) {
            size_t tab1 = line.find('\t');
            if (tab1 == std::string::npos) continue;
            std::string s1Id = line.substr(0, tab1);
            auto it = expected.find(s1Id);
            if (it == expected.end()) continue;
            actual[s1Id].push_back(line.substr(tab1 + 1));
        }
    }
    for (auto& kv : actual) std::sort(kv.second.begin(), kv.second.end());

    int mismatches = 0;
    for (const auto& kv : expected) {
        auto it = actual.find(kv.first);
        const std::vector<std::string>& got = (it == actual.end()) ? kEmpty : it->second;
        if (got != kv.second) {
            if (mismatches < 5) {
                std::cout << "[self-check]   MISMATCH entity " << kv.first << ": expected "
                          << kv.second.size() << " rows, found " << got.size() << " in output file\n";
            }
            ++mismatches;
        }
    }

    if (mismatches == 0) {
        std::cout << "[self-check] PASS: " << expected.size()
                  << "/" << expected.size() << " sampled entities match exactly between the freshly "
                  << "recomputed rows and the merged parallel-export output\n";
        return true;
    }
    std::cout << "[self-check] FAIL: " << mismatches << "/" << expected.size()
              << " sampled entities did NOT match -- parallel export output looks wrong, aborting "
              << "before the validation split (and before wasting more time on it)\n";
    return false;
}

// Processes one contiguous slice of trainIdx, writing candidate-pair feature
// rows (with hard-negative sampling) to its own temp file. Reads only
// (s1/s2/s3/blocker/gtMap all const& / never mutated after construction),
// so this runs safely alongside identical calls on other threads.
TrainPartial trainWorker(const std::vector<uint32_t>& idxChunk, const std::string& partPath,
                          const std::vector<Record>& s1, const std::vector<Record>& s2,
                          const std::vector<Record>& s3, const Blocker& blocker,
                          const std::unordered_map<std::string, const std::vector<std::string>*>& gtMap,
                          int negMin, int negMultiplier,
                          std::atomic<size_t>& globalProcessed, size_t totalToProcess,
                          std::chrono::steady_clock::time_point t0, std::mutex& coutMutex) {
    static const std::vector<std::string> kEmpty;
    auto truthFor = [&](const std::string& id) -> const std::vector<std::string>& {
        auto it = gtMap.find(id);
        return it == gtMap.end() ? kEmpty : *it->second;
    };

    TrainPartial partial;
    std::ofstream out(partPath, std::ios::binary);
    out << std::setprecision(6);

    for (uint32_t idx : idxChunk) {
        const Record& rec = s1[idx];
        const auto& truth = truthFor(rec.id);
        std::unordered_set<std::string> truthSet(truth.begin(), truth.end());
        partial.gtTotal += truthSet.size();

        auto rows = computeCandidateRows(rec, s2, s3, blocker, truthSet, negMin, negMultiplier);
        for (const auto& row : rows) {
            writeFeatureRow(out, rec.id, row.candId, row.f, row.label);
            ++partial.pairs;
            if (row.label) { ++partial.positives; ++partial.gtFound; }
        }

        size_t done = globalProcessed.fetch_add(1) + 1;
        if (done % 200000 == 0) {
            std::lock_guard<std::mutex> lock(coutMutex);
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cout << "[" << elapsed << "s] train-fit candidates: " << done << "/" << totalToProcess << "\n";
        }
    }
    return partial;
}

struct ValEntity {
    std::string s1Id;
    std::vector<std::string> candidateIds;
    std::vector<std::string> truth;
};

struct ValPartial {
    uint64_t gtTotal = 0, gtFound = 0, pairs = 0, positives = 0;
    std::vector<ValEntity> entities;
};

ValPartial valWorker(const std::vector<uint32_t>& idxChunk, const std::string& partPath,
                      const std::vector<Record>& s1, const std::vector<Record>& s2,
                      const std::vector<Record>& s3, const Blocker& blocker,
                      const std::unordered_map<std::string, const std::vector<std::string>*>& gtMap,
                      std::atomic<size_t>& globalProcessed, size_t totalToProcess,
                      std::chrono::steady_clock::time_point t0, std::mutex& coutMutex) {
    static const std::vector<std::string> kEmpty;
    auto truthFor = [&](const std::string& id) -> const std::vector<std::string>& {
        auto it = gtMap.find(id);
        return it == gtMap.end() ? kEmpty : *it->second;
    };

    ValPartial partial;
    partial.entities.reserve(idxChunk.size());
    std::ofstream out(partPath, std::ios::binary);
    out << std::setprecision(6);

    for (uint32_t idx : idxChunk) {
        const Record& rec = s1[idx];
        ValEntity ve;
        ve.s1Id = rec.id;
        ve.truth = truthFor(rec.id);
        std::unordered_set<std::string> truthSet(ve.truth.begin(), ve.truth.end());
        partial.gtTotal += truthSet.size();

        auto cands = blocker.getCandidates(rec);
        auto collisions = addrCollisionCounts(cands, s2, s3);
        ve.candidateIds.reserve(cands.size());
        for (size_t ci = 0; ci < cands.size(); ++ci) {
            const auto& c = cands[ci];
            const Record& other = candidateRecord(c, s2, s3);
            bool isPos = truthSet.count(other.id) > 0;
            if (isPos) ++partial.gtFound;
            ve.candidateIds.push_back(other.id);
            FeatureVec f = computeFeatures(rec, other, c.sharedKeyCount, c.weightedScore, collisions[ci]);
            writeFeatureRow(out, rec.id, other.id, f, isPos ? 1 : 0);
            ++partial.pairs;
            if (isPos) ++partial.positives;
        }
        partial.entities.push_back(std::move(ve));

        size_t done = globalProcessed.fetch_add(1) + 1;
        if (done % 100000 == 0) {
            std::lock_guard<std::mutex> lock(coutMutex);
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cout << "[" << elapsed << "s] val candidates: " << done << "/" << totalToProcess << "\n";
        }
    }
    return partial;
}

template <typename Worker>
auto runParallel(const std::vector<std::vector<uint32_t>>& chunks,
                  const std::vector<std::string>& partPaths, Worker worker) {
    using Partial = std::invoke_result_t<Worker, const std::vector<uint32_t>&, const std::string&>;
    std::vector<std::thread> workers;
    std::vector<Partial> results(chunks.size());
    for (size_t t = 0; t < chunks.size(); ++t) {
        workers.emplace_back([&, t]() { results[t] = worker(chunks[t], partPaths[t]); });
    }
    for (auto& th : workers) th.join();
    return results;
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

    std::cout << "[" << elapsed() << "s] loading train sources...\n";
    std::vector<Record> s1 = loadSource(args.trainDir + "/train_source1.tsv");
    std::vector<Record> s2 = loadSource(args.trainDir + "/train_source2.tsv");
    std::vector<Record> s3 = loadSource(args.trainDir + "/train_source3.tsv");
    std::vector<GroundTruthEntry> gt = loadGroundTruth(args.trainDir + "/train_ground_truth.tsv");
    std::cout << "[" << elapsed() << "s] loaded s1=" << s1.size() << " s2=" << s2.size()
              << " s3=" << s3.size() << " gt=" << gt.size() << "\n";

    std::unordered_map<std::string, const std::vector<std::string>*> gtMap;
    gtMap.reserve(gt.size() * 2);
    for (const auto& e : gt) gtMap[e.source1Id] = &e.matchedIds;

    std::vector<uint32_t> order(s1.size());
    for (uint32_t i = 0; i < s1.size(); ++i) order[i] = i;
    std::mt19937 rng(args.seed);
    std::shuffle(order.begin(), order.end(), rng);
    size_t valCount = static_cast<size_t>(s1.size() * args.valFraction);
    std::vector<uint32_t> valIdx(order.begin(), order.begin() + valCount);
    std::vector<uint32_t> trainIdx(order.begin() + valCount, order.end());
    if (args.limitEntities > 0) {
        if (trainIdx.size() > args.limitEntities) trainIdx.resize(args.limitEntities);
        if (valIdx.size() > args.limitEntities) valIdx.resize(args.limitEntities);
        std::cout << "[" << elapsed() << "s] --limit-entities " << args.limitEntities
                  << " applied (debug/validation mode)\n";
    }
    std::cout << "[" << elapsed() << "s] split: train=" << trainIdx.size()
              << " val=" << valIdx.size() << "\n";

    std::cout << "[" << elapsed() << "s] building blocker (maxDfFraction=" << args.maxDfFraction
              << ", maxDfFloor=" << args.maxDfFloor << ", maxDfCeiling=" << args.maxDfCeiling
              << ", maxCandidates=" << args.maxCandidates << ")...\n";
    Blocker blocker(s2, s3, args.maxDfFraction, args.maxDfFloor, args.maxDfCeiling, args.maxCandidates);
    std::cout << "[" << elapsed() << "s] blocker ready\n";

    std::mutex coutMutex;

    // ---- train-fit split ----
    auto trainChunks = chunkContiguous(trainIdx, numThreads);
    std::vector<std::string> trainPartPaths;
    for (unsigned int t = 0; t < numThreads; ++t)
        trainPartPaths.push_back(args.trainFeaturesOut + ".part" + std::to_string(t));

    std::atomic<size_t> trainProcessed{0};
    std::cout << "[" << elapsed() << "s] exporting train-fit split across " << numThreads << " threads...\n";
    auto trainResults = runParallel(
        trainChunks, trainPartPaths,
        [&](const std::vector<uint32_t>& chunk, const std::string& partPath) {
            return trainWorker(chunk, partPath, s1, s2, s3, blocker, gtMap, args.negMin, args.negMultiplier,
                                trainProcessed, trainIdx.size(), t0, coutMutex);
        });

    uint64_t trainGtTotal = 0, trainGtFound = 0, trainPairs = 0, trainPositives = 0;
    for (const auto& p : trainResults) {
        trainGtTotal += p.gtTotal;
        trainGtFound += p.gtFound;
        trainPairs += p.pairs;
        trainPositives += p.positives;
    }
    mergePartsInto(args.trainFeaturesOut, trainPartPaths);
    std::cout << "[" << elapsed() << "s] wrote " << args.trainFeaturesOut << ": " << trainPairs
              << " pairs (" << trainPositives << " positive), "
              << "recall ceiling on train-fit split = "
              << (trainGtTotal ? 100.0 * trainGtFound / trainGtTotal : 0.0) << "%\n";

    std::cout << "[" << elapsed() << "s] self-check: verifying " << 500
              << " random entities against the merged parallel-export output...\n";
    bool selfCheckOk = selfCheckTrainFit(args.trainFeaturesOut, s1, s2, s3, blocker, gtMap, trainIdx,
                                         args.negMin, args.negMultiplier, args.seed + 1, 500);
    if (!selfCheckOk) {
        std::cerr << "[" << elapsed() << "s] self-check FAILED -- stopping before the validation split.\n";
        return 1;
    }

    // ---- validation split ----
    auto valChunks = chunkContiguous(valIdx, numThreads);
    std::vector<std::string> valPartPaths;
    for (unsigned int t = 0; t < numThreads; ++t)
        valPartPaths.push_back(args.valFeaturesOut + ".part" + std::to_string(t));

    std::atomic<size_t> valProcessed{0};
    std::cout << "[" << elapsed() << "s] exporting validation split across " << numThreads << " threads...\n";
    auto valResults = runParallel(
        valChunks, valPartPaths,
        [&](const std::vector<uint32_t>& chunk, const std::string& partPath) {
            return valWorker(chunk, partPath, s1, s2, s3, blocker, gtMap, valProcessed, valIdx.size(), t0, coutMutex);
        });

    uint64_t valGtTotal = 0, valGtFound = 0, valPairs = 0, valPositives = 0;
    std::vector<ValEntity> val;
    val.reserve(valIdx.size());
    for (auto& p : valResults) {
        valGtTotal += p.gtTotal;
        valGtFound += p.gtFound;
        valPairs += p.pairs;
        valPositives += p.positives;
        for (auto& ve : p.entities) val.push_back(std::move(ve));
    }
    mergePartsInto(args.valFeaturesOut, valPartPaths);
    std::cout << "[" << elapsed() << "s] wrote " << args.valFeaturesOut << ": " << valPairs
              << " pairs (" << valPositives << " positive), "
              << "recall ceiling on validation split = "
              << (valGtTotal ? 100.0 * valGtFound / valGtTotal : 0.0) << "%\n";

    // Every validation entity (including zero-candidate ones, which never
    // appear in val_features.tsv) with its true-match count, so the Python
    // threshold search can score singletons/blocking-misses correctly
    // instead of silently dropping them from the macro average.
    {
        fs::path p(args.valEntitiesOut);
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
        std::ofstream ve_out(args.valEntitiesOut, std::ios::binary);
        ve_out << "source1_entity_id\ttruth_count\thas_candidates\n";
        for (const auto& ve : val) {
            ve_out << ve.s1Id << '\t' << ve.truth.size() << '\t' << (ve.candidateIds.empty() ? 0 : 1) << '\n';
        }
    }
    std::cout << "[" << elapsed() << "s] wrote " << args.valEntitiesOut << ": " << val.size() << " entities\n";

    std::cout << "[" << elapsed() << "s] oracle ceiling (perfect model given current blocking):\n";
    EvalResult oracle;
    {
        double sum = 0.0;
        for (const auto& ve : val) {
            std::unordered_set<std::string> truthSet(ve.truth.begin(), ve.truth.end());
            int tp = 0;
            for (const auto& cid : ve.candidateIds) {
                if (truthSet.count(cid)) ++tp;
            }
            double f = f0_5FromCounts(tp, tp, static_cast<int>(ve.truth.size()));
            sum += f;
            int b = bucketOf(ve.truth.size());
            oracle.bucketMean[b] += f;
            oracle.bucketCount[b]++;
        }
        oracle.meanF0_5 = val.empty() ? 0.0 : sum / val.size();
        for (int b = 0; b < 3; ++b) {
            if (oracle.bucketCount[b] > 0) oracle.bucketMean[b] /= oracle.bucketCount[b];
        }
    }
    printEvalResult("oracle", oracle);

    if (!args.dumpExamples.empty()) {
        std::cout << "[" << elapsed() << "s] dumping blocking-miss FN examples to " << args.dumpExamples << "...\n";
        std::unordered_map<std::string, const Record*> byId;
        byId.reserve((s2.size() + s3.size()) * 2);
        for (const auto& r : s2) byId[r.id] = &r;
        for (const auto& r : s3) byId[r.id] = &r;

        std::unordered_map<std::string, const Record*> s1ById;
        s1ById.reserve(s1.size() * 2);
        for (const auto& r : s1) s1ById[r.id] = &r;

        std::ofstream out(args.dumpExamples);
        out << "=== FALSE NEGATIVES (true match never appeared as a candidate -- blocking miss) ===\n";
        int fnCount = 0;
        const int maxPer = 100;
        for (size_t i = 0; i < val.size() && fnCount < maxPer; ++i) {
            const auto& ve = val[i];
            std::unordered_set<std::string> candSet(ve.candidateIds.begin(), ve.candidateIds.end());
            auto qit = s1ById.find(ve.s1Id);
            if (qit == s1ById.end()) continue;
            const Record& q = *qit->second;
            for (const auto& tid : ve.truth) {
                if (candSet.count(tid)) continue;
                auto it = byId.find(tid);
                if (it == byId.end()) continue;
                const Record* m = it->second;
                out << "S1=" << q.id << " name=[" << q.nameCore << "] addr=[" << q.addr
                    << "] country=" << q.country << "\n  MISSED " << tid << " name=[" << m->nameCore
                    << "] addr=[" << m->addr << "] country=" << m->country << "\n\n";
                if (++fnCount >= maxPer) break;
            }
        }
        std::cout << "  dumped " << fnCount << " blocking-miss FN examples\n";
    }

    std::cout << "[" << elapsed() << "s] done. Train with:\n"
              << "  python3 code/business_entity_resolution/src/train_model.py \\\n"
              << "    --train-features " << args.trainFeaturesOut << " \\\n"
              << "    --val-features " << args.valFeaturesOut << " \\\n"
              << "    --val-entities " << args.valEntitiesOut << " \\\n"
              << "    --model-out code/business_entity_resolution/artifacts/model.txt\n";

    return 0;
}
