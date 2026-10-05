#include "tsv_io.hpp"
#include "normalize.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace er {

namespace {

std::string stripCr(std::string s) {
    if (!s.empty() && s.back() == '\r') s.pop_back();
    return s;
}

// Splits on '\t' without any quoting logic (the challenge's TSV files use
// bare tabs as the only delimiter; commas inside fields are literal).
std::vector<std::string> splitTab(const std::string& line) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (true) {
        size_t pos = line.find('\t', start);
        if (pos == std::string::npos) {
            fields.push_back(line.substr(start));
            break;
        }
        fields.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    return fields;
}

std::vector<std::string> splitComma(const std::string& s) {
    std::vector<std::string> out;
    if (s.empty()) return out;
    size_t start = 0;
    while (true) {
        size_t pos = s.find(',', start);
        if (pos == std::string::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

} // namespace

namespace {
struct RawRow {
    std::string id, name, address, country;
};
} // namespace

std::vector<Record> loadSource(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open source file: " + path);

    std::vector<RawRow> raw;
    std::string line;
    std::getline(in, line); // header

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        line = stripCr(line);
        auto f = splitTab(line);
        if (f.size() < 4) continue; // defensive; data is expected to be well-formed
        raw.push_back(RawRow{std::move(f[0]), std::move(f[1]), std::move(f[2]), std::move(f[3])});
    }

    std::vector<Record> records(raw.size());
    unsigned int nThreads = std::max(1u, std::thread::hardware_concurrency());
    nThreads = std::min<unsigned int>(nThreads, 16);
    std::vector<std::thread> workers;
    size_t n = raw.size();
    size_t chunk = (n + nThreads - 1) / std::max<size_t>(1, nThreads);
    for (unsigned int t = 0; t < nThreads; ++t) {
        size_t lo = t * chunk;
        size_t hi = std::min(n, lo + chunk);
        if (lo >= hi) continue;
        workers.emplace_back([&raw, &records, lo, hi]() {
            for (size_t i = lo; i < hi; ++i) {
                records[i].id = std::move(raw[i].id);
                records[i].country = std::move(raw[i].country);
                records[i].nameCore = cleanField(raw[i].name, FieldKind::NameCore);
                records[i].addr = cleanField(raw[i].address, FieldKind::Address);
            }
        });
    }
    for (auto& w : workers) w.join();

    return records;
}

std::vector<GroundTruthEntry> loadGroundTruth(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open ground truth file: " + path);

    std::vector<GroundTruthEntry> entries;
    std::string line;
    std::getline(in, line); // header

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        line = stripCr(line);
        auto f = splitTab(line);
        if (f.empty()) continue;
        GroundTruthEntry e;
        e.source1Id = f[0];
        if (f.size() > 1 && !f[1].empty()) e.matchedIds = splitComma(f[1]);
        entries.push_back(std::move(e));
    }
    return entries;
}

void writeIdListTsv(const std::string& path,
                     const std::string& secondColumnHeader,
                     const std::vector<std::string>& source1Ids,
                     const std::vector<std::vector<std::string>>& idLists) {
    if (source1Ids.size() != idLists.size()) {
        throw std::runtime_error("writeIdListTsv: id/list size mismatch");
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open output file: " + path);

    out << "source1_entity_id\t" << secondColumnHeader << "\n";
    std::string buf;
    for (size_t i = 0; i < source1Ids.size(); ++i) {
        out << source1Ids[i] << '\t';
        const auto& ids = idLists[i];
        for (size_t j = 0; j < ids.size(); ++j) {
            if (j) out << ',';
            out << ids[j];
        }
        out << '\n';
    }
}

} // namespace er
