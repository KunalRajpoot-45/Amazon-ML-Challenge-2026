#include "blocking.hpp"
#include "tokens.hpp"
#include <algorithm>
#include <cmath>

namespace er {

Blocker::Blocker(const std::vector<Record>& s2, const std::vector<Record>& s3,
                  double maxDfFraction, uint32_t maxDfFloor, uint32_t maxDfCeiling,
                  uint32_t maxCandidatesPerEntity)
    : s2_(s2), s3_(s3), maxDfFraction_(maxDfFraction), maxDfFloor_(maxDfFloor),
      maxDfCeiling_(maxDfCeiling), maxCandidatesPerEntity_(maxCandidatesPerEntity) {
    std::unordered_map<std::string, std::vector<uint32_t>> s2Groups, s3Groups;
    for (uint32_t i = 0; i < s2_.size(); ++i) s2Groups[s2_[i].country].push_back(i);
    for (uint32_t i = 0; i < s3_.size(); ++i) s3Groups[s3_[i].country].push_back(i);

    auto addrTokens = [](const Record& r) { return tokenizeCore(r.addr).tokenHashes; };
    auto nameTokens = [](const Record& r) { return tokenizeCore(r.nameCore).tokenHashes; };
    auto numBigrams = [](const Record& r) { return numericBigramHashes(r.addr); };
    auto addrBigrams = [](const Record& r) { return allTokenBigramHashes(r.addr); };
    auto nameBigrams = [](const Record& r) { return allTokenBigramHashes(r.nameCore); };

    for (auto& kv : s2Groups) {
        CountryIndex ci;
        ci.partitionSize = static_cast<uint32_t>(kv.second.size());
        ci.addrIdx.build(s2_, kv.second, addrTokens);
        ci.nameIdx.build(s2_, kv.second, nameTokens);
        ci.addrNumBigramIdx.build(s2_, kv.second, numBigrams);
        ci.addrBigramIdx.build(s2_, kv.second, addrBigrams);
        ci.nameBigramIdx.build(s2_, kv.second, nameBigrams);
        s2ByCountry_.emplace(kv.first, std::move(ci));
    }
    for (auto& kv : s3Groups) {
        CountryIndex ci;
        ci.partitionSize = static_cast<uint32_t>(kv.second.size());
        ci.addrIdx.build(s3_, kv.second, addrTokens);
        ci.nameIdx.build(s3_, kv.second, nameTokens);
        ci.addrNumBigramIdx.build(s3_, kv.second, numBigrams);
        ci.addrBigramIdx.build(s3_, kv.second, addrBigrams);
        ci.nameBigramIdx.build(s3_, kv.second, nameBigrams);
        s3ByCountry_.emplace(kv.first, std::move(ci));
    }
}

uint32_t Blocker::dfCapFor(uint32_t partitionSize) const {
    uint32_t rel = static_cast<uint32_t>(maxDfFraction_ * partitionSize);
    return std::clamp(rel, maxDfFloor_, maxDfCeiling_);
}

void Blocker::accumulate(const InvertedIndex& idx, const std::vector<uint64_t>& hashes, uint32_t dfCap,
                          uint8_t source, std::unordered_map<uint64_t, std::pair<uint16_t, double>>& acc) const {
    for (uint64_t h : hashes) {
        uint32_t df = idx.df(h);
        if (df == 0 || df > dfCap) continue;
        double weight = 1.0 / std::log2(4.0 + df);
        auto [ptr, len] = idx.postings(h);
        for (uint32_t k = 0; k < len; ++k) {
            uint64_t key = (static_cast<uint64_t>(source) << 32) | ptr[k];
            auto& entry = acc[key];
            entry.first++;
            entry.second += weight;
        }
    }
}

std::vector<Candidate> Blocker::getCandidates(const Record& s1rec) const {
    std::unordered_map<uint64_t, std::pair<uint16_t, double>> acc;
    std::vector<uint64_t> addrHashes = tokenizeCore(s1rec.addr).tokenHashes;
    std::vector<uint64_t> nameHashes = tokenizeCore(s1rec.nameCore).tokenHashes;
    std::vector<uint64_t> numBigramHashes = numericBigramHashes(s1rec.addr);
    std::vector<uint64_t> addrBigramHashes = allTokenBigramHashes(s1rec.addr);
    std::vector<uint64_t> nameBigramHashes = allTokenBigramHashes(s1rec.nameCore);

    auto it2 = s2ByCountry_.find(s1rec.country);
    if (it2 != s2ByCountry_.end()) {
        uint32_t cap = dfCapFor(it2->second.partitionSize);
        accumulate(it2->second.addrIdx, addrHashes, cap, 2, acc);
        accumulate(it2->second.nameIdx, nameHashes, cap, 2, acc);
        accumulate(it2->second.addrNumBigramIdx, numBigramHashes, cap, 2, acc);
        accumulate(it2->second.addrBigramIdx, addrBigramHashes, cap, 2, acc);
        accumulate(it2->second.nameBigramIdx, nameBigramHashes, cap, 2, acc);
    }
    auto it3 = s3ByCountry_.find(s1rec.country);
    if (it3 != s3ByCountry_.end()) {
        uint32_t cap = dfCapFor(it3->second.partitionSize);
        accumulate(it3->second.addrIdx, addrHashes, cap, 3, acc);
        accumulate(it3->second.nameIdx, nameHashes, cap, 3, acc);
        accumulate(it3->second.addrNumBigramIdx, numBigramHashes, cap, 3, acc);
        accumulate(it3->second.addrBigramIdx, addrBigramHashes, cap, 3, acc);
        accumulate(it3->second.nameBigramIdx, nameBigramHashes, cap, 3, acc);
    }

    std::vector<Candidate> out;
    out.reserve(acc.size());
    for (const auto& kv : acc) {
        Candidate c;
        c.source = static_cast<uint8_t>(kv.first >> 32);
        c.idx = static_cast<uint32_t>(kv.first & 0xFFFFFFFFu);
        c.sharedKeyCount = kv.second.first;
        c.weightedScore = kv.second.second;
        out.push_back(c);
    }
    std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
        return a.weightedScore > b.weightedScore;
    });
    if (out.size() > maxCandidatesPerEntity_) out.resize(maxCandidatesPerEntity_);
    return out;
}

} // namespace er
