#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "inverted_index.hpp"
#include "record.hpp"

namespace er {

struct Candidate {
    uint8_t source;        // 2 or 3
    uint32_t idx;           // index into the source2/source3 record vector
    uint16_t sharedKeyCount; // raw count of blocking keys shared (across all key types)
    double weightedScore;   // IDF-weighted score, used for ranking/truncation
};

// Blocking / candidate-generation stage. Builds per-country inverted indices
// over three independent key types for source2 and source3 -- address
// tokens, stopword-stripped core-name tokens, and address numeric-token
// bigrams (adjacent digit-token pairs, e.g. house/plot number fragments,
// far more specific than a single numeric token) -- then for each source1
// record unions the postings of its own keys and ranks candidates by an
// IDF-weighted score.
//
// Document-frequency capping is RELATIVE to each country/source partition's
// own size (maxDfFraction, with an absolute floor/ceiling), not a fixed
// absolute count: an absolute cap makes blocking behave inconsistently
// across countries of very different sizes (and inconsistently between a
// dev subset and the full corpus), silently discarding useful moderately-
// common tokens (e.g. major city names) in large partitions while being too
// permissive in small ones. Even tokens under the cap are weighted by
// inverse document frequency when accumulating a candidate's score, so a
// candidate sharing one rare, highly specific token outranks one sharing
// several generic ones.
//
// Country is treated as an open string key throughout: whatever countries
// appear in the source2/source3 records passed in (train or test) is what
// gets indexed, with no hard-coded country set.
class Blocker {
public:
    Blocker(const std::vector<Record>& s2, const std::vector<Record>& s3,
            double maxDfFraction = 0.01, uint32_t maxDfFloor = 500, uint32_t maxDfCeiling = 50000,
            uint32_t maxCandidatesPerEntity = 50);

    std::vector<Candidate> getCandidates(const Record& s1rec) const;

private:
    struct CountryIndex {
        InvertedIndex addrIdx;
        InvertedIndex nameIdx;
        InvertedIndex addrNumBigramIdx;
        InvertedIndex addrBigramIdx;
        InvertedIndex nameBigramIdx;
        uint32_t partitionSize = 0;
    };

    const std::vector<Record>& s2_;
    const std::vector<Record>& s3_;
    double maxDfFraction_;
    uint32_t maxDfFloor_;
    uint32_t maxDfCeiling_;
    uint32_t maxCandidatesPerEntity_;
    std::unordered_map<std::string, CountryIndex> s2ByCountry_;
    std::unordered_map<std::string, CountryIndex> s3ByCountry_;

    uint32_t dfCapFor(uint32_t partitionSize) const;
    void accumulate(const InvertedIndex& idx, const std::vector<uint64_t>& hashes, uint32_t dfCap,
                     uint8_t source, std::unordered_map<uint64_t, std::pair<uint16_t, double>>& acc) const;
};

inline const Record& candidateRecord(const Candidate& c, const std::vector<Record>& s2,
                                      const std::vector<Record>& s3) {
    return c.source == 2 ? s2[c.idx] : s3[c.idx];
}

// For each candidate, how many OTHER candidates in the same set share its
// exact address string. A proxy for "this address hosts multiple distinct
// businesses" (shared building/plaza), used as a feature so the model can
// discount address-only similarity there.
inline std::vector<uint32_t> addrCollisionCounts(const std::vector<Candidate>& cands,
                                                  const std::vector<Record>& s2,
                                                  const std::vector<Record>& s3) {
    std::unordered_map<std::string, uint32_t> byAddr;
    byAddr.reserve(cands.size() * 2);
    for (const auto& c : cands) {
        const std::string& addr = candidateRecord(c, s2, s3).addr;
        if (!addr.empty()) byAddr[addr]++;
    }
    std::vector<uint32_t> out(cands.size());
    for (size_t i = 0; i < cands.size(); ++i) {
        const std::string& addr = candidateRecord(cands[i], s2, s3).addr;
        out[i] = addr.empty() ? 0 : (byAddr[addr] - 1);
    }
    return out;
}

} // namespace er
