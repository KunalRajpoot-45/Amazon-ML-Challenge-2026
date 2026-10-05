#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "record.hpp"

namespace er {

// Compressed (CSR-style) inverted index: token hash -> list of record
// indices. Built in two passes to avoid per-key vector reallocation, which
// matters at multi-million-record / tens-of-millions-of-postings scale.
class InvertedIndex {
public:
    template <typename TokenGetter>
    void build(const std::vector<Record>& recs,
               const std::vector<uint32_t>& subsetIdx,
               TokenGetter getTokens) {
        std::unordered_map<uint64_t, uint32_t> counts;
        for (uint32_t idx : subsetIdx) {
            for (uint64_t h : getTokens(recs[idx])) counts[h]++;
        }

        offsetLen_.reserve(counts.size());
        uint32_t cum = 0;
        for (const auto& kv : counts) {
            offsetLen_[kv.first] = {cum, kv.second};
            cum += kv.second;
        }
        postings_.resize(cum);

        std::unordered_map<uint64_t, uint32_t> cursor;
        cursor.reserve(counts.size());
        for (const auto& kv : offsetLen_) cursor[kv.first] = kv.second.first;

        for (uint32_t idx : subsetIdx) {
            for (uint64_t h : getTokens(recs[idx])) {
                postings_[cursor[h]++] = idx;
            }
        }
    }

    // Returns {pointer, length}; pointer is null / length 0 if the token is
    // unseen. Caller is expected to skip tokens whose df() exceeds a
    // max-block-size threshold before calling postings().
    std::pair<const uint32_t*, uint32_t> postings(uint64_t tokenHash) const {
        auto it = offsetLen_.find(tokenHash);
        if (it == offsetLen_.end()) return {nullptr, 0};
        return {postings_.data() + it->second.first, it->second.second};
    }

    uint32_t df(uint64_t tokenHash) const {
        auto it = offsetLen_.find(tokenHash);
        return it == offsetLen_.end() ? 0 : it->second.second;
    }

private:
    std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> offsetLen_;
    std::vector<uint32_t> postings_;
};

} // namespace er
