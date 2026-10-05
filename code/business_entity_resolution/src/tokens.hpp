#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include "hashing.hpp"

namespace er {

// Derives token hashes from an already-cleaned, single-space-joined core
// string (see normalize.hpp). Cheap enough to call on demand per candidate
// pair / per index-build pass instead of storing per record, since it's
// just a split + hash + sort, not re-tokenization from raw text.
struct TokenSet {
    std::vector<uint64_t> tokenHashes;        // sorted, unique
    std::vector<uint64_t> numericTokenHashes; // sorted, unique subset containing a digit
};

inline TokenSet tokenizeCore(const std::string& core) {
    TokenSet ts;
    size_t i = 0;
    while (i < core.size()) {
        size_t j = core.find(' ', i);
        if (j == std::string::npos) j = core.size();
        if (j > i) {
            std::string_view tok(core.data() + i, j - i);
            uint64_t h = fnv1a64(tok);
            ts.tokenHashes.push_back(h);
            for (char c : tok) {
                if (c >= '0' && c <= '9') { ts.numericTokenHashes.push_back(h); break; }
            }
        }
        i = j + 1;
    }
    std::sort(ts.tokenHashes.begin(), ts.tokenHashes.end());
    ts.tokenHashes.erase(std::unique(ts.tokenHashes.begin(), ts.tokenHashes.end()), ts.tokenHashes.end());
    std::sort(ts.numericTokenHashes.begin(), ts.numericTokenHashes.end());
    ts.numericTokenHashes.erase(std::unique(ts.numericTokenHashes.begin(), ts.numericTokenHashes.end()),
                                 ts.numericTokenHashes.end());
    return ts;
}

namespace detail {
inline std::vector<std::string_view> splitTokens(const std::string& core) {
    std::vector<std::string_view> toks;
    size_t i = 0;
    while (i < core.size()) {
        size_t j = core.find(' ', i);
        if (j == std::string::npos) j = core.size();
        if (j > i) toks.emplace_back(core.data() + i, j - i);
        i = j + 1;
    }
    return toks;
}

inline std::vector<uint64_t> bigramsFrom(const std::vector<std::string_view>& toks) {
    std::vector<uint64_t> bg;
    if (toks.size() < 2) return bg;
    bg.reserve(toks.size() - 1);
    std::string buf;
    for (size_t k = 0; k + 1 < toks.size(); ++k) {
        buf.clear();
        buf.append(toks[k]);
        buf.push_back('_');
        buf.append(toks[k + 1]);
        bg.push_back(fnv1a64(buf));
    }
    std::sort(bg.begin(), bg.end());
    bg.erase(std::unique(bg.begin(), bg.end()), bg.end());
    return bg;
}
} // namespace detail

// Adjacent-pair keys over the ORDERED subsequence of digit-containing
// tokens (non-numeric tokens are skipped when deciding adjacency, so a
// prefix/suffix like "D/" or a missing PIN code doesn't break the pairing).
// A lone numeric token like "11" or "6" is common enough to be nearly
// useless as a blocking key on its own, but the pair "11_23" is far more
// specific -- this recovers house/plot-number matches that individual
// numeric-token blocking loses to document-frequency capping.
inline std::vector<uint64_t> numericBigramHashes(const std::string& core) {
    std::vector<std::string_view> numTok;
    for (auto tok : detail::splitTokens(core)) {
        for (char c : tok) {
            if (c >= '0' && c <= '9') { numTok.push_back(tok); break; }
        }
    }
    return detail::bigramsFrom(numTok);
}

// Adjacent-pair keys over ALL tokens in original order (no filtering).
// Individually-common words ("global", "trust", "bangalore") are often
// excluded by document-frequency capping even after relative scaling, but
// the two-word phrase ("global_trust", "bangalore_karnataka") is far rarer
// and survives. Used for both name and address; more insertion-sensitive
// than numericBigramHashes since nothing is skipped when finding
// adjacency, so it's a complementary signal, not a replacement.
inline std::vector<uint64_t> allTokenBigramHashes(const std::string& core) {
    return detail::bigramsFrom(detail::splitTokens(core));
}

// First digit-containing token in the address, e.g. the house/plot number.
// Empty if the address has no numeric token. Used to penalize near-miss
// addresses (same street, off-by-a-few house number) that token/trigram
// Jaccard alone doesn't punish enough when the rest of a long address
// string matches.
inline std::string_view firstNumericToken(const std::string& core) {
    for (auto tok : detail::splitTokens(core)) {
        for (char c : tok) {
            if (c >= '0' && c <= '9') return tok;
        }
    }
    return std::string_view();
}

inline std::vector<uint64_t> trigramsOf(const std::string& core) {
    std::vector<uint64_t> tri;
    if (core.size() < 3) {
        if (!core.empty()) tri.push_back(fnv1a64(core));
        return tri;
    }
    tri.reserve(core.size());
    for (size_t k = 0; k + 3 <= core.size(); ++k) {
        tri.push_back(fnv1a64(std::string_view(core.data() + k, 3)));
    }
    std::sort(tri.begin(), tri.end());
    tri.erase(std::unique(tri.begin(), tri.end()), tri.end());
    return tri;
}

} // namespace er
