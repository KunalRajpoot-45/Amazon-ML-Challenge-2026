#pragma once
#include <cstdint>
#include <string>
#include <string_view>

namespace er {

// FNV-1a 64-bit hash. Used to turn normalized tokens / n-grams into compact
// integer keys for the inverted index and Jaccard/set-similarity features,
// avoiding repeated string hashing/compares deeper in the pipeline.
inline uint64_t fnv1a64(std::string_view s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

} // namespace er
