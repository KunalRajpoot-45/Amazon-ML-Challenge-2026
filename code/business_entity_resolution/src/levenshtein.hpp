#pragma once
#include <algorithm>
#include <string>
#include <vector>

namespace er {

// Normalized similarity in [0,1]: 1 - edit_distance / max(len_a, len_b).
// Strings are capped to `maxLen` bytes (from the front) to bound worst-case
// cost across tens of millions of candidate pairs; this only affects a small
// tail of unusually long names/addresses and still yields a meaningful
// approximate similarity for them.
inline double levenshteinSimilarity(const std::string& a, const std::string& b, size_t maxLen = 80) {
    size_t na = std::min(a.size(), maxLen);
    size_t nb = std::min(b.size(), maxLen);
    if (na == 0 && nb == 0) return 1.0;
    if (na == 0 || nb == 0) return 0.0;

    std::vector<int> prev(nb + 1), cur(nb + 1);
    for (size_t j = 0; j <= nb; ++j) prev[j] = static_cast<int>(j);

    for (size_t i = 1; i <= na; ++i) {
        cur[0] = static_cast<int>(i);
        char ca = a[i - 1];
        for (size_t j = 1; j <= nb; ++j) {
            int cost = (ca == b[j - 1]) ? 0 : 1;
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost});
        }
        std::swap(prev, cur);
    }
    int dist = prev[nb];
    double denom = static_cast<double>(std::max(na, nb));
    return 1.0 - static_cast<double>(dist) / denom;
}

} // namespace er
