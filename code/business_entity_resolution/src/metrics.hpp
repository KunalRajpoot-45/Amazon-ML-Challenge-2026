#pragma once
#include <string>
#include <unordered_set>
#include <vector>

namespace er {

// Per-entity F_0.5 exactly as specified: F_0.5 = 1.25*P*R / (0.25*P + R).
// Singletons (empty predicted and empty truth) score 1.0.
inline double f0_5ForEntity(const std::vector<std::string>& predicted,
                             const std::vector<std::string>& truth) {
    std::unordered_set<std::string> predSet(predicted.begin(), predicted.end());
    std::unordered_set<std::string> truthSet(truth.begin(), truth.end());
    if (predSet.empty() && truthSet.empty()) return 1.0;

    int tp = 0;
    for (const auto& p : predSet) {
        if (truthSet.count(p)) ++tp;
    }
    double precision = predSet.empty() ? 0.0 : static_cast<double>(tp) / predSet.size();
    double recall = truthSet.empty() ? 0.0 : static_cast<double>(tp) / truthSet.size();
    if (precision == 0.0 && recall == 0.0) return 0.0;
    return (1.25 * precision * recall) / (0.25 * precision + recall);
}

} // namespace er
