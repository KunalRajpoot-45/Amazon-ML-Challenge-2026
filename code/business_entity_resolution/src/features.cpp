#include "features.hpp"
#include "levenshtein.hpp"
#include "tokens.hpp"
#include <algorithm>
#include <functional>

namespace er {

namespace {

std::pair<uint32_t, uint32_t> intersectAndUnion(const std::vector<uint64_t>& a,
                                                  const std::vector<uint64_t>& b) {
    uint32_t inter = 0;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) { ++inter; ++i; ++j; }
        else if (a[i] < b[j]) ++i;
        else ++j;
    }
    uint32_t uni = static_cast<uint32_t>(a.size() + b.size()) - inter;
    return {inter, uni};
}

double jaccard(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    if (a.empty() && b.empty()) return 0.0;
    auto [inter, uni] = intersectAndUnion(a, b);
    if (uni == 0) return 0.0;
    return static_cast<double>(inter) / static_cast<double>(uni);
}

double lenRatio(size_t a, size_t b) {
    if (a == 0 && b == 0) return 1.0;
    size_t mn = std::min(a, b), mx = std::max(a, b);
    if (mx == 0) return 0.0;
    return static_cast<double>(mn) / static_cast<double>(mx);
}

std::string firstToken(const std::string& core) {
    size_t sp = core.find(' ');
    return sp == std::string::npos ? core : core.substr(0, sp);
}

} // namespace

const char* featureName(int i) {
    static const char* names[kFeatureCount] = {
        "name_token_jaccard", "name_trigram_jaccard", "name_lev_sim",
        "addr_token_jaccard", "addr_numeric_token_jaccard", "addr_trigram_jaccard",
        "addr_lev_sim", "name_len_ratio", "addr_len_ratio",
        "shared_block_keys_norm", "name_first_token_match", "addr_numeric_overlap_norm",
        "weighted_block_score_norm", "addr_primary_num_exact", "addr_collision_norm",
    };
    return (i >= 0 && i < kFeatureCount) ? names[i] : "?";
}

FeatureVec computeFeatures(const Record& a, const Record& b, uint16_t sharedBlockKeys,
                           double weightedBlockScore, uint32_t addrCollisionCount) {
    FeatureVec f{};

    TokenSet nameA = tokenizeCore(a.nameCore);
    TokenSet nameB = tokenizeCore(b.nameCore);
    TokenSet addrA = tokenizeCore(a.addr);
    TokenSet addrB = tokenizeCore(b.addr);

    f[F_NAME_TOKEN_JACCARD] = jaccard(nameA.tokenHashes, nameB.tokenHashes);
    f[F_NAME_TRIGRAM_JACCARD] = jaccard(trigramsOf(a.nameCore), trigramsOf(b.nameCore));
    f[F_NAME_LEV_SIM] = levenshteinSimilarity(a.nameCore, b.nameCore);

    f[F_ADDR_TOKEN_JACCARD] = jaccard(addrA.tokenHashes, addrB.tokenHashes);
    auto [numInter, numUnion] = intersectAndUnion(addrA.numericTokenHashes, addrB.numericTokenHashes);
    f[F_ADDR_NUMERIC_TOKEN_JACCARD] = numUnion == 0 ? 0.0 : static_cast<double>(numInter) / numUnion;
    f[F_ADDR_TRIGRAM_JACCARD] = jaccard(trigramsOf(a.addr), trigramsOf(b.addr));
    f[F_ADDR_LEV_SIM] = levenshteinSimilarity(a.addr, b.addr);

    f[F_NAME_LEN_RATIO] = lenRatio(a.nameCore.size(), b.nameCore.size());
    f[F_ADDR_LEN_RATIO] = lenRatio(a.addr.size(), b.addr.size());

    f[F_SHARED_BLOCK_KEYS_NORM] = static_cast<double>(sharedBlockKeys) / (1.0 + sharedBlockKeys);

    std::string fa = firstToken(a.nameCore);
    std::string fb = firstToken(b.nameCore);
    f[F_NAME_FIRST_TOKEN_MATCH] = (!fa.empty() && fa == fb) ? 1.0 : 0.0;

    f[F_ADDR_NUMERIC_OVERLAP_NORM] = std::min(5u, numInter) / 5.0;

    f[F_WEIGHTED_BLOCK_SCORE_NORM] = weightedBlockScore / (1.0 + weightedBlockScore);

    std::string_view fa_num = firstNumericToken(a.addr);
    std::string_view fb_num = firstNumericToken(b.addr);
    f[F_ADDR_PRIMARY_NUM_EXACT] = (!fa_num.empty() && fa_num == fb_num) ? 1.0 : 0.0;

    f[F_ADDR_COLLISION_NORM] = std::min(5u, addrCollisionCount) / 5.0;

    return f;
}

} // namespace er
