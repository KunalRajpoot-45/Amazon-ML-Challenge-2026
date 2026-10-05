#pragma once
#include <array>
#include <cstdint>
#include "record.hpp"

namespace er {

constexpr int kFeatureCount = 15;

enum FeatureIndex {
    F_NAME_TOKEN_JACCARD = 0,
    F_NAME_TRIGRAM_JACCARD,
    F_NAME_LEV_SIM,
    F_ADDR_TOKEN_JACCARD,
    F_ADDR_NUMERIC_TOKEN_JACCARD,
    F_ADDR_TRIGRAM_JACCARD,
    F_ADDR_LEV_SIM,
    F_NAME_LEN_RATIO,
    F_ADDR_LEN_RATIO,
    F_SHARED_BLOCK_KEYS_NORM,
    F_NAME_FIRST_TOKEN_MATCH,
    F_ADDR_NUMERIC_OVERLAP_NORM,
    F_WEIGHTED_BLOCK_SCORE_NORM,
    F_ADDR_PRIMARY_NUM_EXACT,
    F_ADDR_COLLISION_NORM,
};

using FeatureVec = std::array<double, kFeatureCount>;

const char* featureName(int i);

// Derives token/trigram sets from a.nameCore/a.addr and b.nameCore/b.addr on
// demand (see tokens.hpp) -- cheap since it only runs per candidate pair,
// not per record. sharedBlockKeys/weightedBlockScore come from the Blocker's
// Candidate (raw count and IDF-weighted score of shared blocking keys).
// addrCollisionCount is how many OTHER candidates in this same entity's
// candidate set share b's exact address -- a proxy for "this address hosts
// multiple distinct businesses" (shared buildings/plazas), where address
// similarity alone is a weaker signal and name similarity should matter
// more; 0 if the caller doesn't track this (e.g. not yet grouped).
FeatureVec computeFeatures(const Record& a, const Record& b, uint16_t sharedBlockKeys,
                           double weightedBlockScore, uint32_t addrCollisionCount = 0);

} // namespace er
