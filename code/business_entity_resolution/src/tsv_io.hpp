#pragma once
#include <string>
#include <vector>
#include "record.hpp"

namespace er {

// Loads a *_source{1,2,3}.tsv file (columns: entity_id, business_name,
// business_address, country) and computes normalization for every record,
// using all available cores.
std::vector<Record> loadSource(const std::string& path);

struct GroundTruthEntry {
    std::string source1Id;
    std::vector<std::string> matchedIds;
};

// Loads train_ground_truth.tsv (source1_entity_id, comma-separated matched_entity_ids).
std::vector<GroundTruthEntry> loadGroundTruth(const std::string& path);

// Writes a two-column ID-list TSV: source1_entity_id \t comma,separated,ids
// Used for both matching_results.tsv and candidate_pairs.tsv (second column
// header differs).
void writeIdListTsv(const std::string& path,
                     const std::string& secondColumnHeader,
                     const std::vector<std::string>& source1Ids,
                     const std::vector<std::vector<std::string>>& idLists);

} // namespace er
