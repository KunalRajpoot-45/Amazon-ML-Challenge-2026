#pragma once
#include <string>

namespace er {

enum class FieldKind {
    NameCore,  // business_name: abbreviations expanded, generic legal-entity words dropped
    Address    // business_address: abbreviations expanded (street/state terms)
};

// Lowercases ASCII, best-effort case-folds Latin-1 Supplement accented
// letters (covers French/Spanish/German diacritics in the test set), expands
// common abbreviations for the given field kind, and returns a cleaned,
// single-space-joined token string. Non-ASCII scripts (Hindi, Telugu,
// Bengali, ...) are kept as intact word tokens rather than shredded.
//
// Only this compact string is kept per record in memory; token hashes and
// character trigrams are derived from it on demand (see tokens.hpp) only for
// the much smaller set of records that actually become candidates, instead
// of being precomputed and stored for all ~12M records up front.
std::string cleanField(const std::string& raw, FieldKind kind);

} // namespace er
