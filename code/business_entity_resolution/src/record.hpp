#pragma once
#include <string>

namespace er {

// One business record from any of the three sources. Only compact cleaned
// text is kept per record (not raw text, not precomputed token/trigram hash
// vectors) to keep memory bounded across ~12M records; token/trigram sets
// are derived on demand from `nameCore`/`addr` (see tokens.hpp) only for the
// much smaller set of records that actually become candidates.
struct Record {
    std::string id;
    std::string country;
    std::string nameCore; // cleaned business_name, legal-suffix noise stripped
    std::string addr;     // cleaned business_address
};

} // namespace er
