#include "normalize.hpp"
#include <cctype>
#include <unordered_map>
#include <unordered_set>

namespace er {

namespace {

// ASCII lowercase + best-effort Latin-1 Supplement case fold (covers French
// accented uppercase letters like 'É' -> 'é'; U+00C0-U+00DE encode as UTF-8
// bytes 0xC3 0x80-0x9E, and the lowercase block U+00E0-U+00FE is a fixed
// +0x20 offset on the second byte, mirroring plain ASCII case folding).
std::string asciiFold(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(in[i]);
        if (c < 0x80) {
            out.push_back(static_cast<char>(std::tolower(c)));
        } else if (c == 0xC3 && i + 1 < in.size()) {
            unsigned char c2 = static_cast<unsigned char>(in[i + 1]);
            if (c2 >= 0x80 && c2 <= 0x9E && c2 != 0x97) c2 += 0x20;
            out.push_back(static_cast<char>(c));
            out.push_back(static_cast<char>(c2));
            ++i;
        } else {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

bool isAsciiAlnum(unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z');
}

// Treat non-ASCII UTF-8 bytes as "word" characters too, so tokens in other
// scripts (Devanagari, Telugu, Bengali, ...) stay intact instead of being
// shredded at punctuation-like continuation bytes.
bool isWordByte(unsigned char c) { return isAsciiAlnum(c) || c >= 0x80; }

const std::unordered_map<std::string, std::string>& nameAbbrevMap() {
    static const std::unordered_map<std::string, std::string> m = {
        {"ltd", "limited"}, {"pvt", "private"}, {"corp", "corporation"},
        {"inc", "incorporated"}, {"co", "company"}, {"mfg", "manufacturing"},
        {"mgmt", "management"}, {"intl", "international"}, {"natl", "national"},
        {"assoc", "association"}, {"assn", "association"}, {"dept", "department"},
        {"svc", "service"}, {"svcs", "services"}, {"dist", "distribution"},
        {"grp", "group"}, {"bros", "brothers"}, {"mfrs", "manufacturers"},
        {"tech", "technology"}, {"sys", "systems"}, {"ent", "enterprises"},
    };
    return m;
}

const std::unordered_map<std::string, std::string>& addressAbbrevMap() {
    static const std::unordered_map<std::string, std::string> m = {
        {"rd", "road"}, {"st", "street"}, {"ave", "avenue"}, {"blvd", "boulevard"},
        {"dr", "drive"}, {"ln", "lane"}, {"ct", "court"}, {"cir", "circle"},
        {"pl", "place"}, {"sq", "square"}, {"ste", "suite"}, {"fl", "floor"},
        {"flr", "floor"}, {"apt", "apartment"}, {"bldg", "building"},
        {"hwy", "highway"}, {"twp", "township"}, {"no", "number"},
        {"opp", "opposite"}, {"nr", "near"}, {"blk", "block"}, {"jn", "junction"},
        {"rly", "railway"}, {"stn", "station"}, {"mkt", "market"},
        {"ind", "industrial"}, {"est", "estate"}, {"ph", "phase"},
        {"sec", "sector"}, {"ext", "extension"}, {"nr.", "near"},
        // US state codes
        {"al", "alabama"}, {"ak", "alaska"}, {"az", "arizona"}, {"ar", "arkansas"},
        {"ca", "california"}, {"colo", "colorado"}, {"ct.", "connecticut"},
        {"de", "delaware"}, {"fl.", "florida"}, {"ga", "georgia"}, {"hi", "hawaii"},
        {"id", "idaho"}, {"il", "illinois"}, {"ia", "iowa"}, {"ks", "kansas"},
        {"ky", "kentucky"}, {"la", "louisiana"}, {"me", "maine"}, {"md", "maryland"},
        {"ma", "massachusetts"}, {"mi", "michigan"}, {"mn", "minnesota"},
        {"ms", "mississippi"}, {"mo", "missouri"}, {"mt", "montana"},
        {"ne", "nebraska"}, {"nv", "nevada"}, {"nh", "newhampshire"},
        {"nj", "newjersey"}, {"nm", "newmexico"}, {"ny", "newyork"},
        {"nc", "northcarolina"}, {"nd", "northdakota"}, {"oh", "ohio"},
        {"ok", "oklahoma"}, {"or", "oregon"}, {"pa", "pennsylvania"},
        {"ri", "rhodeisland"}, {"sc", "southcarolina"}, {"sd", "southdakota"},
        {"tn", "tennessee"}, {"tx", "texas"}, {"ut", "utah"}, {"vt", "vermont"},
        {"va", "virginia"}, {"wa", "washington"}, {"wv", "westvirginia"},
        {"wi", "wisconsin"}, {"wy", "wyoming"}, {"dc", "districtofcolumbia"},
        // India state/UT codes
        {"mh", "maharashtra"}, {"tg", "telangana"}, {"ts", "telangana"},
        {"gj", "gujarat"}, {"wb", "westbengal"}, {"dl", "delhi"},
        {"up", "uttarpradesh"}, {"mp", "madhyapradesh"}, {"rj", "rajasthan"},
        {"tn.", "tamilnadu"}, {"ka", "karnataka"}, {"kl", "kerala"},
        {"ap", "andhrapradesh"}, {"br", "bihar"}, {"pb", "punjab"},
        {"hr", "haryana"}, {"jh", "jharkhand"}, {"cg", "chhattisgarh"},
        {"od", "odisha"}, {"or.", "odisha"}, {"as", "assam"}, {"jk", "jammukashmir"},
        {"uk", "uttarakhand"}, {"ut.", "uttarakhand"}, {"hp", "himachalpradesh"},
        {"ga.", "goa"}, {"py", "puducherry"},
    };
    return m;
}

const std::unordered_set<std::string>& legalNoiseWords() {
    static const std::unordered_set<std::string> s = {
        "private", "limited", "ltd", "pvt", "inc", "incorporated", "corp",
        "corporation", "llc", "llp", "co", "company", "group", "holdings",
        "enterprises", "enterprise", "services", "service", "and", "the",
        "of", "plc",
    };
    return s;
}

} // namespace

std::string cleanField(const std::string& raw, FieldKind kind) {
    std::string folded = asciiFold(raw);
    const auto& abbrev = (kind == FieldKind::Address) ? addressAbbrevMap() : nameAbbrevMap();
    const bool stripLegal = (kind == FieldKind::NameCore);

    std::string core;
    core.reserve(folded.size());

    size_t i = 0;
    while (i < folded.size()) {
        unsigned char c = static_cast<unsigned char>(folded[i]);
        if (!isWordByte(c)) { ++i; continue; }
        size_t j = i;
        while (j < folded.size() && isWordByte(static_cast<unsigned char>(folded[j]))) ++j;
        std::string tok = folded.substr(i, j - i);
        i = j;

        auto it = abbrev.find(tok);
        if (it != abbrev.end()) tok = it->second;

        if (stripLegal && legalNoiseWords().count(tok)) continue;

        if (!core.empty()) core.push_back(' ');
        core += tok;
    }

    return core;
}

} // namespace er
