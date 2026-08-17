#include "ml_lib/core/tokenizer.h"
#include <algorithm>
#include <cctype>

Tokenizer::Tokenizer()
    : tokenizer_type(TOKENIZER_TYPE::WORD) {}

void Tokenizer::setType(TOKENIZER_TYPE type) {
    tokenizer_type = type;
}

std::vector<std::string> Tokenizer::wordTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string new_word = "";
    for (size_t i = 0; i < text.size(); ++i) {
        if (std::isspace(text[i])) {
            if (!new_word.empty()) {
                tokens.push_back(new_word);
                new_word = "";
            }
            continue;
        }
        new_word += text[i];
    }
    if (!new_word.empty()) {
        tokens.push_back(new_word);
    }
    return tokens;
}

std::vector<std::string> Tokenizer::characterTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    for (size_t i = 0; i < text.size(); ++i) {
        tokens.push_back(std::string(1, text[i]));
    }
    return tokens;
}

std::vector<std::string> Tokenizer::sentenceTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string sentence = "";
    for (size_t i = 0; i < text.size(); ++i) {
        sentence += text[i];
        if (text[i] == '.' || text[i] == '!' || text[i] == '?') {
            tokens.push_back(sentence);
            sentence = "";
        }
    }
    if (!sentence.empty()) {
        tokens.push_back(sentence);
    }
    return tokens;
}

void Tokenizer::trainBPE(const std::vector<std::string>& corpus, size_t num_merges) {

    // word frequencies
    std::unordered_map<std::string, int> word_freqs;

    for (const std::string& text : corpus) {
        std::string word = "";
        for (const char c : text) {
            if (std::isspace(c)) {
                if (!word.empty()) {
                    word += "</w>";
                    word_freqs[word]++;
                    word = "";
                }
            } else {
                if (!word.empty()) word += " ";
                word += c;
            }
        }
        if (!word.empty()) {
            word += " </w>";
            word_freqs[word]++;
        }
    }

    // BPE merges
    for (size_t merge_idx = 0; merge_idx < num_merges; ++merge_idx) {
        std::unordered_map<std::pair<std::string, std::string>, int, PairHash> pair_counts;

        // count pairs
        for (const auto &[word, freq] : word_freqs) {
            std::vector<std::string> symbols;
            std::string current = "";
            for (const char c : word) {
                if (c == ' ') {
                    if (!current.empty()) {
                        symbols.push_back(current);
                        current = "";
                    }
                } else {
                    current += c;
                }
            }
            if (!current.empty()) symbols.push_back(current);

            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                pair_counts[{symbols[i], symbols[i + 1]}] += freq;
            }
        }

        if (pair_counts.empty()) break;

        // find best pair
        auto best_pair = std::max_element(
            pair_counts.begin(), pair_counts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        );

        if (best_pair->second < 2) break;

        // merge best pair
        auto [first, second] = best_pair->first;
        std::string merged = first + second;

        bpe_merges.push_back({first, second});

        std::unordered_map<std::string, int> new_word_freqs;
        std::string pattern = first + " " + second;

        for (const auto& [word, freq] : word_freqs) {
            std::string new_word = word;
            size_t pos;
            while ((pos = new_word.find(pattern)) != std::string::npos) {
                new_word.replace(pos, pattern.length(), merged);
            }
            new_word_freqs[new_word] = freq;
        }
        word_freqs = std::move(new_word_freqs);
    }

    // build vocab
    int id = 0;
    for (const auto &[word, _] : word_freqs) {
        std::string current = "";
        for (const char c : word) {
            if (c == ' ') {
                if (!current.empty() && vocab.find(current) == vocab.end()) {
                    vocab[current] = id++;
                }
                current = "";
            } else {
                current += c;
            }
        }
        if (!current.empty() && vocab.find(current) == vocab.end()) {
            vocab[current] = id++;
        }
    }

    buildReverseVocab();
}

std::vector<std::string> Tokenizer::applyBPEMerges(const std::string& word) {
    if (word.empty()) return {};

    std::vector<std::string> tokens;
    for (char c : word) {
        tokens.push_back(std::string(1, c));
    }
    tokens.back() += "</w>"; 

    for (const auto &[first, second] : bpe_merges) {
        std::vector<std::string> new_tokens;
        size_t i = 0;
        while (i < tokens.size()) {
            if (i + 1 < tokens.size() && tokens[i] == first && tokens[i + 1] == second) {
                new_tokens.push_back(first + second);
                i += 2;
            } else {
                new_tokens.push_back(tokens[i]);
                i++;
            }
        }
        tokens = std::move(new_tokens);
        if (tokens.size() == 1) break;
    }

    return tokens;
}

static std::string byteToBpeChar(unsigned char b) {
    static const bool init = []() { return true; }();
    (void)init;

    bool is_direct =
        (b >= 33 && b <= 126) ||
        (b >= 161 && b <= 172) ||
        (b >= 174);

    if (is_direct) {
        return std::string(1, static_cast<char>(b));
    }

    static const unsigned char remap_bytes[] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
        127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,
        150,151,152,153,154,155,156,157,158,159,160,173
    };
    for (int i = 0; i < 68; i++) {
        if (remap_bytes[i] == b) {
            int codepoint = 256 + i;
            char buf[4];
            buf[0] = static_cast<char>(0xC0 | (codepoint >> 6));
            buf[1] = static_cast<char>(0x80 | (codepoint & 0x3F));
            return std::string(buf, 2);
        }
    }
    return std::string(1, static_cast<char>(b));
}

static std::vector<std::string> splitUtf8Chars(const std::string& s) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < s.size(); ) {
        auto b = static_cast<unsigned char>(s[i]);
        int len = 1;
        if (b >= 0xF0) len = 4;
        else if (b >= 0xE0) len = 3;
        else if (b >= 0xC0) len = 2;
        if (i + len > s.size()) len = 1;
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

std::vector<std::string> Tokenizer::bpeTokenize(const std::string& text) {
    std::vector<std::string> tokens;

    if (bpe_merges.empty()) {
        return characterTokenize(text);
    }

    bool byte_level = false;
    for (const auto& [tok, id] : vocab) {
        if (tok.size() >= 2) {
            auto b0 = static_cast<unsigned char>(tok[0]);
            auto b1 = static_cast<unsigned char>(tok[1]);
            if (b0 == 0xC4 && b1 == 0xA0) {
                byte_level = true;
                break;
            }
        }
    }

    if (!byte_level) {
        std::string word;
        for (size_t i = 0; i < text.size(); ++i) {
            if (std::isspace(text[i])) {
                if (!word.empty()) {
                    auto word_tokens = applyBPEMerges(word);
                    tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
                    word = "";
                }
            } else {
                word += text[i];
            }
        }
        if (!word.empty()) {
            auto word_tokens = applyBPEMerges(word);
            tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
        }
        return tokens;
    }

    std::string encoded;
    for (unsigned char b : text) {
        encoded += byteToBpeChar(b);
    }

    std::string space_marker = byteToBpeChar(' ');
    std::vector<std::string> words;
    std::string current;
    for (size_t i = 0; i < encoded.size(); ) {
        auto b = static_cast<unsigned char>(encoded[i]);
        int len = 1;
        if (b >= 0xF0) len = 4;
        else if (b >= 0xE0) len = 3;
        else if (b >= 0xC0) len = 2;
        if (i + len > encoded.size()) len = 1;

        std::string ch = encoded.substr(i, len);
        if (ch == space_marker && !current.empty()) {
            words.push_back(current);
            current = ch;
        } else {
            current += ch;
        }
        i += len;
    }
    if (!current.empty()) words.push_back(current);

    for (const auto& word : words) {
        auto chars = splitUtf8Chars(word);

        for (const auto& [first, second] : bpe_merges) {
            std::vector<std::string> new_chars;
            size_t i = 0;
            while (i < chars.size()) {
                if (i + 1 < chars.size() && chars[i] == first && chars[i + 1] == second) {
                    new_chars.push_back(first + second);
                    i += 2;
                } else {
                    new_chars.push_back(chars[i]);
                    i++;
                }
            }
            chars = std::move(new_chars);
            if (chars.size() == 1) break;
        }

        tokens.insert(tokens.end(), chars.begin(), chars.end());
    }

    return tokens;
}

std::vector<std::string> Tokenizer::tokenize(const std::string& text) {
    switch (tokenizer_type) {
        case TOKENIZER_TYPE::WORD:
            return wordTokenize(text);
        case TOKENIZER_TYPE::CHARACTER:
            return characterTokenize(text);
        case TOKENIZER_TYPE::BPE:
            return bpeTokenize(text);
        case TOKENIZER_TYPE::SENTENCE:
            return sentenceTokenize(text);
        default:
            return {};
    }
}

void Tokenizer::buildReverseVocab() {
    reverse_vocab.clear();
    for (const auto& [token, id] : vocab) {
        reverse_vocab[id] = token;
    }
}

void Tokenizer::buildVocab(const std::vector<std::string>& corpus) {
    vocab.clear();
    int id = 0;

    for (const std::string& text : corpus) {
        auto tokens = tokenize(text);
        for (const std::string& token : tokens) {
            if (vocab.find(token) == vocab.end()) {
                vocab[token] = id++;
            }
        }
    }

    buildReverseVocab();
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> ids;

    if (!special_tokens.empty()) {
        std::vector<std::string> segments = {text};
        std::vector<bool> is_special = {false};

        for (const auto& st : special_tokens) {
            std::vector<std::string> new_segments;
            std::vector<bool> new_is_special;
            for (size_t i = 0; i < segments.size(); i++) {
                if (is_special[i]) {
                    new_segments.push_back(segments[i]);
                    new_is_special.push_back(true);
                    continue;
                }
                const std::string& seg = segments[i];
                size_t pos = 0;
                while (pos < seg.size()) {
                    size_t found = seg.find(st, pos);
                    if (found == std::string::npos) {
                        if (pos < seg.size()) {
                            new_segments.push_back(seg.substr(pos));
                            new_is_special.push_back(false);
                        }
                        break;
                    }
                    if (found > pos) {
                        new_segments.push_back(seg.substr(pos, found - pos));
                        new_is_special.push_back(false);
                    }
                    new_segments.push_back(st);
                    new_is_special.push_back(true);
                    pos = found + st.size();
                }
            }
            segments = std::move(new_segments);
            is_special = std::move(new_is_special);
        }

        for (size_t i = 0; i < segments.size(); i++) {
            if (is_special[i]) {
                auto it = vocab.find(segments[i]);
                if (it != vocab.end()) ids.push_back(it->second);
            } else {
                auto tokens = tokenize(segments[i]);
                for (const auto& token : tokens) {
                    auto it = vocab.find(token);
                    if (it != vocab.end()) ids.push_back(it->second);
                }
            }
        }
    } else {
        auto tokens = tokenize(text);
        for (const auto& token : tokens) {
            auto it = vocab.find(token);
            if (it != vocab.end()) ids.push_back(it->second);
        }
    }

    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) {
    std::string result;

    for (size_t i = 0; i < ids.size(); ++i) {
        auto it = reverse_vocab.find(ids[i]);
        if (it == reverse_vocab.end()) continue;

        const std::string& token = it->second;

        switch (tokenizer_type) {
            case TOKENIZER_TYPE::WORD:
            case TOKENIZER_TYPE::SENTENCE:
                if (i > 0) result += " ";
                result += token;
                break;
            case TOKENIZER_TYPE::CHARACTER:
                result += token;
                break;
            case TOKENIZER_TYPE::BPE: {
                std::string clean = token;
                size_t pos = clean.find("</w>");
                if (pos != std::string::npos) {
                    clean.replace(pos, 4, " ");
                }
                std::string decoded;
                for (size_t c = 0; c < clean.size(); ) {
                    auto b0 = static_cast<unsigned char>(clean[c]);
                    if (b0 == 0xC4 && c + 1 < clean.size()) {
                        auto b1 = static_cast<unsigned char>(clean[c + 1]);
                        if (b1 >= 0x80 && b1 <= 0xBF) {
                            decoded += static_cast<char>((b1 - 0x80) + 0x00);
                            c += 2;
                            continue;
                        }
                    }
                    if (b0 == 0xC5 && c + 1 < clean.size()) {
                        auto b1 = static_cast<unsigned char>(clean[c + 1]);
                        if (b1 >= 0x80 && b1 <= 0xBF) {
                            decoded += static_cast<char>((b1 - 0x80) + 0x40);
                            c += 2;
                            continue;
                        }
                    }
                    decoded += clean[c];
                    c++;
                }
                result += decoded;
                break;
            }
        }
    }

    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }

    return result;
}
