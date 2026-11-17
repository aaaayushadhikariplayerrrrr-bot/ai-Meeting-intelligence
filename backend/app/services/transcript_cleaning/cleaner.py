"""Multilingual Transcript Cleaning Service.

Improves transcript readability while strictly preserving:
- Speaker's intended meaning
- Original language and code-switching (no translation)
- Meaningful lexical reduplications and rhetorical emphasis (Hindi, Gujarati, English, Hinglish)
- Technical terms, numbers, abbreviations, dates, and named entities
- Imperfect speech without inventing words or summarizing
"""

import re
from typing import Tuple, List, Optional, Set
from pydantic import BaseModel, Field


class CleaningResult(BaseModel):
    """Result of cleaning a transcript segment."""
    cleaned_text: str
    original_text: str
    transformations_applied: List[str] = Field(default_factory=list)
    has_changed: bool = False


# Meaningful lexicalized reduplications that must NOT be collapsed as accidental stutters
MEANINGFUL_REDUPLICATIONS: Set[str] = {
    # English
    "very very",
    "bye bye",
    "really really",
    "no no",
    "no no no",
    "hear hear",
    "not not",
    "never never",
    "that that",
    "had had",
    # Hindi / Hinglish in Devanagari
    "धीरे धीरे",
    "धीरे-धीरे",
    "बार बार",
    "बार-बार",
    "साथ साथ",
    "साथ-साथ",
    "अलग अलग",
    "अलग-अलग",
    "पास पास",
    "पास-पास",
    "चलते चलते",
    "चलते-चलते",
    "थोड़ा थोड़ा",
    "थोड़ा-थोड़ा",
    "कभी कभी",
    "कभी-कभी",
    "जल्दी जल्दी",
    "जल्दी-जल्दी",
    "छोटे छोटे",
    "छोटे-छोटे",
    "बड़े बड़े",
    "बड़े-बड़े",
    "गरम गरम",
    "गरम-गरम",
    "राम राम",
    "नमस्ते नमस्ते",
    "नहीं नहीं",
    "नहीं-नहीं",
    "नही नही",
    "नही-नही",
    "ना ना",
    "ना-ना",
    "मत मत",
    "मत-मत",
    # Hinglish in Latin script
    "dhire dhire",
    "dheere dheere",
    "jaldi jaldi",
    "kabhi kabhi",
    "saath saath",
    "alag alag",
    "thoda thoda",
    "baar baar",
    "pass pass",
    "chote chote",
    "bade bade",
    "kam se kam",
    "nahi nahi",
    "nahin nahin",
    "na na",
    # Gujarati
    "ધીમે ધીમે",
    "ધીમે-ધીમે",
    "વારંવાર",
    "વારં-વાર",
    "સાથે સાથે",
    "સાથે-સાથે",
    "જુદા જુદા",
    "જુદા-જુદા",
    "નજીક નજીક",
    "નજીક-નજીક",
    "થોડું થોડું",
    "થોડું-થોડું",
    "મોટા મોટા",
    "મોટા-મોટા",
    "નાના નાના",
    "નાના-નાના",
    "જલ્દી જલ્દી",
    "જલ્દી-જલ્દી",
    "જય જય",
    "નહીં નહીં",
    "નહીં-નહીં",
    "નહિ નહિ",
    "નહિ-નહિ",
    "ના ના",
    "ના-ના",
}

# Legitimate hyphenated prefixes and compound words that must NOT be collapsed as speech stutters
LEGITIMATE_HYPHENATED_PREFIXES: Set[str] = {
    "co", "re", "de", "pre", "sub", "pro", "ex", "non", "anti", "semi",
    "multi", "un", "self", "post", "inter", "intra", "cross", "quasi",
    "उप", "प्रति", "पुन", "पुनः", "अधि", "सह", "अनु", "गैर", "स्व",
    "ઉપ", "પ્રતિ", "પુન", "પુનઃ", "સહ", "બિન", "સ્વ"
}

LEGITIMATE_HYPHENATED_TERMS: Set[str] = {
    "co-counsel", "re-review", "de-facto", "pre-existing", "sub-clause",
    "sub-section", "cross-examination", "pro-rata", "ex-parte", "inter-alia",
    "co-conspirator", "co-contractor", "co-chair", "re-register", "re-release",
    "re-record", "उप-उपायुक्त", "उप-उपबंध", "ઉપ-ઉપાયુક્ત", "ઉપ-ઉપબંધ"
}

# Isolated English filler words (pure hesitation disfluencies)
# Note: Context-sensitive words like "like", "well", "so", "actually", "basically", "you know"
# are deliberately omitted from unconditional removal to avoid altering meaning.
ENGLISH_ISOLATED_FILLERS = re.compile(
    r"\b(?:uh|um|er|ah|uhm|umm|er|erm)\b",
    re.IGNORECASE
)

# Isolated Hindi / Gujarati hesitation noises (NOT meaningful words like "तो", "मतलब", "हाँ", "હા")
INDIC_ISOLATED_FILLERS = re.compile(
    r"\b(?:अह|उम|अम्म|उह|અહ|ઉમ)\b"
)

# Word-initial stutters (e.g., "th- the", "w- we", "क- क्या", "હ- હવે")
INITIAL_STUTTER_PATTERN = re.compile(
    r"\b([a-zA-Z\u0900-\u097F\u0A80-\u0AFF]{1,2})\s*-\s*(\1[a-zA-Z\u0900-\u097F\u0A80-\u0AFF]+)\b",
    re.IGNORECASE | re.UNICODE
)


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive spaces, tabs, and non-breaking spaces into a single space."""
    # Replace non-breaking spaces and other irregular spaces
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()


def remove_disfluent_fillers(text: str, language_code: str) -> Tuple[str, bool]:
    """Remove clearly disfluent hesitation noises while strictly preserving meaningful vocabulary."""
    original = text
    lang = (language_code or "en").lower()

    if lang in ("en", "hi-en", "hinglish"):
        text = ENGLISH_ISOLATED_FILLERS.sub("", text)

    if lang in ("hi", "gu", "hi-en", "hinglish"):
        text = INDIC_ISOLATED_FILLERS.sub("", text)

    text = normalize_whitespace(text)
    return text, text != original


def reduce_accidental_repetitions(text: str) -> Tuple[str, bool]:
    """Reduce accidental stutters and immediate word duplicates while preserving lexicalized reduplication."""
    original = text

    # 1. Clean hyphenated stutters like "w- we" -> "we", while strictly preserving legitimate compounds
    def _clean_initial_stutter(match: re.Match) -> str:
        prefix = match.group(1).lower()
        full_word = match.group(2)
        full_compound = f"{prefix}-{full_word.lower()}"
        if prefix in LEGITIMATE_HYPHENATED_PREFIXES or full_compound in LEGITIMATE_HYPHENATED_TERMS:
            return match.group(0)
        return full_word

    text = INITIAL_STUTTER_PATTERN.sub(_clean_initial_stutter, text)

    # 2. Clean immediate word repetitions (e.g. "the the meeting" -> "the meeting")
    words = text.split()
    if len(words) < 2:
        return text, text != original

    cleaned_words = []
    i = 0
    while i < len(words):
        current_word = words[i]
        if i + 1 < len(words):
            next_word = words[i + 1]
            pair_lower = f"{current_word.lower()} {next_word.lower()}"

            # Check if current_word ends a sentence (boundary terminator)
            # Never collapse repeated words across sentence boundaries
            has_sentence_end = bool(re.search(r"[.!?।॥][\"')\]]?$", current_word))

            # Clean punctuation from words for exact duplicate check and numeric inspection
            clean_curr = re.sub(r"[^\w\u0900-\u097F\u0A80-\u0AFF]", "", current_word.lower())
            clean_next = re.sub(r"[^\w\u0900-\u097F\u0A80-\u0AFF]", "", next_word.lower())

            # Check if this pair is a recognized meaningful lexicalized reduplication
            is_meaningful = (
                pair_lower in MEANINGFUL_REDUPLICATIONS
                or f"{pair_lower.replace(' ', '-')}" in MEANINGFUL_REDUPLICATIONS
                or f"{clean_curr} {clean_next}" in MEANINGFUL_REDUPLICATIONS
            )

            # Check if tokens are numeric (representing section numbers, dates, quantities, notice periods)
            is_numeric = bool(re.search(r"[\d\u0966-\u096F\u0AE6-\u0AEF]", clean_curr))

            if clean_curr and clean_curr == clean_next and not is_meaningful and not has_sentence_end and not is_numeric:
                # Accidental immediate duplication detected: keep the one with punctuation if present
                chosen = next_word if re.search(r"[,.?!;:]", next_word) else current_word
                cleaned_words.append(chosen)
                i += 2
                continue

        cleaned_words.append(current_word)
        i += 1

    result = " ".join(cleaned_words)
    return result, result != original


def normalize_punctuation_and_capitalization(text: str, language_code: str) -> Tuple[str, bool]:
    """Fix irregular punctuation spacing, double punctuation, and sentence-initial English capitalization."""
    if not text:
        return text, False

    original = text
    lang = (language_code or "en").lower()

    # 1. Strip orphan leading punctuation resulting from filler removal (e.g., ", we need" -> "we need")
    # Preserve leading minus sign if it belongs to a numeric value (e.g. -5%, -10%, -25)
    neg_match = re.match(r"^[,\s;:–—]*(-(?:\s*)[\d\u0966-\u096F\u0AE6-\u0AEF].*)", text)
    if neg_match:
        text = neg_match.group(1)
    else:
        text = re.sub(r"^[,\s;:\-–—]+", "", text)

    # 2. Fix spaced punctuation (e.g., "word , word" -> "word, word")
    text = re.sub(r"\s+([,.?!:;।॥])", r"\1", text)

    # 3. Fix duplicate commas, colons, semicolons
    text = re.sub(r",+", ",", text)
    text = re.sub(r";+", ";", text)
    text = re.sub(r":+", ":", text)

    # 4. Collapse multiple question marks or exclamation marks
    text = re.sub(r"\?+", "?", text)
    text = re.sub(r"!+", "!", text)

    # 5. Collapse multiple periods to single (preserving ellipsis "...")
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)

    # 6. English sentence capitalization
    if lang in ("en", "hi-en", "hinglish"):
        # Capitalize first character of text
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        # Capitalize letters following sentence terminators (. ! ?)
        def _cap_match(match):
            return match.group(1) + match.group(2).upper()

        text = re.sub(r"([.!?]\s+)([a-z])", _cap_match, text)

    # 7. Clean trailing stutter dashes (e.g. "We discussed the-" -> "We discussed the")
    text = re.sub(r"\s*-\s*$", "", text)

    return text.strip(), text.strip() != original


def clean_segment_text(
    text: str,
    language_code: str = "en"
) -> CleaningResult:
    """Clean a single transcript segment's text with language-aware rules.
    
    If cleaning fails or produces an empty string, falls back to original text.
    """
    if not text or not text.strip():
        return CleaningResult(cleaned_text="", original_text=text, has_changed=False)

    original = text
    cleaned = text
    transformations = []

    # 1. Whitespace normalization
    cleaned_ws = normalize_whitespace(cleaned)
    if cleaned_ws != cleaned:
        cleaned = cleaned_ws
        transformations.append("whitespace_normalization")

    # 2. Disfluent filler removal
    cleaned_fillers, changed_f = remove_disfluent_fillers(cleaned, language_code)
    if changed_f:
        cleaned = cleaned_fillers
        transformations.append("disfluent_filler_removal")

    # 3. Stutter and duplicate reduction
    cleaned_reps, changed_r = reduce_accidental_repetitions(cleaned)
    if changed_r:
        cleaned = cleaned_reps
        transformations.append("repetition_reduction")

    # 4. Punctuation and capitalization normalization
    cleaned_punct, changed_p = normalize_punctuation_and_capitalization(cleaned, language_code)
    if changed_p:
        cleaned = cleaned_punct
        transformations.append("punctuation_normalization")

    # Safety check: if cleaning emptied the text or became corrupted, preserve original
    if not cleaned.strip():
        return CleaningResult(
            cleaned_text=original.strip(),
            original_text=original,
            transformations_applied=["preserved_original_fallback"],
            has_changed=False
        )

    return CleaningResult(
        cleaned_text=cleaned,
        original_text=original,
        transformations_applied=transformations,
        has_changed=(cleaned != original.strip())
    )
