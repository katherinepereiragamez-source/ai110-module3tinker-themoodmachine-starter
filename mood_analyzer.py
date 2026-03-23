"""
Rule based mood analyzer for short text snippets.
"""

import re
from typing import List, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


# ---------------------------------------------------------------------------
# Emoji / slang signal tables
# ---------------------------------------------------------------------------

EMOJI_SCORES: dict[str, int] = {
    "😂": 1,  "🥲": -1, "😭": -1, "💀": -1,
    "🙄": -1, "❤️": 2,  "😊": 1,  "😍": 2,
    ":)": 1,  ":-)": 1, ":(": -1, ":-(": -1,
}

SLANG_SCORES: dict[str, int] = {
    "lowkey": 0,   # modifier — handled in negation/context logic
    "highkey": 1,
    "no cap": 1,   # matched as a phrase before splitting
    "idk": 0,
    "rn": 0,
    "lol": 1,
    "lmao": 1,
    "ugh": -1,
    "meh": -1,
}

NEGATION_WORDS = {"not", "never", "no", "don't", "doesn't", "didn't", "isn't", "wasn't"}


class MoodAnalyzer:
    """A rule-based mood classifier with negation handling and emoji/slang signals."""

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # -----------------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a clean list of tokens.

        Improvements over the starter version:
          1. Extract emojis as their own tokens BEFORE stripping them.
          2. Normalize text-face emoticons (:), :( etc.) as tokens.
          3. Replace punctuation with spaces (keeps words intact).
          4. Collapse repeated characters: "soooo" -> "soo" (two max).
          5. Lowercase everything.
          6. Split on whitespace.
        """
        # 1. Pull out unicode emojis first (they'd be destroyed by regex below)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F9FF"  # transport, misc symbols
            "\u2600-\u26FF"
            "\u2700-\u27BF"
            "]+",
            flags=re.UNICODE,
        )
        emoji_tokens = emoji_pattern.findall(text)
        text_no_emoji = emoji_pattern.sub(" ", text)

        # 2. Preserve text-face emoticons before stripping punctuation
        emoticon_pattern = re.compile(r"[:;]-?[)(\\/|DPp]")
        emoticon_tokens = emoticon_pattern.findall(text_no_emoji)
        text_no_emoji = emoticon_pattern.sub(" ", text_no_emoji)

        # 3. Remove remaining punctuation
        text_clean = re.sub(r"[^\w\s]", " ", text_no_emoji)

        # 4. Collapse repeated characters (e.g. "sooo" -> "soo")
        text_clean = re.sub(r"(.)\1{2,}", r"\1\1", text_clean)

        # 5. Lowercase and split
        tokens = text_clean.lower().split()

        # 6. Add emoji/emoticon tokens back
        tokens += [e.lower() for e in emoji_tokens + emoticon_tokens]

        return tokens

    # -----------------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric mood score.

        Enhancements:
          - Negation handling: "not happy" flips the score of the next word.
          - Emoji signals: mapped directly to scores via EMOJI_SCORES.
          - Slang signals: common slang mapped via SLANG_SCORES.
          - Phrase detection: "no cap" counted as a positive phrase before tokenizing.
        """
        # Phrase-level check before tokenizing (e.g. "no cap")
        phrase_score = 0
        lower_text = text.lower()
        for phrase, val in SLANG_SCORES.items():
            if " " in phrase and phrase in lower_text:
                phrase_score += val

        tokens = self.preprocess(text)
        score = phrase_score
        negate_next = False

        for token in tokens:
            # Check for negation trigger
            if token in NEGATION_WORDS:
                negate_next = True
                continue

            multiplier = -1 if negate_next else 1
            negate_next = False  # reset after one word

            # Emoji signals
            if token in EMOJI_SCORES:
                score += multiplier * EMOJI_SCORES[token]

            # Slang signals (single-word only; phrases handled above)
            elif token in SLANG_SCORES and " " not in token:
                score += multiplier * SLANG_SCORES[token]

            # Standard word lists
            elif token in self.positive_words:
                score += multiplier * 1
            elif token in self.negative_words:
                score += multiplier * -1

        return score

    # -----------------------------------------------------------------------
    # Label prediction
    # -----------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Map a numeric score to a mood label.

        Thresholds (chosen based on our small word lists):
          score >= 2   -> "positive"   (strong positive signal)
          score == 1   -> "mixed"      (weak positive — could go either way)
          score == 0   -> "neutral"
          score == -1  -> "mixed"      (weak negative — could go either way)
          score <= -2  -> "negative"   (strong negative signal)
        """
        score = self.score_text(text)

        if score >= 2:
            return "positive"
        elif score == 1:
            return "mixed"
        elif score == 0:
            return "neutral"
        elif score == -1:
            return "mixed"
        else:
            return "negative"

    # -----------------------------------------------------------------------
    # Explanation
    # -----------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """Return a human-readable breakdown of why the model chose its label."""
        tokens = self.preprocess(text)
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        negated_hits: List[str] = []
        score = 0
        negate_next = False

        for token in tokens:
            if token in NEGATION_WORDS:
                negate_next = True
                continue

            multiplier = -1 if negate_next else 1
            was_negated = negate_next
            negate_next = False

            hit_val = None
            if token in EMOJI_SCORES:
                hit_val = EMOJI_SCORES[token]
            elif token in SLANG_SCORES and " " not in token:
                hit_val = SLANG_SCORES[token]
            elif token in self.positive_words:
                hit_val = 1
            elif token in self.negative_words:
                hit_val = -1

            if hit_val is not None:
                actual = multiplier * hit_val
                score += actual
                if was_negated:
                    negated_hits.append(token)
                elif actual > 0:
                    positive_hits.append(token)
                elif actual < 0:
                    negative_hits.append(token)

        label = self.predict_label(text)
        return (
            f"Label: {label!r} | Score = {score} | "
            f"positive: {positive_hits or []} | "
            f"negative: {negative_hits or []} | "
            f"negated: {negated_hits or []}"
        )


# ---------------------------------------------------------------------------
# Quick smoke test — run this file directly to verify behavior
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    analyzer = MoodAnalyzer()

    test_posts = [
        "I love this class so much",
        "Today was a terrible day",
        "I am not happy about this",
        "it's fine i'm fine everything is fine",
        "no cap this is the best day of my life 😂",
        "tired but grateful i guess 🥲",
        "Lowkey stressed but kind of proud of myself",
        "honestly idk how to feel rn",
    ]

    print(f"{'POST':<45} {'TOKENS'}")
    print("-" * 90)
    for post in test_posts:
        tokens = analyzer.preprocess(post)
        print(f"{post:<45} {tokens}")

    print("\n")
    print(f"{'POST':<45} {'EXPLANATION'}")
    print("-" * 110)
    for post in test_posts:
        print(f"{post:<45} {analyzer.explain(post)}")