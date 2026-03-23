"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    "stuck",    # added: common in frustration contexts ("stuck in traffic")
    "traffic",  # added: almost always negative in casual language
]

# ---------------------------------------------------------------------
# Labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    # --- Starter posts ---
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",

    # --- Extended posts ---
    "Lowkey stressed but kind of proud of myself",   # slang, mixed emotions
    "tired but grateful i guess 🥲",                 # ambiguous emoji, edge case
    "it's fine i'm fine everything is fine",         # sarcasm, fools rule-based models
    "no cap this is the best day of my life 😂",     # slang, clearly positive
    "honestly idk how to feel rn",                   # genuinely neutral/ambiguous
]

# Human labels for each post above.
# Allowed labels:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    # --- Starter labels ---
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"

    # --- Extended labels ---
    "mixed",     # "Lowkey stressed but kind of proud of myself"
    "mixed",     # "tired but grateful i guess 🥲"
    "negative",  # "it's fine i'm fine everything is fine" — repetition implies distress
    "positive",  # "no cap this is the best day of my life 😂"
    "neutral",   # "honestly idk how to feel rn"
]

# Sanity check — will crash loudly if lists ever get misaligned
assert len(SAMPLE_POSTS) == len(TRUE_LABELS), (
    f"Mismatch: {len(SAMPLE_POSTS)} posts but {len(TRUE_LABELS)} labels"
)