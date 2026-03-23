# Model Card: Mood Machine

---

## 1. Model Overview

**Model type:**
I used both models and compared them. The rule-based model was my primary implementation, with the ML model used to observe how a learned approach behaves on the same small dataset.

**Intended purpose:**
Classify short, informal text messages (social media posts, chat messages) into one of four mood labels: `positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**
The rule-based model assigns a numeric score to each post by scanning tokens against word lists, an emoji table, and a slang table. Negation words flip the score of the next token. The final score maps to a label using fixed thresholds.

The ML model (scikit-learn) uses a bag-of-words representation (CountVectorizer) to turn each post into a vector of word counts, then trains a classifier on the labeled posts. Instead of hand-written rules, it learns which words tend to appear in which mood categories from the training data.

---

## 2. Data

**Dataset description:**
The dataset contains 11 posts total — 6 from the lab starter and 5 added manually. Posts are short (under 15 words) and written in informal English.

**Labeling process:**
Labels were assigned by reading each post as a human would, attending to tone, implication, and context — not just individual words. Several posts were genuinely difficult to label:

- `"tired but grateful i guess 🥲"` — could be `neutral` or `mixed`; labeled `mixed` because the 🥲 emoji signals restrained emotion
- `"it's fine i'm fine everything is fine"` — reads as `neutral` on the surface but labeled `negative` because the repetition implies suppressed distress
- `"Lowkey stressed but kind of proud of myself"` — labeled `mixed`, though a case could be made for `negative` depending on how much weight you give "stressed"

**Important characteristics:**
- Contains slang: `lowkey`, `no cap`, `idk`, `rn`
- Contains emojis: `😂`, `🥲`
- Includes sarcasm: `"I absolutely love getting stuck in traffic 🙄"`
- Several posts express mixed or ambiguous feelings
- Some posts have no strong sentiment signal at all

**Possible issues:**
- The dataset is very small (11 posts), making it easy for either model to overfit
- Only one labeler assigned all labels, introducing personal bias
- Sarcastic posts are labeled by human interpretation, but neither model can verify that interpretation
- The dataset skews toward a specific register of informal American English

---

## 3. How the Rule-Based Model Works

**Scoring rules:**
- Each token is checked against a positive word list (`+1`), negative word list (`-1`), emoji table (fixed values), and slang table (fixed values)
- **Negation handling:** words like `not`, `never`, `don't` flip the sign of the immediately following token — e.g. `"not happy"` scores `-1` instead of `+1`
- **Vocabulary additions:** `stuck` and `traffic` were added to the negative word list after observing the sarcasm failure case
- **Thresholds:** scores of `±1` map to `mixed` rather than a confident label, because the word lists are small and a single match is weak evidence

**Strengths:**
- Transparent and fully explainable — every prediction can be traced back to specific tokens
- Negation handling correctly flips obvious cases like `"I am not happy about this"`
- Predictable behavior: the same input always produces the same output

**Weaknesses:**
- Cannot detect sarcasm — `"I love getting stuck in traffic"` scores as `mixed` at best because "love" counteracts the negative words
- Vocabulary coverage is narrow; any emotional language outside the word lists scores 0
- Negation only reaches one word forward, missing phrases like `"not happy at all"`
- Structural signals (repetition, punctuation patterns) are completely invisible

---

## 4. How the ML Model Works

**Features used:**
Bag-of-words representation using `CountVectorizer`. Each post becomes a vector where each dimension is a word from the vocabulary and the value is how many times it appears in that post.

**Training data:**
The model trained on all 11 posts in `SAMPLE_POSTS` with labels from `TRUE_LABELS`.

**Training behavior:**
With only 11 posts and 4 possible labels, the ML model is essentially memorizing the training data rather than generalizing. Adding or changing even a single label visibly shifts predictions, which shows how sensitive it is at this scale.

**Strengths and weaknesses:**
- Strength: learns word-label associations automatically without hand-written rules
- Weakness: with 11 examples, it overfits heavily — it performs well on training data but would likely fail on new posts it hasn't seen
- Weakness: it has no concept of negation, word order, or emoji meaning — `"not happy"` and `"happy"` look nearly identical to a bag-of-words model

---

## 5. Evaluation

**How the model was evaluated:**
Both models were evaluated on the same 11 labeled posts in `dataset.py`. This is training accuracy, not test accuracy — the model has already seen these examples, so results are optimistic.

**Examples of correct predictions:**

| Post | Label | Why it worked |
|---|---|---|
| `"I love this class so much"` | positive | "love" is a strong positive signal, no negation |
| `"Today was a terrible day"` | negative | "terrible" is unambiguous |
| `"no cap this is the best day of my life 😂"` | positive | Slang table catches "no cap", emoji adds signal |

**Examples of incorrect predictions:**

| Post | True Label | Predicted | Why it failed |
|---|---|---|---|
| `"I absolutely love getting stuck in traffic 🙄"` | negative | mixed | "love" adds a positive point; model can't detect sarcasm |
| `"it's fine i'm fine everything is fine"` | negative | neutral | "fine" is not in any word list; repetition is invisible |
| `"Lowkey stressed but kind of proud of myself"` | mixed | mixed (correct score, wrong path) | "proud" not in positive list, only "stressed" fires |

---

## 6. Limitations

- **Dataset is too small** for the ML model to generalize — 11 examples across 4 classes is not enough for reliable learning
- **Sarcasm is undetectable** with either approach at this scale — it requires contextual understanding neither model has
- **Rule-based vocabulary is narrow** — any post using emotional language outside the 22-word lists defaults to neutral
- **No held-out test set** — all evaluation is on training data, so reported accuracy is not a reliable measure of real performance
- **Single labeler** — all ground-truth labels reflect one person's interpretation; reasonable people might disagree on several examples
- **English-only and register-specific** — both models are built around a particular style of informal American English and would likely underperform on other dialects or more formal text

---

## 7. Ethical Considerations

**Misclassifying distress:**
A post like `"it's fine i'm fine everything is fine"` expresses implied distress but is classified as `neutral`. In a real application — such as a mental health support tool or a content moderation system — this kind of false neutral could mean someone in genuine distress is overlooked.

**Language and dialect bias:**
The slang and vocabulary in this model reflect a specific cultural context. Posts written in AAVE, British slang, or other English dialects may use words not in the word lists, causing them to score as neutral by default — systematically underrepresenting those communities' emotional expression.

**Privacy:**
Mood classification on personal messages raises significant privacy concerns. Even if the model is used with good intentions (e.g. detecting distress), analyzing private communication without explicit consent is ethically problematic and potentially illegal in many jurisdictions.

**Overconfidence in a weak model:**
The model's output looks authoritative (a label with a score) but is based on very thin evidence. Deploying even a simple tool like this in a real context risks giving false confidence to decisions that should involve human judgment.

---

## 8. Ideas for Improvement

- **Add more labeled data** — at minimum 50–100 examples per label class for the ML model to generalize
- **Use TF-IDF instead of CountVectorizer** — down-weights common words that appear everywhere and up-weights distinctive ones
- **Build a real test set** — hold out 20% of data before training so accuracy reflects genuine generalization
- **Improve emoji handling** — expand the emoji table and handle multi-emoji sequences
- **Extend negation scope** — handle phrases like `"not happy at all"` by negating a window of tokens rather than just one
- **Add a sarcasm signal** — certain patterns (positive word + negative emoji, or phrases like "absolutely love") could trigger a sarcasm flag
- **Use multiple labelers** and measure inter-annotator agreement to produce more reliable ground truth
- **Try a small transformer model** (e.g. a fine-tuned DistilBERT) for context-aware classification that can detect tone and sarcasm