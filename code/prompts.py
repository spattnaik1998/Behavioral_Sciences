"""
All prompts used in the revision, in one place (so they can be printed
verbatim in the paper's appendix).

The classification prompt is assembled from switchable parts so that the
ablation conditions differ in exactly one component at a time:
  retrieval   -> the "Context from literature" block
  definitions -> the two REGRET / DISAPPOINTMENT definition lines
Everything else (role, output schema) is identical across conditions.
"""

SYSTEM = "You are an expert in behavioral economics."

# Definitions: verbatim from the original paper (Listing 6)
DEFINITIONS = (
    'REGRET: Personal responsibility, "I should have done differently"\n'
    'DISAPPOINTMENT: External factors, "It should have been better"'
)

TASK_NO_DEFS = "Classify the emotion expressed in the customer review as either REGRET or DISAPPOINTMENT."
TASK_WITH_DEFS = "Classify the emotion expressed in the customer review as either REGRET or DISAPPOINTMENT, using these definitions:"

OUTPUT_SCHEMA = (
    'Respond with JSON only:\n'
    '{"emotion": "REGRET" or "DISAPPOINTMENT", "confidence": <number between 0 and 1>, '
    '"rationale": "<one sentence>"}'
)


def classification_messages(review: str, context: str | None, definitions: bool) -> list[dict]:
    parts = []
    if context is not None:
        parts.append(f"Context from literature:\n{context}")
    parts.append(f"Customer Review:\n{review}")
    parts.append(f"{TASK_WITH_DEFS}\n{DEFINITIONS}" if definitions else TASK_NO_DEFS)
    parts.append(OUTPUT_SCHEMA)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


# --------------------------------------------------------------------------
# Original prompt, verbatim from the notebook cell that produced the paper's
# results (single human message, no system message, no JSON mode).
# --------------------------------------------------------------------------
ORIGINAL_TEMPLATE = """You are an expert in behavioral economics.

Context: {context}
Review: {question}

REGRET: Personal responsibility, "I should have..."
DISAPPOINTMENT: External factors, "It should have..."

Respond with JSON only:
{{"emotion": "REGRET" or "DISAPPOINTMENT", "confidence": 0.X}}
"""


def original_messages(review: str, context: str) -> list[dict]:
    return [{"role": "user", "content": ORIGINAL_TEMPLATE.format(context=context, question=review)}]


# --------------------------------------------------------------------------
# RDS dimensional assessment - verbatim from the paper (Listing 7)
# --------------------------------------------------------------------------
RDS_TEMPLATE = """Rate this review (1-7 scale):

Review: {text}

1. Counterfactual Thinking (1=none, 7=extensive)
2. Personal Responsibility (1=external, 7=personal)
3. Controllability (1=uncontrollable, 7=controllable)
4. Negative Affect (1=mild, 7=intense)

JSON only:
{{"counterfactual": X, "responsibility": X, "controllability": X, "affect": X}}
"""


def rds_messages(review: str) -> list[dict]:
    return [{"role": "user", "content": RDS_TEMPLATE.format(text=review)}]


# --------------------------------------------------------------------------
# Counterfactual focus (Section 3.3.5 of the paper; Zeelenberg et al., 1998).
# Described in the paper but absent from the original code - implemented here.
# --------------------------------------------------------------------------
CF_TEMPLATE = """Read the customer review and identify counterfactual thinking ("if only...", "should have...", "wish...").

Review: {text}

Classify the PRIMARY focus of the counterfactual thinking:
BEHAVIOR  - centred on the reviewer's own actions or choices ("If only I had bought another brand", "I should have read the reviews")
SITUATION - centred on the product, seller or circumstances ("If only it had lasted longer", "It should have worked out of the box")
NONE      - no counterfactual thinking is expressed

Respond with JSON only:
{{"focus": "BEHAVIOR" or "SITUATION" or "NONE", "evidence": "<short quote from the review, or empty>"}}
"""


def cf_messages(review: str) -> list[dict]:
    return [{"role": "user", "content": CF_TEMPLATE.format(text=review)}]


# --------------------------------------------------------------------------
# Speculative RAG (Wang et al., 2024, arXiv:2407.08223)
#   drafter  : small model, one draft (rationale + answer) per document subset
#   verifier : large generalist model scores each draft; best draft is selected
# --------------------------------------------------------------------------
DRAFTER_TEMPLATE = """Evidence from the behavioral-economics literature:
{context}

Customer Review:
{review}

{task}
{definitions}

First write a short rationale grounded in the evidence and the review, then give the answer.
Respond with JSON only:
{{"rationale": "<1-2 sentences>", "emotion": "REGRET" or "DISAPPOINTMENT"}}"""


def drafter_messages(review: str, context: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": DRAFTER_TEMPLATE.format(
                context=context, review=review, task=TASK_WITH_DEFS, definitions=DEFINITIONS)}]


# Self-consistency: verifier answers the question given the drafted rationale;
# we read P(label of the draft) from its log-probabilities.
VERIFY_SC_TEMPLATE = """Customer Review:
{review}

{task}
{definitions}

Rationale: {rationale}

Respond with JSON only:
{{"emotion": "REGRET" or "DISAPPOINTMENT"}}"""

# Self-reflection: "Do you think the rationale supports the answer, yes or no?"
VERIFY_SR_TEMPLATE = """Customer Review:
{review}

Proposed answer: {emotion}
Rationale: {rationale}

Do you think the rationale supports the answer, yes or no? Answer with a single word: Yes or No."""


def verifier_sc_messages(review: str, rationale: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": VERIFY_SC_TEMPLATE.format(
                review=review, task=TASK_WITH_DEFS, definitions=DEFINITIONS, rationale=rationale)}]


def verifier_sr_messages(review: str, emotion: str, rationale: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": VERIFY_SR_TEMPLATE.format(review=review, emotion=emotion, rationale=rationale)}]


def all_prompts_text() -> str:
    """Human-readable dump of every prompt, for the paper's appendix."""
    ex = "<REVIEW TEXT>"; ctx = "<RETRIEVED CHUNKS>"
    blocks = {
        "C0 labels only (no retrieval, no definitions)": classification_messages(ex, None, False),
        "C1 definitions (no retrieval)": classification_messages(ex, None, True),
        "C2 RAG without definitions": classification_messages(ex, ctx, False),
        "C3 RAG with definitions (proposed)": classification_messages(ex, ctx, True),
        "Original paper prompt (replication)": original_messages(ex, ctx),
        "RDS dimensional assessment": rds_messages(ex),
        "Counterfactual focus": cf_messages(ex),
        "Speculative RAG - drafter": drafter_messages(ex, ctx),
        "Speculative RAG - verifier (self-consistency)": verifier_sc_messages(ex, "<DRAFT RATIONALE>"),
        "Speculative RAG - verifier (self-reflection)": verifier_sr_messages(ex, "<DRAFT LABEL>", "<DRAFT RATIONALE>"),
    }
    out = []
    for name, msgs in blocks.items():
        out.append("=" * 78 + f"\n{name}\n" + "=" * 78)
        for m in msgs:
            out.append(f"[{m['role']}]\n{m['content']}\n")
    return "\n".join(out)
