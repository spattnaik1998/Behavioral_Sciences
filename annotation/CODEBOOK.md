# Coding manual: regret vs disappointment in customer reviews

You will read low-rated Amazon product reviews and code the emotion the reviewer
expresses. Work **independently**: do not discuss items with the other coder until
both of you have finished. Star ratings and all model outputs are deliberately hidden.

## Before you start (15 minutes, together)
Both coders label the 10 reviews in the `practice` sheet on their own, then compare and
discuss them using this manual. The practice reviews are NOT part of the study. After that,
code the `coding` sheet alone.

## Theoretical basis
Zeelenberg et al. (1998, 2000); Zeelenberg & Pieters (2007); Marcatto & Ferrante (2008).
Regret and disappointment are both negative, outcome-related emotions. They differ in
**who is responsible** for the bad outcome.

| | REGRET | DISAPPOINTMENT |
|---|---|---|
| Responsibility | The reviewer's own choice or action | The product, seller, brand or circumstances |
| Typical thought | "I should have bought X instead", "I wish I had read the reviews" | "It should have lasted longer", "It did not work as advertised" |
| Counterfactual | Changes own **behaviour** | Changes the **situation** |
| Control | Outcome was controllable by the reviewer | Outcome was outside the reviewer's control |

## Fields (REQUIRED fields are marked)
1. **primary_emotion** (REQUIRED): REGRET / DISAPPOINTMENT / BOTH / NEITHER.
   - BOTH: clear self-blame **and** clear blame of the product, with neither dominant.
   - NEITHER: no regret or disappointment, e.g. a neutral description, a mostly positive
     3-star review, a question, or spam.
2. **forced_choice** (REQUIRED): if you had to choose only REGRET or DISAPPOINTMENT, which one?
   Always fill this in, even when primary_emotion is BOTH or NEITHER.
3. **certainty** (REQUIRED): your confidence in the primary_emotion code (LOW / MEDIUM / HIGH).
4. **cf_focus** (recommended): primary focus of any counterfactual thinking: BEHAVIOR (own actions),
   SITUATION (product/circumstances), or NONE (no counterfactual expressed).
5. **counterfactual / responsibility / controllability / affect** (recommended, 1-7 each):
   - counterfactual: how much "what if / if only / should have" reasoning (1 = none, 7 = extensive)
   - responsibility: who is held responsible (1 = entirely external, 7 = entirely the reviewer)
   - controllability: how controllable the outcome was for the reviewer (1 = uncontrollable, 7 = fully)
   - affect: intensity of negative emotion (1 = mild, 7 = intense)
6. **notes** (optional): flag sarcasm, mixed reviews, non-English text, etc.

## Decision rules
- Code what the text **expresses**, not what you think the reviewer *should* feel.
- Product defects, wrong/damaged item, poor quality, short battery life, or "not as described"
  = DISAPPOINTMENT.
- **"Would not buy again" / "Won't buy this brand again" / "Stick with Duracell"** on their own are
  statements of future intention, not self-blame: DISAPPOINTMENT. They become REGRET only when the
  reviewer also blames their own past decision ("I should have stuck with Duracell",
  "lesson learned, I should have paid more").
- **"Waste of money"** on its own = DISAPPOINTMENT; with self-blame for the decision
  ("I wasted my money, should have listened to the reviews") = REGRET.
- Self-blame for trusting the brand, for not reading the description or reviews, for ordering the
  wrong item, or for misusing the product ("my fault", "I ordered AAA by mistake") = REGRET.
- Missing a return window: REGRET if the reviewer blames themselves ("I waited too long"),
  DISAPPOINTMENT if they blame the policy.
- Short or uninformative reviews ("ok", "not great"): primary_emotion = NEITHER if no emotion can
  be read; still make a forced_choice.

## Practical
- Use the drop-down menus. Do not change the review_uid column or the item order.
- Expect about 20-30 seconds per review for the required fields, about 1 minute with all fields.
- Save the file under the same name in `annotation/returned/`. It is fine for coder B to code only
  part of the sheet (top to bottom); agreement is computed on the reviews both coders completed.
