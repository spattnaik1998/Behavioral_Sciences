# Annotator guide: regret and disappointment in customer reviews

This guide is for the two people coding reviews for the study. Part 1 explains the theory you need. Part 2 tells you which file, sheet and columns to fill in. Part 3 lists the decision rules for hard cases. The same rules are in the `codebook` sheet inside your workbook.

Code alone. Do not discuss reviews with the other coder until you have both finished. Star ratings and all model outputs are hidden on purpose.

---

## Part 1. The theory

### Two emotions, one difference

Regret and disappointment are both negative feelings about a bad outcome. They differ in **who is to blame**.

- **Regret** is the feeling that *you* would be better off if *you* had chosen differently. It needs a choice the person made and now wishes they had not made. Its typical thought is "I should have…" (Zeelenberg et al., 1998; Zeelenberg & Pieters, 2007).
- **Disappointment** is the feeling that an outcome fell short of what you expected, for reasons outside your control. Its typical thought is "It should have…" (Zeelenberg et al., 1998; van Dijk et al., 1999).

A person can feel both about the same purchase. The product broke (disappointment), and they also blame themselves for buying the cheap brand (regret).

### How to tell them apart in text

Marcatto and Ferrante (2008) found that people cannot reliably name their own feeling as regret or disappointment. Their Regret and Disappointment Scale therefore measures the thoughts behind the feeling, not the label. We use the same approach. Look at four things in the review.

| Cue | Points to regret | Points to disappointment |
|---|---|---|
| **Responsibility**: who caused the bad outcome? | The reviewer: their choice, research, order, or use of the product | The product, brand, seller, delivery, or bad luck |
| **Controllability**: could the reviewer have prevented it? | Yes: "I could have read the reviews" | No: "There was no way to know it would fail" |
| **Counterfactual**: what does the reviewer imagine changing? | Their own **behavior**: "If only I had bought Duracell" | The **situation**: "If only it had lasted longer" |
| **Typical words** | "I should have", "I wish I had(n't)", "my mistake", "lesson learned", "I regret buying" | "It should have", "didn't work", "stopped working", "not as described", "poor quality" |

The counterfactual cue comes from Zeelenberg et al. (1998). People who recalled a regret mostly imagined changing what *they* did. People who recalled a disappointment mostly imagined the *circumstances* being different. We record this cue separately in the `cf_focus` column.

### Why regret is less common in reviews

When a product fails, most reviewers blame the product, which is disappointment. Regret appears when the reviewer turns the blame on their own decision: choosing the cheap option, ignoring warnings, ordering the wrong size. Expect most reviews to be disappointment, but do not let that expectation decide any single review. Code what each text says.

### References

- Marcatto, F., & Ferrante, D. (2008). The Regret and Disappointment Scale: An instrument for assessing regret and disappointment in decision making. *Judgment and Decision Making*, 3(1), 87–99.
- van Dijk, W. W., Zeelenberg, M., & van der Pligt, J. (1999). Not having what you want versus having what you do not want: The impact of type of negative outcome on the experience of disappointment and related emotions. *Cognition and Emotion*, 13(2), 129–148.
- Zeelenberg, M., van Dijk, W. W., Manstead, A. S. R., & van der Pligt, J. (1998). The experience of regret and disappointment. *Cognition and Emotion*, 12(2), 221–230.
- Zeelenberg, M., & Pieters, R. (2007). A theory of regret regulation 1.0. *Journal of Consumer Psychology*, 17(1), 3–18.

---

## Part 2. Where to fill in the labels

### Your file

Each coder has a workbook in `Behavioral_Economics/Revision/annotation/`:

| Coder | Workbook |
|---|---|
| Coder A | `annotation_coder_A.xlsx` |
| Coder B | `annotation_coder_B.xlsx` |

Both workbooks contain the same 285 reviews in a different order. Use only your own file.

### The four sheets

| Sheet | What it is | Do you fill it in? |
|---|---|---|
| `practice` | 10 reviews that are **not** in the study, for a 15-minute warm-up | Yes, first. Then compare with the other coder and discuss. |
| `coding` | The 285 study reviews | **Yes. This is the main task.** Fill it in alone. |
| `codebook` | The same rules as this guide | No, read only |
| `meta` | Coder ID and item count | No, do not edit |

### Columns in the `coding` sheet

Columns A to D are filled in for you. Do not change them, and do not sort or reorder the rows. Fill in columns E to M. Every coded column has a drop-down menu; use it instead of typing.

| Col | Column name | Required? | Allowed values | What to record |
|---|---|---|---|---|
| A | `item` | given | | Row number |
| B | `review_uid` | given | | Review ID. **Never edit.** |
| C | `product` | given | | Product name |
| D | `review_text` | given | | The review |
| E | `primary_emotion` | **Required** | REGRET, DISAPPOINTMENT, BOTH, NEITHER | The main emotion expressed |
| F | `forced_choice` | **Required** | REGRET, DISAPPOINTMENT | If you could pick only one of the two. Fill it in even when E is BOTH or NEITHER. |
| G | `cf_focus` | Recommended | BEHAVIOR, SITUATION, NONE | Main focus of any "if only / should have" thought |
| H | `counterfactual` | Recommended | 1–7 | How much "what if" thinking (1 = none, 7 = a lot) |
| I | `responsibility` | Recommended | 1–7 | Who is blamed (1 = entirely external, 7 = entirely the reviewer) |
| J | `controllability` | Recommended | 1–7 | How far the reviewer could have prevented it (1 = not at all, 7 = fully) |
| K | `affect` | Recommended | 1–7 | Strength of the negative feeling (1 = mild, 7 = intense) |
| L | `certainty` | **Required** | LOW, MEDIUM, HIGH | Your confidence in column E |
| M | `notes` | Optional | free text | Sarcasm, mixed review, unreadable text, anything odd |

The `practice` sheet has only `primary_emotion`, `forced_choice`, `certainty` and `notes`.

Expect about 20–30 seconds per review for the required columns and about 1 minute with all columns.

### Returning your file

1. Save the workbook under its **original name** (`annotation_coder_A.xlsx` or `annotation_coder_B.xlsx`).
2. Put it in `Behavioral_Economics/Revision/annotation/returned/`.
3. Coder B may stop partway if time runs out, as long as the rows are coded from the top down. Agreement is computed on the reviews both coders completed.

---

## Part 3. Decision rules for hard cases

Code what the text **says**, not what you think the reviewer ought to feel.

| Situation | Code |
|---|---|
| Defect, poor quality, short battery life, damaged or wrong item sent, "not as described" | DISAPPOINTMENT |
| "Won't buy again", "Stick with Duracell", "Not buying this brand again" **on their own** | DISAPPOINTMENT. This is a plan for the future, not self-blame. |
| The same, **plus** blame for the past decision: "I *should have* stuck with Duracell" | REGRET |
| "Waste of money" on its own | DISAPPOINTMENT |
| "I wasted my money, should have listened to the reviews" | REGRET |
| Blames self for trusting the brand, skipping the description or reviews, ordering the wrong item, or misusing the product ("my fault", "I ordered AAA by mistake") | REGRET |
| Missed the return window: "I waited too long" | REGRET |
| Missed the return window: "Their return policy is too short" | DISAPPOINTMENT |
| Clear self-blame **and** clear product blame, neither dominant | BOTH (and still make a forced choice) |
| "ok", "not great", a question, spam, or a mostly positive 3-star review | NEITHER (and still make a forced choice) |
| "I expected too much" | Self-blame for the expectation, not the choice: usually DISAPPOINTMENT. Use REGRET only if the reviewer says they should have chosen differently. Note it in column M. |

### Worked examples

| Review | primary_emotion | forced_choice | cf_focus | Why |
|---|---|---|---|---|
| "These batteries died in a day." | DISAPPOINTMENT | DISAPPOINTMENT | NONE | Blames the product; no counterfactual |
| "It should have worked out of the box." | DISAPPOINTMENT | DISAPPOINTMENT | SITUATION | Imagines the product being different |
| "I should have stuck with Duracell." | REGRET | REGRET | BEHAVIOR | Blames own brand choice |
| "Terrible decision on my part to trust Amazon to make a decent battery." | REGRET | REGRET | BEHAVIOR | Explicit self-blame for the decision |
| "Batteries are junk, and honestly my fault for going cheap." | BOTH | REGRET | BEHAVIOR | Product blame and self-blame together |
| "ok" | NEITHER | DISAPPOINTMENT | NONE | No emotion can be read |
