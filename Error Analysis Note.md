# Error Analysis Note

This note reviews representative mistakes made by the transformation-aware detector. A **false positive** is a real image predicted as fake; a **false negative** is a fake image predicted as real. Probabilities below are fake-class probabilities at a `0.5` decision threshold.

The detailed error-analysis runs contain 2,000 images from each evaluation set, tested in clean form and under seven additional transformation or laundering conditions (16,000 evaluated rows per dataset).

## Error Overview

| Evaluation set | Clean errors | False positives | False negatives | Images failing at least one transformation | Images whose prediction flips | Gate helped / hurt | Net gate effect |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAKE | 41 | 24 | 17 | 811 | 849 | 57 / 210 | **−153** |
| SID | 12 | 3 | 9 | 73 | 76 | 49 / 123 | **−74** |

The counts show two distinct patterns: CIFAKE has many more transformation-sensitive examples, while SID has fewer overall errors but still contains cases where gating changes a correct base-fusion decision into an incorrect final decision. “Gate helped” and “gate hurt” are row-level comparisons against the ungated base fusion, not unique-image counts.

## Representative False Positives

| Dataset and example | Final fake probability | Branch behavior | Interpretation |
|---|---:|---|---|
| CIFAKE `REAL/0431 (5).jpg` | 98.95% | Semantic 99.99%; forensic 98.33%; weights 50.43% / 49.57% | Both branches agree on the wrong class. The calibrated reliability is only 0.28%, so the system has evidence that this prediction is unstable even though the class score is extreme. |
| CIFAKE `REAL/0350 (3).jpg` | 96.95% | Semantic 99.99%; forensic 50.84%; semantic weight 77.82% | A high-reliability false positive: the semantic branch dominates and the confidence estimate does not identify the mistake. |
| SID `72ff457e54c70c90_9f9206d13ef4.jpg` | 97.37% | Semantic 99.99%; forensic 61.07%; semantic weight 73.38% | A real image contains features that resemble the learned synthetic-image concept in both branches, producing a confident false alarm. |

These examples show why a confidence threshold alone is insufficient. Some errors have low estimated reliability and could be referred for review, but others remain confidently and consistently wrong.

## Representative False Negatives

| Dataset and example | Final fake probability | Branch behavior | Interpretation |
|---|---:|---|---|
| CIFAKE `FAKE/349 (4).jpg` | 4.82% | Semantic ≈0%; forensic 83.04%; semantic weight 96.65% | The forensic branch identifies the image as fake, but the gate heavily favors the incorrect semantic branch. |
| CIFAKE `FAKE/81 (4).jpg` | 8.77% | Semantic ≈0%; forensic 94.87%; semantic weight 98.61% | Very large branch disagreement is resolved in the wrong direction. Under noise + JPEG, the same image's base fusion is correct at 50.77%, but gating lowers the final score to 21.04%. |
| SID `full_synthetic_001303_1d75e0068d69.png` | 2.43% | Semantic 0.01%; forensic 6.38% | Both branches agree on the wrong class, so changing the gate alone cannot fix this case. |
| SID `full_synthetic_003183_b06c3b1f9f8a.png` | 2.94% | Semantic 0.04%; forensic 89.51%; semantic weight 99.15% | As in the CIFAKE cases, the gate suppresses a useful forensic signal because the semantic branch is strongly favored. |

False negatives are the more important operational risk if the detector is used to screen synthetic content: a missed fake passes through as real. The examples also separate **representation failures** (both branches are wrong) from **fusion failures** (one branch is correct but is underweighted).

## Transformation-Induced Failures

- CIFAKE `FAKE/182 (6).jpg` falls to a 0.33% fake probability under aggressive reposting; both semantic and forensic evidence are largely erased.
- CIFAKE `FAKE/996 (4).jpg` falls to 0.41% after blur, resizing, and JPEG compression.
- SID `full_synthetic_003725_4a0d599c81f3.png` falls to 0.27% under full laundering. Its forensic branch remains at 76.42%, but a 97.55% semantic weight overrides it.
- SID `full_synthetic_000594_b30819d2b3a4.png` falls to 0.40% under noise plus JPEG, with both branches close to the real class.

These cases explain the much lower CIFAKE accuracy under extreme reposting (71.33%) and full laundering (74.03%). Severe processing either removes the forensic traces used by the model or shifts the semantic representation far enough to resemble the real class.

## Main Trade-offs

### Semantic Gating vs. Forensic Evidence

The gate is beneficial when the forensic branch reacts to misleading compression or texture cues. For example, on clean CIFAKE `REAL/0913.jpg`, the base fusion predicts fake at 51.92%, while the gated result correctly drops to 21.33% by favoring the semantic branch. However, the same behavior causes false negatives such as `FAKE/81 (4).jpg`, where a correct forensic signal is suppressed. Across the error-analysis rows, this trade-off is net negative relative to base fusion: 57 improvements versus 210 regressions on CIFAKE, and 49 versus 123 on SID.

### Corruption Robustness vs. Domain Generalization

Training for synthetic transformations produces strong in-domain robustness, but does not guarantee that the learned forensic direction transfers. On clean WildFake, the full detector reaches only 44.67% AUROC and 14.05% fake recall. On the balanced diagnostic subset, the forensic score is effectively reversed (2.78% AUROC; 97.22% after inversion), indicating reliance on dataset-specific acquisition or generator cues.

### Recall vs. False Alarms

A lower decision threshold could recover more fake images, but would increase false positives. This is visible on WildFake resize 0.25×: fake recall rises to 75.60%, but the false-positive rate also rises to 52.66%. A single global `0.5` threshold is therefore unlikely to be optimal across datasets and transformation families.

### Calibration vs. Correctness

Temperature calibration improves probability quality without changing the predicted class. On CIFAKE, expected calibration error falls from 2.58% to 0.90%, and on SID from 0.57% to 0.55%, while accuracy stays unchanged. Calibration makes confidence easier to interpret, but cannot correct systematic branch inversion, class-representation errors, or a poorly chosen fusion weight.

## Recommended Improvements

- Train and validate the gate on more datasets and generator families, with an explicit penalty when it suppresses a correct branch during strong disagreement.
- Add out-of-domain validation and forensic-sign inversion checks before deployment.
- Use reliability and branch disagreement to create an abstention or human-review region instead of forcing every image into a binary decision.
- Tune operating thresholds for the intended cost of false positives versus false negatives, and report threshold-free AUROC alongside deployment-specific recall and false-positive rate.
- Expand laundering augmentation with repeated reposting, screenshots, aggressive rescaling, and mixed codec pipelines, especially for the CIFAKE domain.

## Qualitative Evidence and Raw Reports

| Category | CIFAKE examples | SID examples |
|---|---|---|
| False positives | [Contact sheet](outputs/evaluation/error_analysis/cifake/contact_sheets/clean_false_positive.png) | [Contact sheet](outputs/evaluation/error_analysis/sid/contact_sheets/clean_false_positive.png) |
| False negatives | [Contact sheet](outputs/evaluation/error_analysis/cifake/contact_sheets/clean_false_negative.png) | [Contact sheet](outputs/evaluation/error_analysis/sid/contact_sheets/clean_false_negative.png) |
| Gate helped | [Contact sheet](outputs/evaluation/error_analysis/cifake/contact_sheets/gate_helped.png) | [Contact sheet](outputs/evaluation/error_analysis/sid/contact_sheets/gate_helped.png) |
| Gate hurt | [Contact sheet](outputs/evaluation/error_analysis/cifake/contact_sheets/gate_hurt.png) | [Contact sheet](outputs/evaluation/error_analysis/sid/contact_sheets/gate_hurt.png) |
| Transformation failures | [Contact sheet](outputs/evaluation/error_analysis/cifake/contact_sheets/transformation_failure.png) | [Contact sheet](outputs/evaluation/error_analysis/sid/contact_sheets/transformation_failure.png) |

Raw summaries: [CIFAKE error report](outputs/evaluation/error_analysis/cifake/error_report.json) · [SID error report](outputs/evaluation/error_analysis/sid/error_report.json)
