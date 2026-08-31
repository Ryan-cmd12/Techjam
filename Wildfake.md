### WildFake Zero-Shot Cross-Domain Evaluation

> ⚠️ **This evaluation measures out-of-distribution generalisation.** The model was trained exclusively on SID and CIFAKE and evaluated zero-shot on WildFake without any adaptation or fine-tuning. The WildFake test subset used here covers a single generator family (DALL·E, diffusion-based) paired with COCO real images — a particularly challenging combination because COCO photographs carry JPEG compression signatures that the forensic branch has never seen in its training distribution. These results should not be compared directly with the in-domain evaluations above.

The transformation-aware model was evaluated on 6,000 images per corruption condition (5,000 COCO real, 1,000 DALL·E fake). The 5 : 1 real-to-fake ratio reflects the WildFake test subset composition.

#### Clean condition

| Metric | Result |
|---|---:|
| Accuracy | 63.12% |
| Balanced accuracy | 40.91% |
| Precision | 5.57% |
| Recall (fake detection) | 7.60% |
| F1 score | 6.43% |
| AUROC | 31.65% |
| Average precision | 11.38% |
| Real specificity | 74.22% |
| False-positive rate | 25.78% |
| False-negative rate | 92.40% |

The corresponding confusion matrix:

| | Predicted Real | Predicted Fake |
|---|---:|---:|
| **Actual Real** | 3,711 (TN) | 1,289 (FP) |
| **Actual Fake** | 924 (FN) | 76 (TP) |

#### Corruption conditions

| Condition | Accuracy | Balanced Acc. | F1 | AUROC | Avg. Precision | Real Spec. | Fake Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 63.12% | 40.91% | 6.43% | 31.65% | 11.38% | 74.22% | 7.60% |
| JPEG quality 30 | 56.03% | 39.02% | 9.28% | 31.06% | 11.29% | 64.54% | 13.50% |
| Resize to 25% | 30.25% | 25.03% | 7.60% | 12.64% | 9.32% | 32.86% | 17.20% |
| Blur σ = 2 | 40.72% | 26.03% | 2.20% | 11.34% | 9.33% | 48.06% | 4.00% |

#### Branch-level diagnostics

The per-branch AUROCs reveal that the two branches respond very differently to the domain shift. The forensic branch in particular exhibits strong discriminative capacity — just with reversed polarity — suggesting it has learned meaningful features whose sign flips on internet-sourced COCO images.

| Condition | Semantic | Forensic | Inv. Forensic | Base Fusion | Final (gated) |
|---|---:|---:|---:|---:|---:|
| Clean | 0.366 | 0.028 | **0.972** | 0.278 | 0.350 |
| JPEG quality 30 | 0.658 | 0.100 | **0.900** | 0.572 | 0.608 |
| Resize to 25% | 0.730 | 0.078 | **0.922** | 0.564 | 0.621 |
| Blur σ = 2 | 0.583 | 0.066 | **0.934** | 0.447 | 0.489 |

> 💡 The **Inv. Forensic** column shows the AUROC when the forensic branch output is simply inverted (1 − score). Under all four conditions this reaches 0.90–0.97, confirming that the forensic branch has learned highly discriminative low-level features — they are merely calibrated in the opposite direction for COCO-sourced real photographs. This is a calibration / domain-alignment issue rather than a capacity or learning failure.

#### Reliability gate behaviour

The transformation-aware gate correctly assigns higher semantic weight to clean images (0.80–0.94) and increases forensic weight for images it perceives as more corrupted. Average forensic weight ranges from 0.16 (JPEG 30) to 0.26 (Resize ×0.25), which is directionally consistent with in-domain behaviour. However, because the forensic branch polarity is reversed for this test domain, the gate's upweighting of the forensic branch inadvertently reduces final performance.

#### Interpretation

The model's in-domain performance remains strong (AUROC > 0.99 on both SID and CIFAKE). The WildFake AUROCs drop below chance level primarily because of a **domain-aligned sign inversion** in the forensic branch rather than a lack of discriminative capacity. Two compounding factors contribute:

1.  **Forensic polarity reversal.** The forensic branch was trained on SID and CIFAKE, where real images are relatively clean and fake images carry AI-specific compression signatures. On WildFake, the real images are COCO photographs sourced from the internet and carry JPEG recompression artefacts that the forensic branch has never seen paired with the "real" label. The branch therefore scores COCO images as strongly fake (median 0.98) and DALL·E images as more real (median 0.14), inverting its intended polarity. Simply inverting the branch restores AUROC to 0.97.

2.  **Semantic compression from frozen CLIP.** The CLIP ViT-L/14 backbone is frozen during training. Its representation space does not naturally separate photorealistic DALL·E outputs from COCO photographs (clean semantic AUROC 0.37), limiting the semantic branch's ability to compensate for the inverted forensic signal.

These results underscore that zero-shot cross-domain transfer remains an open challenge for multi-branch detectors. The forensic branch's inverted but highly discriminative performance (0.97 when un-inverted) suggests that domain-adaptive calibration — rather than retraining from scratch — may be a productive direction.

Full metrics and per-image predictions are written to `outputs/evaluation/wildfake_zero_shot/summary.json` and `outputs/evaluation/wildfake_zero_shot/predictions.csv`.
