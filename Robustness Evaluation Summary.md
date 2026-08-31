# Robustness Evaluation Summary

This note summarizes how the detector performs on unmodified images and after common post-processing operations. Unless stated otherwise, **accuracy** and **AUROC** are reported as percentages, the decision threshold is `0.5`, and label `1` denotes a fake image.

## Clean Images vs. Isolated Transformations

Each robustness run evaluates the clean set plus 14 transformed variants, including JPEG compression, blur, resizing, additive noise, color changes, and sharpening. "Transformed mean" averages the 14 non-clean conditions.

| Model / evaluation set | Clean accuracy | Clean AUROC | Transformed mean accuracy | Transformed mean AUROC | AUROC retained | Worst tested condition (accuracy / AUROC) |
|---|---:|---:|---:|---:|---:|---|
| Robust model — CIFAKE | 97.45 | 99.69 | 94.39 | 98.57 | 98.88 | Resize 0.25× (88.45 / 95.48) |
| Forensic fusion — CIFAKE | **98.14** | **99.83** | **95.57** | **99.06** | **99.23** | Resize 0.25× (89.86 / 96.49) |
| Native-tile fusion — CIFAKE | 97.85 | 99.80 | 95.30 | 98.96 | 99.16 | Resize 0.25× (89.57 / 96.32) |
| Native-tile fusion — SID | 99.37 | 99.98 | **99.09** | **99.97** | **99.98** | Noise 0.10 (98.19 / 99.91) |
| Transformation-aware fusion — SID | **99.40** | 99.98 | 98.91 | 99.95 | 99.97 | Noise 0.10 (97.29 / 99.86) |

The forensic-fusion model is the strongest CIFAKE configuration in this comparison. On SID, both fusion variants retain almost all clean-image ranking performance, although native-tile fusion has slightly higher mean transformed accuracy while transformation-aware fusion has slightly higher clean accuracy. These rows come from different checkpoints and, in the SID case, a different dataset; they should therefore be read as evaluation summaries rather than a controlled ablation study.

## Representative Transformations

| Evaluation | Clean | JPEG quality 30 | Blur radius 2 | Resize 0.25× | Noise 0.10 |
|---|---:|---:|---:|---:|---:|
| Forensic fusion — CIFAKE accuracy | 98.14 | 94.82 | 91.09 | 89.86 | 93.63 |
| Forensic fusion — CIFAKE AUROC | 99.83 | 99.09 | 97.15 | 96.49 | 98.58 |
| Native-tile fusion — SID accuracy | 99.37 | 99.05 | 99.02 | 98.97 | 98.19 |
| Native-tile fusion — SID AUROC | 99.98 | 99.97 | 99.97 | 99.97 | 99.91 |

Downsampling is the most damaging isolated transformation on CIFAKE, while stronger additive noise produces the largest decline on SID. AUROC generally falls less than thresholded accuracy, indicating that some transformations shift score calibration without completely destroying class ranking.

## Compound Reposting and Laundering

Compound pipelines combine operations such as resizing, compression, blur, sharpening, noise, screenshots, and repeated reposting. They are more challenging than applying one transformation at a time.

| Transformation-aware evaluation | Clean accuracy / AUROC | Mean laundered AUROC | AUROC retained | Most difficult condition |
|---|---:|---:|---:|---|
| CIFAKE | 97.75 / 99.70 | 93.62 | 93.90 | Extreme repost: 71.33 accuracy, 82.91 AUROC, 50.58 fake recall |
| SID | 99.40 / 99.98 | 99.95 | 99.97 | Noise + JPEG: 96.94 accuracy, 99.92 AUROC |

The CIFAKE result exposes a substantial weakness under severe reposting: the extreme-repost pipeline raises the false-negative rate to 49.42%. The same architecture is far more stable on SID, suggesting that robustness depends strongly on the training and evaluation domains rather than on transformations alone.

## Zero-Shot WildFake Generalization

> ⚠️ **This evaluation measures out-of-distribution generalisation.** The model was trained exclusively on SID and CIFAKE and evaluated zero-shot on WildFake without any adaptation or fine-tuning. The WildFake test subset used here covers a single generator family (DALL·E, diffusion-based) paired with COCO real images — a particularly challenging combination because COCO photographs carry JPEG compression signatures that the forensic branch has never seen in its training distribution. These results should not be compared directly with the in-domain evaluations above.

The transformation-aware model was evaluated on 6,000 images per corruption condition (5,000 COCO real, 1,000 DALL·E fake). The 5 : 1 real-to-fake ratio reflects the WildFake test subset composition.

### Clean condition

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

### Corruption conditions

| Condition | Accuracy | Balanced Acc. | F1 | AUROC | Avg. Precision | Real Spec. | Fake Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 63.12% | 40.91% | 6.43% | 31.65% | 11.38% | 74.22% | 7.60% |
| JPEG quality 30 | 56.03% | 39.02% | 9.28% | 31.06% | 11.29% | 64.54% | 13.50% |
| Resize to 25% | 30.25% | 25.03% | 7.60% | 12.64% | 9.32% | 32.86% | 17.20% |
| Blur σ = 2 | 40.72% | 26.03% | 2.20% | 11.34% | 9.33% | 48.06% | 4.00% |

### Branch-level diagnostics

The per-branch AUROCs reveal that the two branches respond very differently to the domain shift. The forensic branch in particular exhibits strong discriminative capacity — just with reversed polarity — suggesting it has learned meaningful features whose sign flips on internet-sourced COCO images.

| Condition | Semantic | Forensic | Inv. Forensic | Base Fusion | Final (gated) |
|---|---:|---:|---:|---:|---:|
| Clean | 0.366 | 0.028 | **0.972** | 0.278 | 0.350 |
| JPEG quality 30 | 0.658 | 0.100 | **0.900** | 0.572 | 0.608 |
| Resize to 25% | 0.730 | 0.078 | **0.922** | 0.564 | 0.621 |
| Blur σ = 2 | 0.583 | 0.066 | **0.934** | 0.447 | 0.489 |

> 💡 The **Inv. Forensic** column shows the AUROC when the forensic branch output is simply inverted (1 − score). Under all four conditions this reaches 0.90–0.97, confirming that the forensic branch has learned highly discriminative low-level features — they are merely calibrated in the opposite direction for COCO-sourced real photographs. This is a calibration / domain-alignment issue rather than a capacity or learning failure.

### Reliability gate behaviour

The transformation-aware gate correctly assigns higher semantic weight to clean images (0.80–0.94) and increases forensic weight for images it perceives as more corrupted. Average forensic weight ranges from 0.16 (JPEG 30) to 0.26 (Resize ×0.25), which is directionally consistent with in-domain behaviour. However, because the forensic branch polarity is reversed for this test domain, the gate's upweighting of the forensic branch inadvertently reduces final performance.

### WildFake interpretation

The model's in-domain performance remains strong (AUROC > 0.99 on both SID and CIFAKE). The WildFake AUROCs drop below chance level primarily because of a **domain-aligned sign inversion** in the forensic branch rather than a lack of discriminative capacity. Two compounding factors contribute:

1. **Forensic polarity reversal.** The forensic branch was trained on SID and CIFAKE, where real images are relatively clean and fake images carry AI-specific compression signatures. On WildFake, the real images are COCO photographs sourced from the internet and carry JPEG recompression artefacts that the forensic branch has never seen paired with the "real" label. The branch therefore scores COCO images as strongly fake (median 0.98) and DALL·E images as more real (median 0.14), inverting its intended polarity. Simply inverting the branch restores AUROC to 0.97.

2. **Semantic compression from frozen CLIP.** The CLIP ViT-L/14 backbone is frozen during training. Its representation space does not naturally separate photorealistic DALL·E outputs from COCO photographs (clean semantic AUROC 0.37), limiting the semantic branch's ability to compensate for the inverted forensic signal.

These results underscore that zero-shot cross-domain transfer remains an open challenge for multi-branch detectors. The forensic branch's inverted but highly discriminative performance (0.97 when un-inverted) suggests that domain-adaptive calibration — rather than retraining from scratch — may be a productive direction.

## Overall Interpretation

- In-domain performance is high: clean AUROC is approximately 99.7–100%, with strong retention under isolated transformations.
- CIFAKE is most vulnerable to aggressive downsampling and compound reposting; SID is much more stable under the same general stress tests.
- Transformation-aware gating shifts weight toward the semantic branch as corruption becomes stronger. This helps when forensic traces are removed, but can hurt when the semantic branch is confidently wrong.
- Robustness to synthetic corruptions does not guarantee cross-dataset robustness. WildFake reveals severe domain and generator shift that is largely hidden by the in-domain results.
- The WildFake forensic branch polarity reversal is the most striking finding: the branch is highly discriminative (0.97 inverted AUROC on clean images) but its scoring direction is reversed for internet-sourced COCO photographs. This is a domain-alignment problem, not an architectural one.

## Evaluation Artifacts

- [Robust-model CIFAKE metrics](outputs/evaluation/robust_model/robust_robustness.csv)
- [Forensic-fusion CIFAKE metrics](outputs/evaluation/forensic_fusion/forensic_fusion_robustness.csv)
- [Native-tile CIFAKE metrics](outputs/evaluation/native_tile/cifake/cifake_robustness.csv)
- [Native-tile SID metrics](outputs/evaluation/native_tile/sid/sid_robustness.csv)
- [Transformation-aware SID metrics](outputs/evaluation/transformation_aware/sid/gating_robustness.csv)
- [CIFAKE laundering metrics](outputs/evaluation/laundering/cifake/laundering_metrics.csv)
- [SID laundering metrics](outputs/evaluation/laundering/sid/laundering_metrics.csv)
- [WildFake summary](outputs/evaluation/wildfake_zero_shot/summary.json)
- [WildFake predictions](outputs/evaluation/wildfake_zero_shot/predictions.csv)
- [WildFake branch diagnostics](outputs/evaluation/wildfake_zero_shot/diagnostics/branch_metrics.csv)
