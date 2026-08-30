# Robustness Evaluation Summary

This note summarizes how the detector performs on unmodified images and after common post-processing operations. Unless stated otherwise, **accuracy** and **AUROC** are reported as percentages, the decision threshold is `0.5`, and label `1` denotes a fake image.

## Clean Images vs. Isolated Transformations

Each robustness run evaluates the clean set plus 14 transformed variants, including JPEG compression, blur, resizing, additive noise, color changes, and sharpening. “Transformed mean” averages the 14 non-clean conditions.

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

WildFake contains 7,000 images per condition (5,000 real and 2,000 fake) in this evaluation. These results measure out-of-domain transfer without retraining.

| WildFake condition | Accuracy | Balanced accuracy | AUROC | Fake recall | False-positive rate |
|---|---:|---:|---:|---:|---:|
| Clean | 63.77 | 48.86 | 44.67 | 14.05 | 16.34 |
| JPEG quality 30 | 69.84 | 60.97 | **66.93** | 40.25 | 18.32 |
| Resize 0.25× | 55.41 | **61.47** | 60.75 | **75.60** | 52.66 |
| Blur radius 2 | 55.21 | 52.60 | 50.00 | 46.50 | 41.30 |

This is the clearest robustness limitation. On clean WildFake, the forensic branch is strongly inverted on the balanced diagnostic subset: its AUROC is 2.78%, while reversing its score gives 97.22%. This points to a dataset-specific forensic cue being interpreted in the wrong direction. The semantic gate reduces some errors, but cannot fully recover reliable zero-shot performance.

## Overall Interpretation

- In-domain performance is high: clean AUROC is approximately 99.7–100%, with strong retention under isolated transformations.
- CIFAKE is most vulnerable to aggressive downsampling and compound reposting; SID is much more stable under the same general stress tests.
- Transformation-aware gating shifts weight toward the semantic branch as corruption becomes stronger. This helps when forensic traces are removed, but can hurt when the semantic branch is confidently wrong.
- Robustness to synthetic corruptions does not guarantee cross-dataset robustness. WildFake reveals severe domain and generator shift that is largely hidden by the in-domain results.

## Evaluation Artifacts

- [Robust-model CIFAKE metrics](outputs/evaluation/robust_model/robust_robustness.csv)
- [Forensic-fusion CIFAKE metrics](outputs/evaluation/forensic_fusion/forensic_fusion_robustness.csv)
- [Native-tile CIFAKE metrics](outputs/evaluation/native_tile/cifake/cifake_robustness.csv)
- [Native-tile SID metrics](outputs/evaluation/native_tile/sid/sid_robustness.csv)
- [Transformation-aware SID metrics](outputs/evaluation/transformation_aware/sid/gating_robustness.csv)
- [CIFAKE laundering metrics](outputs/evaluation/laundering/cifake/laundering_metrics.csv)
- [SID laundering metrics](outputs/evaluation/laundering/sid/laundering_metrics.csv)
- [WildFake summary](outputs/evaluation/wildfake_zero_shot/summary.json)

