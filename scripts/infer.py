from __future__ import annotations

import argparse
import json

from pathlib import Path

from src.inference.predictor import (
    AIGCInferenceEngine,
)

from src.utils.config import (
    load_config,
)

from src.utils.device import (
    get_device,
    print_device_info,
)

from src.utils.seed import (
    seed_everything,
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run AIGC image detection "
            "on one image or a directory."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Image file or directory "
            "containing images."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Required prediction JSON "
            "output path."
        ),
    )

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "transformation_aware_best.pt"
        ),
    )

    parser.add_argument(
        "--calibration",
        default=(
            "outputs/calibration/"
            "calibration.json"
        ),
    )

    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help=(
            "Disable temperature calibration."
        ),
    )

    parser.add_argument(
        "--diagnostics-output",
        default=None,
        help=(
            "Optional JSON containing "
            "branch/gating/tile diagnostics."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    seed = int(
        config[
            "project"
        ][
            "seed"
        ]
    )

    seed_everything(
        seed
    )

    device = get_device()

    print_device_info(
        device
    )

    calibration_path = (
        None

        if args.no_calibration

        else args.calibration
    )

    engine = (
        AIGCInferenceEngine(

            config=
                config,

            checkpoint_path=
                args.checkpoint,

            device=
                device,

            calibration_path=
                calibration_path,
        )
    )

    (
        predictions,
        diagnostics,
    ) = (
        engine.predict_directory(

            input_path=
                args.input,

            batch_size=
                args.batch_size,

            num_workers=
                args.num_workers,
        )
    )

    # ==================================================
    # REQUIRED OUTPUT
    # ==================================================

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            predictions,
            file,
            indent=2,
        )

    print(
        "\n========================================"
    )

    print(
        "INFERENCE COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"\nImages: "
        f"{len(predictions):,}"
    )

    print(
        "\nPredictions:"
    )

    print(
        output_path.resolve()
    )

    # ==================================================
    # OPTIONAL DIAGNOSTICS
    # ==================================================

    if (
        args.diagnostics_output
        is not None
    ):

        diagnostics_path = Path(
            args.diagnostics_output
        )

        diagnostics_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with diagnostics_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                diagnostics,
                file,
                indent=2,
            )

        print(
            "\nDiagnostics:"
        )

        print(
            diagnostics_path.resolve()
        )


if __name__ == "__main__":

    main()