import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import argparse
from utils.utils import seed_everything
from pipeline.kpgt_pipeline import run_kpgt_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run the pretrained KPGT + Morgan Fingerprint pipeline"
    )
    parser.add_argument(
        "--config",
        default="config/kpgt_pretrained_regressor_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Override split type (random / Murcko_scaffold)",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    config_path = args.config
    if args.split:
        from utils.io_tools import load_yaml
        config = load_yaml(config_path)
        config["split"] = args.split
        config["split_path"] = f"data/splits/{config.get('dataset', 'AGILE')}/{args.split}.npy"

        import yaml, tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config, tmp)
        tmp.close()
        config_path = tmp.name

    test_metrics = run_kpgt_pipeline(config_path)

    # Clean up temp file
    if config_path != args.config and os.path.exists(config_path):
        os.unlink(config_path)


if __name__ == "__main__":
    main()
