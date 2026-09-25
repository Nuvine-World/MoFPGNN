import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import argparse
from utils.utils import seed_everything
from pipeline.kpgt_pipeline import run_kpgt_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run the Morgan-fingerprint MLP baselines (CMF-only / BMF-only)"
    )
    parser.add_argument(
        "--config",
        default="config/morgan_only_config.yaml",
        help="Path to YAML config (morgan_only_config.yaml = CMF-only, "
             "morgan_binary_only_config.yaml = BMF-only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed in the config file (default: use config's `seed`, or 42)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Override split type (random)",
    )
    args = parser.parse_args()

    from utils.io_tools import load_yaml
    config = load_yaml(args.config)

    # Seed precedence: CLI --seed > config `seed:` > 42.
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    seed_everything(seed)
    print(f"Using seed: {seed}")

    config_path = args.config
    # The trainer reads the seed from the config, so always write it through.
    config["seed"] = seed
    overridden = True
    if args.split:
        config["split"] = args.split
        split_path = f"data/splits/{config.get('dataset', 'AGILE')}/{args.split}.npy"
        if not os.path.exists(split_path):
            import glob
            avail = sorted(os.path.basename(p)[:-4] for p in
                           glob.glob(f"data/splits/{config.get('dataset', 'AGILE')}/*.npy"))
            raise SystemExit(
                f"ERROR: unknown split '{args.split}' (no file at {split_path}).\n"
                f"       Available splits: {', '.join(avail) or 'none'}")
        config["split_path"] = split_path
        overridden = True

    if overridden:
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
