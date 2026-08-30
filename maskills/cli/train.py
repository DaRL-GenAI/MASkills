"""CLI entry point: ``maskills train --config configs/...json``."""

import argparse


def _parse_overrides(overrides: list | None) -> dict:
    """Turn ``key=value`` strings into a dict, coercing numeric values."""
    parsed = {}
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        for cast in (int, float):
            try:
                val = cast(val)
                break
            except ValueError:
                continue
        parsed[key] = val
    return parsed


def main():
    parser = argparse.ArgumentParser(description="MASkills training CLI")
    parser.add_argument("command", choices=["train"], help="Command to run")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument(
        "--override",
        nargs="*",
        help="Key=value overrides (e.g. num_iterations=10)",
    )
    args = parser.parse_args()

    if args.command == "train":
        import maskills

        maskills.train(args.config, **_parse_overrides(args.override))
        print("\nTraining complete!")


if __name__ == "__main__":
    main()
