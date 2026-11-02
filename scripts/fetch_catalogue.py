"""Download the KOI cumulative table from the NASA Exoplanet Archive."""

from __future__ import annotations

import argparse

from exoplanet.config import load_config
from exoplanet.data.catalogue import fetch_koi_catalogue, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    config = load_config(args.config)
    csv_path = fetch_koi_catalogue(config, force=args.force)
    print(f"Catalogue: {csv_path}")
    print(f"Metadata:  {metadata_path(csv_path)}")


if __name__ == "__main__":
    main()
