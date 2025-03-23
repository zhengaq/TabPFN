"""Download all TabPFN model files for offline use."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from tabpfn.model.loading import _user_cache_dir, download_all_models


def main() -> None:
    """Download all TabPFN models and save to cache directory."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    # Get default cache directory using TabPFN's internal function
    cache_dir = _user_cache_dir(platform=sys.platform, appname="tabpfn")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading all models to {cache_dir}")
    download_all_models(cache_dir)
    logger.info(f"All models downloaded to {cache_dir}")


if __name__ == "__main__":
    main()