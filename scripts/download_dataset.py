#!/usr/bin/env python3
"""
Download Raja Ravi Varma paintings from Wikimedia Commons.

The top-level Raja Ravi Varma category contains subcategories, so this
script recursively searches those subcategories until it has collected
the requested number of images.

Usage:
    python scripts/download_dataset.py --limit 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "src"),
)

from ravi_varma.utils.logging import setup_logging


COMMONS_API = "https://commons.wikimedia.org/w/api.php"

ROOT_CATEGORY = "Category:Paintings by Raja Ravi Varma"

HEADERS = {
    "User-Agent": (
        "TheNextRaviVarma/0.1 "
        "(local research project; educational use)"
    )
}

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
}


def api_get(
    session: requests.Session,
    params: dict,
    logger,
    retries: int = 5,
):
    """
    Make a Wikimedia API request with retry handling for 429/5xx errors.
    """

    for attempt in range(retries):

        try:
            response = session.get(
                COMMONS_API,
                params=params,
                headers=HEADERS,
                timeout=60,
            )

            # Wikimedia rate limit
            if response.status_code == 429:

                retry_after = response.headers.get(
                    "Retry-After",
                    "5",
                )

                try:
                    wait_time = int(retry_after)
                except ValueError:
                    wait_time = 5

                wait_time = max(wait_time, 5)

                logger.warning(
                    "Wikimedia rate limit (429). "
                    "Waiting %s seconds...",
                    wait_time,
                )

                time.sleep(wait_time)
                continue

            # Temporary server errors
            if response.status_code in {
                500,
                502,
                503,
                504,
            }:

                wait_time = 5 * (attempt + 1)

                logger.warning(
                    "Wikimedia server error %s. "
                    "Retrying in %s seconds...",
                    response.status_code,
                    wait_time,
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:

            if attempt == retries - 1:
                raise

            wait_time = 5 * (attempt + 1)

            logger.warning(
                "Request failed: %s. "
                "Retrying in %s seconds...",
                e,
                wait_time,
            )

            time.sleep(wait_time)

    return None


def get_category_contents(
    session: requests.Session,
    category: str,
    logger,
):
    """
    Return all direct files and subcategories inside a Wikimedia category.
    """

    files = []
    subcategories = []

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "file|subcat",
        "cmlimit": "500",
        "format": "json",
    }

    while True:

        data = api_get(
            session,
            params,
            logger,
        )

        if not data:
            break

        members = (
            data
            .get("query", {})
            .get("categorymembers", [])
        )

        for member in members:

            namespace = member.get("ns")
            title = member.get("title")

            # Namespace 6 = File
            if namespace == 6:
                files.append(title)

            # Namespace 14 = Category
            elif namespace == 14:
                subcategories.append(title)

        continuation = data.get("continue")

        if not continuation:
            break

        params.update(continuation)

        # Small delay between API requests
        time.sleep(0.5)

    return files, subcategories


def collect_paintings(
    session: requests.Session,
    root_category: str,
    limit: int,
    logger,
):
    """
    Recursively walk Wikimedia categories and collect file titles.
    """

    queue = [root_category]

    visited_categories = set()
    collected_files = []
    seen_files = set()

    while queue and len(collected_files) < limit:

        category = queue.pop(0)

        if category in visited_categories:
            continue

        visited_categories.add(category)

        logger.info(
            "Scanning category: %s",
            category,
        )

        try:
            files, subcategories = get_category_contents(
                session,
                category,
                logger,
            )

        except requests.RequestException as e:

            logger.warning(
                "Could not scan category '%s': %s",
                category,
                e,
            )

            continue

        # Add files
        for file_title in files:

            if file_title in seen_files:
                continue

            seen_files.add(file_title)
            collected_files.append(file_title)

            logger.info(
                "Found painting %d/%d: %s",
                len(collected_files),
                limit,
                file_title,
            )

            if len(collected_files) >= limit:
                break

        # Add subcategories
        for subcategory in subcategories:

            if subcategory not in visited_categories:
                queue.append(subcategory)

        # Don't hammer Wikimedia
        time.sleep(1)

    logger.info(
        "Collected %d painting files "
        "from %d categories.",
        len(collected_files),
        len(visited_categories),
    )

    return collected_files[:limit]


def get_image_info(
    session: requests.Session,
    titles: list[str],
    logger,
):
    """
    Get image URLs and metadata for multiple files in one API request.
    """

    if not titles:
        return []

    params = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "format": "json",
    }

    data = api_get(
        session,
        params,
        logger,
    )

    if not data:
        return []

    pages = (
        data
        .get("query", {})
        .get("pages", {})
    )

    results = []

    for page in pages.values():

        imageinfo_list = page.get(
            "imageinfo",
            [],
        )

        if not imageinfo_list:
            continue

        imageinfo = imageinfo_list[0]

        url = imageinfo.get("url")

        if not url:
            continue

        ext_meta = imageinfo.get(
            "extmetadata",
            {},
        )

        license_short = (
            ext_meta
            .get("LicenseShortName", {})
            .get("value", "unknown")
        )

        obj_name = (
            ext_meta
            .get("ObjectName", {})
            .get("value", page.get("title", ""))
        )

        results.append(
            {
                "title": page.get("title", ""),
                "url": url,
                "license": license_short,
                "object_name": obj_name,
            }
        )

    return results


def download_image(
    session: requests.Session,
    url: str,
    destination: Path,
    logger,
):
    """
    Download one image with retry handling.
    """

    for attempt in range(5):

        try:

            response = session.get(
                url,
                headers=HEADERS,
                timeout=120,
            )

            if response.status_code == 429:

                retry_after = response.headers.get(
                    "Retry-After",
                    "10",
                )

                try:
                    wait_time = int(retry_after)
                except ValueError:
                    wait_time = 10

                wait_time = max(wait_time, 10)

                logger.warning(
                    "Image server rate limited us. "
                    "Waiting %s seconds...",
                    wait_time,
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()

            if not response.content:
                raise RuntimeError(
                    "Downloaded file is empty."
                )

            destination.write_bytes(
                response.content
            )

            return True

        except (
            requests.RequestException,
            RuntimeError,
        ) as e:

            if attempt == 4:

                logger.warning(
                    "Failed to download %s: %s",
                    url,
                    e,
                )

                return False

            wait_time = 5 * (attempt + 1)

            logger.warning(
                "Download failed: %s. "
                "Retrying in %s seconds...",
                e,
                wait_time,
            )

            time.sleep(wait_time)

    return False


def main() -> int:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Number of images to download.",
    )

    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory for raw images.",
    )

    parser.add_argument(
        "--metadata-file",
        default="data/metadata/metadata.jsonl",
        help="Metadata JSONL file.",
    )

    parser.add_argument(
        "--category",
        default=ROOT_CATEGORY,
        help="Root Wikimedia category.",
    )

    args = parser.parse_args()

    logger = setup_logging(
        "download_dataset"
    )

    raw_dir = Path(
        args.raw_dir
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = Path(
        args.metadata_file
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()

    session.headers.update(
        HEADERS
    )

    # ---------------------------------------------------------
    # STEP 1
    # Find paintings recursively
    # ---------------------------------------------------------

    logger.info(
        "Searching Wikimedia Commons recursively..."
    )

    file_titles = collect_paintings(
        session=session,
        root_category=args.category,
        limit=args.limit,
        logger=logger,
    )

    if not file_titles:

        logger.error(
            "No paintings were found."
        )

        return 1

    logger.info(
        "Found %d candidate paintings.",
        len(file_titles),
    )

    # ---------------------------------------------------------
    # STEP 2
    # Get metadata in batches
    # ---------------------------------------------------------

    image_infos = []

    # Wikimedia allows multiple titles per API call.
    # Use batches of 20 to stay safe.
    batch_size = 20

    for start in range(
        0,
        len(file_titles),
        batch_size,
    ):

        batch = file_titles[
            start:start + batch_size
        ]

        logger.info(
            "Getting metadata for paintings %d-%d...",
            start + 1,
            min(
                start + batch_size,
                len(file_titles),
            ),
        )

        infos = get_image_info(
            session,
            batch,
            logger,
        )

        image_infos.extend(infos)

        time.sleep(1)

    # ---------------------------------------------------------
    # STEP 3
    # Download images
    # ---------------------------------------------------------

    downloaded = 0

    # Start numbering based on existing images.
    existing_images = [
        p
        for p in raw_dir.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in ALLOWED_EXTENSIONS
    ]

    next_index = len(existing_images)

    # Open metadata in append mode.
    with metadata_path.open(
        "a",
        encoding="utf-8",
    ) as meta_f:

        for info in image_infos:

            if downloaded >= args.limit:
                break

            url = info["url"]

            # -------------------------------------------------
            # Correctly extract extension from URL PATH.
            # Query parameters are NOT included.
            # -------------------------------------------------

            parsed_url = urlparse(url)

            suffix = Path(
                parsed_url.path
            ).suffix.lower()

            if suffix not in ALLOWED_EXTENSIONS:
                suffix = ".jpg"

            filename = (
                f"{next_index:04d}{suffix}"
            )

            destination = (
                raw_dir / filename
            )

            logger.info(
                "Downloading %d/%d: %s",
                downloaded + 1,
                args.limit,
                info["title"],
            )

            success = download_image(
                session=session,
                url=url,
                destination=destination,
                logger=logger,
            )

            if not success:
                continue

            # -------------------------------------------------
            # Save metadata
            # -------------------------------------------------

            record = {
                "image": filename,
                "caption": None,
                "artist": "Raja Ravi Varma",
                "source": (
                    "Wikimedia Commons "
                    f"({info['title']}), "
                    f"license: {info['license']}"
                ),
                "year": None,
                "title": info["object_name"],
            }

            meta_f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            meta_f.flush()

            logger.info(
                "Downloaded %s -> %s (%s)",
                info["title"],
                destination,
                info["license"],
            )

            downloaded += 1
            next_index += 1

            # Important: don't trigger Wikimedia's rate limit.
            time.sleep(2)

    # ---------------------------------------------------------
    # STEP 4
    # Summary
    # ---------------------------------------------------------

    logger.info(
        "Downloaded %d/%d images -> %s",
        downloaded,
        args.limit,
        raw_dir,
    )

    logger.info(
        "Metadata written to %s",
        metadata_path,
    )

    if downloaded > 0:

        logger.info(
            "Next: python scripts/validate_dataset.py"
        )

        return 0

    logger.error(
        "No images were downloaded."
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )