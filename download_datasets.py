from pathlib import Path
import json
import re
import sys
import time
import urllib.request


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS_DIR = PROJECT_ROOT / "data" / "external"

SLEEP_EDF_ST_DIR = DATASETS_DIR / "sleep_edf_telemetry"
MUSE2_DIR = DATASETS_DIR / "muse2_cognitive"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
SLEEP_EDF_ST_DIR.mkdir(parents=True, exist_ok=True)
MUSE2_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

USER_AGENT = "NeuroTrainer/0.1 dataset downloader"


def download_file(url, destination, retries=5):
    destination = Path(destination)

    if destination.exists() and destination.stat().st_size > 0:
        print(f"[SKIP] {destination.name}")
        return

    print(f"[DOWNLOAD] {destination.name}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                total = response.headers.get("Content-Length")

                if total:
                    total = int(total)
                else:
                    total = None

                temp_path = destination.with_suffix(
                    destination.suffix + ".part"
                )

                downloaded = 0

                with open(temp_path, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)

                        if not chunk:
                            break

                        f.write(chunk)

                        downloaded += len(chunk)

                        if total:
                            percent = downloaded / total * 100

                            print(
                                f"\r    "
                                f"{downloaded / 1024 / 1024:.1f} MB"
                                f" / "
                                f"{total / 1024 / 1024:.1f} MB"
                                f"  ({percent:.1f}%)",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\r    "
                                f"{downloaded / 1024 / 1024:.1f} MB",
                                end="",
                                flush=True,
                            )

                print()

                temp_path.replace(destination)

                return

        except Exception as exc:
            print(
                f"\n[ERROR] attempt {attempt}/{retries}: {exc}"
            )

            if attempt == retries:
                raise

            time.sleep(3)


def read_text_url(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )

    with urllib.request.urlopen(
        request,
        timeout=60,
    ) as response:
        return response.read().decode(
            "utf-8",
            errors="ignore",
        )


# ============================================================
# 1. SLEEP-EDF TELEMETRY
# ============================================================

def download_sleep_edf_telemetry():
    print("\n" + "=" * 70)
    print("SLEEP-EDF TELEMETRY")
    print("=" * 70)

    base_url = (
        "https://physionet.org/files/"
        "sleep-edfx/1.0.0/sleep-telemetry/"
    )

    print("Reading PhysioNet directory...")

    html = read_text_url(base_url)

    filenames = sorted(
        set(
            re.findall(
                r'href="([^"]+\.edf)"',
                html,
                flags=re.IGNORECASE,
            )
        )
    )

    # Only ST telemetry files.
    filenames = [
        name
        for name in filenames
        if name.upper().startswith("ST")
    ]

    print(
        f"Found {len(filenames)} EDF files."
    )

    for index, filename in enumerate(
        filenames,
        start=1,
    ):
        print(
            f"\n[{index}/{len(filenames)}]"
        )

        url = base_url + filename

        destination = (
            SLEEP_EDF_ST_DIR
            / filename
        )

        download_file(
            url,
            destination,
        )

    print(
        "\nSleep-EDF Telemetry finished."
    )


# ============================================================
# 2. MUSE 2 COGNITIVE DATASET
# ============================================================

def download_muse2_dataset():
    print("\n" + "=" * 70)
    print("MUSE 2 COGNITIVE DATASET")
    print("=" * 70)

    record_id = "20089290"

    api_url = (
        f"https://zenodo.org/api/records/"
        f"{record_id}"
    )

    print("Reading Zenodo metadata...")

    metadata_text = read_text_url(
        api_url
    )

    metadata = json.loads(
        metadata_text
    )

    files = metadata.get(
        "files",
        []
    )

    print(
        f"Found {len(files)} files."
    )

    for index, file_info in enumerate(
        files,
        start=1,
    ):
        filename = file_info["key"]

        links = file_info.get(
            "links",
            {}
        )

        url = (
            links.get("content")
            or links.get("self")
        )

        if not url:
            print(
                f"[WARNING] No URL for {filename}"
            )
            continue

        print(
            f"\n[{index}/{len(files)}]"
        )

        destination = (
            MUSE2_DIR
            / filename
        )

        download_file(
            url,
            destination,
        )

    print(
        "\nMuse 2 dataset finished."
    )


# ============================================================
# 3. DATASETS THAT REQUIRE MANUAL ACCESS
# ============================================================

def print_manual_datasets():
    print("\n" + "=" * 70)
    print("MANUAL / AUTHENTICATED DATASETS")
    print("=" * 70)

    print(
        "\nISRUC-SLEEP"
    )

    print(
        "Official downloads are hosted on MEGA."
    )

    print(
        "Cohort I:   14.12 GB"
    )

    print(
        "Cohort II:   2.2 GB"
    )

    print(
        "Cohort III:  1.35 GB"
    )

    print(
        "\nOfficial page:"
    )

    print(
        "https://sleeptight.isr.uc.pt/?page_id=48"
    )

    print(
        "\nMUSE SLEEP-ONSET 2026"
    )

    print(
        "Requires Codabench registration "
        "and acceptance of competition terms."
    )

    print(
        "After registration we will download it "
        "using the official NeuralBench setup."
    )

    print(
        "\nCompetition:"
    )

    print(
        "https://www.codabench.org/competitions/17983/"
    )


# ============================================================
# SUMMARY
# ============================================================

def show_summary():
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    for folder in [
        SLEEP_EDF_ST_DIR,
        MUSE2_DIR,
    ]:
        files = [
            f
            for f in folder.rglob("*")
            if f.is_file()
            and not f.name.endswith(".part")
        ]

        size_bytes = sum(
            f.stat().st_size
            for f in files
        )

        print(
            f"{folder.name}: "
            f"{len(files)} files, "
            f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("NEUROTRAINER DATASET DOWNLOADER")
    print("=" * 70)

    print(
        "Destination:",
        DATASETS_DIR,
    )

    try:
        download_sleep_edf_telemetry()
    except Exception as exc:
        print(
            "\nSleep-EDF download failed:",
            exc,
        )

    try:
        download_muse2_dataset()
    except Exception as exc:
        print(
            "\nMuse 2 download failed:",
            exc,
        )

    print_manual_datasets()

    show_summary()

    print("\nDONE")


if __name__ == "__main__":
    main()