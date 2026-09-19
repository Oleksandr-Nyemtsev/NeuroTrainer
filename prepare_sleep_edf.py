from pathlib import Path
import gc

import numpy as np
from mne.datasets.sleep_physionet.age import fetch_data

from src.data.sleep_edf import preprocess_sleep_edf_recording


PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sleep_edf"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Sleep-EDF subject IDs
SUBJECTS = [
    subject
    for subject in range(83)
    if subject not in [39, 68, 69, 78, 79]
]

RECORDINGS = [1, 2]


def main():

    print("=" * 60)
    print("NeuroTrainer - Sleep-EDF streaming preprocessing")
    print("=" * 60)

    print("Subjects:", len(SUBJECTS))
    print("Recordings:", RECORDINGS)
    print("Output:", OUTPUT_DIR)
    print()

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    total_epochs = 0

    total_jobs = len(SUBJECTS) * len(RECORDINGS)
    job_number = 0

    for subject in SUBJECTS:

        for recording in RECORDINGS:

            job_number += 1

            print()
            print("=" * 60)
            print(
                f"[{job_number}/{total_jobs}] "
                f"Subject {subject:02d} | "
                f"Recording {recording}"
            )

            try:

                # STEP 1:
                # Download / find ONE recording only
                files = fetch_data(
                    subjects=[subject],
                    recording=[recording],
                    on_missing="warn",
                )

                if len(files) == 0:

                    print("Recording unavailable - skipping.")

                    skipped_count += 1

                    continue

                psg_file, hypnogram_file = files[0]

                psg_path = Path(psg_file)

                recording_name = (
                    psg_path
                    .stem
                    .replace("-PSG", "")
                )

                output_file = (
                    OUTPUT_DIR
                    / f"{recording_name}.npz"
                )


                # STEP 2:
                # Do not process twice
                if output_file.exists():

                    print(
                        f"Already processed: "
                        f"{output_file.name}"
                    )

                    skipped_count += 1

                    continue


                # STEP 3:
                # Preprocess ONE night
                print("Preprocessing...")

                X, y = preprocess_sleep_edf_recording(
                    psg_file=psg_file,
                    hypnogram_file=hypnogram_file,
                )


                # STEP 4:
                # Save processed recording
                np.savez_compressed(
                    output_file,
                    X=X,
                    y=y,
                    subject_id=subject,
                    recording_id=recording,
                    recording_name=recording_name,
                )


                epochs_count = len(y)

                processed_count += 1
                total_epochs += epochs_count


                labels, counts = np.unique(
                    y,
                    return_counts=True,
                )

                class_counts = dict(
                    zip(
                        labels.tolist(),
                        counts.tolist(),
                    )
                )


                print(
                    f"Saved: {output_file.name}"
                )

                print(
                    f"X shape: {X.shape}"
                )

                print(
                    f"y shape: {y.shape}"
                )

                print(
                    f"Epochs: {epochs_count}"
                )

                print(
                    f"Class counts: {class_counts}"
                )


            except KeyboardInterrupt:

                print()
                print("STOP requested by user.")
                print("Finished files are safe.")
                print("Run the script again to continue.")

                return


            except Exception as error:

                failed_count += 1

                print(
                    f"FAILED: "
                    f"subject={subject}, "
                    f"recording={recording}"
                )

                print(
                    f"Reason: {error}"
                )


            finally:

                if "X" in locals():

                    del X

                if "y" in locals():

                    del y

                gc.collect()


    print()
    print("=" * 60)
    print("PREPROCESSING FINISHED")
    print("=" * 60)

    print(
        f"Processed recordings: {processed_count}"
    )

    print(
        f"Skipped recordings: {skipped_count}"
    )

    print(
        f"Failed recordings: {failed_count}"
    )

    print(
        f"Total epochs: {total_epochs}"
    )

    print(
        f"Saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()