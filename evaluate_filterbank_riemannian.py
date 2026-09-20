import numpy as np

from scipy.signal import butter, sosfiltfilt

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from src.config import SLEEP_EDF_DIR
from src.data.sleep_dataset import (
    find_recordings,
    split_subjects,
    load_subject_data,
)


STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

FS = 100.0

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}


def bandpass(X, low, high, fs):
    sos = butter(
        4,
        [low, high],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    return sosfiltfilt(
        sos,
        X,
        axis=-1,
    )


def make_strictly_spd(covariances, min_eigenvalue=1e-4):
    repaired = np.empty_like(covariances)

    for i, cov in enumerate(covariances):
        cov = (cov + cov.T) / 2.0

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        eigenvalues = np.clip(
            eigenvalues,
            min_eigenvalue,
            None,
        )

        cov_spd = (
            eigenvectors
            @ np.diag(eigenvalues)
            @ eigenvectors.T
        )

        cov_spd = (cov_spd + cov_spd.T) / 2.0

        repaired[i] = cov_spd

    return repaired


def extract_filterbank_riemannian_features(
    X_train,
    X_test,
):
    train_features = []
    test_features = []

    for band_name, (low, high) in BANDS.items():

        print(
            f"\nBand: {band_name} "
            f"{low}-{high} Hz"
        )

        X_train_band = bandpass(
            X_train,
            low,
            high,
            FS,
        )

        X_test_band = bandpass(
            X_test,
            low,
            high,
            FS,
        )

        covariance = Covariances(
            estimator="oas"
        )

        train_cov = covariance.fit_transform(
            X_train_band
        )

        test_cov = covariance.transform(
            X_test_band
        )

        train_cov = make_strictly_spd(
            train_cov
        )

        test_cov = make_strictly_spd(
            test_cov
        )

        tangent_space = TangentSpace(
            metric="riemann"
        )

        train_ts = tangent_space.fit_transform(
            train_cov
        )

        test_ts = tangent_space.transform(
            test_cov
        )

        print(
            "Train tangent shape:",
            train_ts.shape,
        )

        train_features.append(train_ts)
        test_features.append(test_ts)

    X_train_features = np.concatenate(
        train_features,
        axis=1,
    )

    X_test_features = np.concatenate(
        test_features,
        axis=1,
    )

    return X_train_features, X_test_features


def main():

    print("Loading recordings...")

    files = find_recordings(
        SLEEP_EDF_DIR
    )

    train_subjects, val_subjects, test_subjects = split_subjects(
        files
    )

    print("Train subjects:", len(train_subjects))
    print("Validation subjects:", len(val_subjects))
    print("Test subjects:", len(test_subjects))

    print("\nLoading train data...")

    X_train, y_train = load_subject_data(
        files,
        train_subjects,
    )

    print("\nLoading test data...")

    X_test, y_test = load_subject_data(
        files,
        test_subjects,
    )

    print("\nTrain:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    print(
        "\nExtracting Filter-Bank "
        "Riemannian features..."
    )

    X_train_features, X_test_features = (
        extract_filterbank_riemannian_features(
            X_train,
            X_test,
        )
    )

    print(
        "\nFinal train feature shape:",
        X_train_features.shape,
    )

    print(
        "Final test feature shape :",
        X_test_features.shape,
    )

    print(
        "NaN:",
        np.isnan(X_train_features).sum(),
    )

    print(
        "Inf:",
        np.isinf(X_train_features).sum(),
    )

    print(
        "\nTraining Logistic Regression..."
    )

    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
    )

    classifier.fit(
        X_train_features,
        y_train,
    )

    print("Predicting...")

    y_pred = classifier.predict(
        X_test_features
    )

    print("\n========================================")
    print("FILTER-BANK RIEMANNIAN REPORT")
    print("========================================")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=STAGE_NAMES,
            digits=3,
            zero_division=0,
        )
    )

    print("========================================")
    print("CONFUSION MATRIX")
    print("========================================")

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    print("\nRows = TRUE")
    print("Columns = PREDICTED\n")

    print(
        "        Wake     N1      N2      N3     REM"
    )

    for name, row in zip(STAGE_NAMES, cm):

        values = " ".join(
            f"{value:7d}"
            for value in row
        )

        print(
            f"{name:5s} {values}"
        )


if __name__ == "__main__":
    main()