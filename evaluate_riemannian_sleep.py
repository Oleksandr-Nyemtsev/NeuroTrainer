import numpy as np

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


def make_strictly_spd(covariances, min_eigenvalue=1e-4):
    """
    Force every covariance matrix to be strictly
    symmetric positive definite.
    """

    repaired = np.empty_like(covariances)

    repaired_count = 0
    minimum_before = np.inf
    minimum_after = np.inf

    for i, cov in enumerate(covariances):

        # Numerical symmetry
        cov = (cov + cov.T) / 2.0

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        minimum_before = min(
            minimum_before,
            float(eigenvalues.min()),
        )

        if eigenvalues.min() < min_eigenvalue:
            repaired_count += 1

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

        new_eigenvalues = np.linalg.eigvalsh(cov_spd)

        minimum_after = min(
            minimum_after,
            float(new_eigenvalues.min()),
        )

    print("Matrices repaired:", repaired_count)
    print("Minimum eigenvalue before:", minimum_before)
    print("Minimum eigenvalue after :", minimum_after)

    return repaired


def main():

    print("Loading recordings...")

    files = find_recordings(SLEEP_EDF_DIR)

    train_subjects, val_subjects, test_subjects = split_subjects(files)

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

    # --------------------------------------------------
    # COVARIANCE
    # --------------------------------------------------

    print("\nCalculating covariance matrices...")

    covariance_estimator = Covariances(
        estimator="oas",
    )

    train_cov = covariance_estimator.fit_transform(
        X_train
    )

    test_cov = covariance_estimator.transform(
        X_test
    )

    print("Train covariance:", train_cov.shape)
    print("Test covariance :", test_cov.shape)

    # --------------------------------------------------
    # FORCE STRICT SPD
    # --------------------------------------------------

    print("\nChecking TRAIN covariance matrices...")

    train_cov = make_strictly_spd(
        train_cov,
        min_eigenvalue=1e-4,
    )

    print("\nChecking TEST covariance matrices...")

    test_cov = make_strictly_spd(
        test_cov,
        min_eigenvalue=1e-4,
    )

    # --------------------------------------------------
    # TANGENT SPACE
    # --------------------------------------------------

    print("\nFitting Riemannian Tangent Space...")

    tangent_space = TangentSpace(
        metric="riemann"
    )

    X_train_riemann = tangent_space.fit_transform(
        train_cov
    )

    X_test_riemann = tangent_space.transform(
        test_cov
    )

    print(
        "Train Riemannian features:",
        X_train_riemann.shape,
    )

    print(
        "Test Riemannian features :",
        X_test_riemann.shape,
    )

    print(
        "NaN features:",
        np.isnan(X_train_riemann).sum(),
    )

    print(
        "Inf features:",
        np.isinf(X_train_riemann).sum(),
    )

    # --------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------

    print("\nTraining Logistic Regression...")

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )

    classifier.fit(
        X_train_riemann,
        y_train,
    )

    print("Predicting...")

    y_pred = classifier.predict(
        X_test_riemann
    )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print("\n========================================")
    print("RIEMANNIAN CLASSIFICATION REPORT")
    print("========================================")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=STAGE_NAMES,
            digits=3,
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