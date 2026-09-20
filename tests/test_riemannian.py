import numpy as np

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


def main():
    print("Creating synthetic EEG...")

    rng = np.random.default_rng(42)

    # 10 EEG epochs
    # 2 channels
    # 3000 samples
    X = rng.normal(size=(10, 2, 3000))

    print("EEG shape:", X.shape)

    cov = Covariances(estimator="scm")
    cov_matrices = cov.fit_transform(X)

    print("Covariance shape:", cov_matrices.shape)

    print("\nFirst covariance matrix:")
    print(cov_matrices[0])

    ts = TangentSpace(metric="riemann")
    features = ts.fit_transform(cov_matrices)

    print("\nTangent-space feature shape:", features.shape)

    print("\nFirst feature vector:")
    print(features[0])

    print("\nRIEMANNIAN TEST OK")


if __name__ == "__main__":
    main()