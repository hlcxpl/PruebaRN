import pandas as pd
import numpy as np
from scipy.stats import mode


def bagging_het(X_train, y_train, T, estimators, X_test):
    trained_model = []
    yhat_test = np.zeros((X_test.shape[0], T))
    idx_oob = []
    for t in np.arange(0, T):
        sa1 = X_train.sample(n=X_train.shape[0], replace=True)

        idx_oob = list(set(idx_oob + list(set(X_train.index) - set(sa1.index))))

        idx_estimator = np.random.randint(0, len(estimators))
        estimator = estimators[idx_estimator]

        estimator.fit(sa1, y_train[sa1.index])
        trained_model.append(estimator)

        yhat_test[:, t] = estimator.predict(X_test)

    mode_result = mode(yhat_test, axis=1)[0]
    print(f"Shape of mode_result: {mode_result.shape}")  # Añadido para depuración
    print(mode_result)  # Añadido para depuración
    mode_result_flat = mode_result.ravel()  # Aplanar explícitamente
    print(
        f"Shape of mode_result_flat: {mode_result_flat.shape}"
    )  # Añadido para depuración
    print(mode_result_flat)  # Añadido para depuración
    yhat_out = pd.Series(mode_result_flat, name="yhat")
    print(f"Shape of yhat_out: {yhat_out.shape}")  # Añadido para depuración

    return trained_model, yhat_test, yhat_out, idx_oob


def bagging_het_predict(X, estimators):
    yhat = np.zeros((X.shape[0], len(estimators)))

    for i, est in enumerate(estimators):
        yhat[:, i] = est.predict(X)

    mode_result = mode(yhat, axis=1)[0]
    mode_result_flat = mode_result.ravel()  # Aplanar explícitamente
    return pd.Series(mode_result_flat, name="yhat")
