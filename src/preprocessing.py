from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class ICUPreprocessor:
    """Preprocess raw WiDS ICU rows into the final model feature schema."""

    id_cols: list[str] = field(
        default_factory=lambda: ["encounter_id", "patient_id", "hospital_id"]
    )
    leakage_cols: list[str] = field(
        default_factory=lambda: [
            "apache_4a_hospital_death_prob",
            "apache_4a_icu_death_prob",
        ]
    )
    target_col: str = "hospital_death"
    high_missing_threshold: float = 50.0

    high_missing_cols_: list[str] = field(default_factory=list, init=False)
    missing_indicator_cols_: list[str] = field(default_factory=list, init=False)
    numeric_cols_: list[str] = field(default_factory=list, init=False)
    categorical_cols_: list[str] = field(default_factory=list, init=False)
    train_medians_: pd.Series | None = field(default=None, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, raw_df: pd.DataFrame) -> "ICUPreprocessor":
        """Learn preprocessing decisions from the training split only."""
        X = self._drop_initial_columns(raw_df)

        train_missing_pct = X.isnull().sum() / len(X) * 100
        self.high_missing_cols_ = (
            train_missing_pct[train_missing_pct > self.high_missing_threshold]
            .index.tolist()
        )

        X = X.drop(columns=self.high_missing_cols_)
        self.missing_indicator_cols_ = [c for c in X.columns if X[c].isnull().any()]

        X = self._add_missing_indicators(X)

        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(include="object").columns.tolist()
        self.train_medians_ = X[self.numeric_cols_].median()

        X = self._impute(X)
        X = pd.get_dummies(X, columns=self.categorical_cols_, drop_first=False)

        self.feature_names_ = X.columns.tolist()
        self.fitted_ = True
        return self

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing decisions to raw validation/test/new rows."""
        self._check_is_fitted()

        X = self._drop_initial_columns(raw_df)
        X = X.drop(columns=self.high_missing_cols_, errors="ignore")
        X = self._add_missing_indicators(X)
        X = self._impute(X)
        X = pd.get_dummies(X, columns=self.categorical_cols_, drop_first=False)

        X = X.reindex(columns=self.feature_names_, fill_value=0)
        return X

    def fit_transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on raw training rows and return the processed training matrix."""
        return self.fit(raw_df).transform(raw_df)

    def _drop_initial_columns(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = self.id_cols + self.leakage_cols + [self.target_col]
        return raw_df.drop(columns=drop_cols, errors="ignore").copy()

    def _add_missing_indicators(self, X: pd.DataFrame) -> pd.DataFrame:
        indicators = {
            f"{col}_missing": X[col].isnull().astype(int)
            for col in self.missing_indicator_cols_
            if col in X.columns
        }
        if not indicators:
            return X
        return pd.concat([X, pd.DataFrame(indicators, index=X.index)], axis=1)

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.train_medians_ is None:
            raise RuntimeError("ICUPreprocessor must be fitted before imputation.")

        numeric_cols = [c for c in self.numeric_cols_ if c in X.columns]
        categorical_cols = [c for c in self.categorical_cols_ if c in X.columns]

        X[numeric_cols] = X[numeric_cols].fillna(self.train_medians_)
        X[categorical_cols] = X[categorical_cols].fillna("Unknown")
        return X

    def _check_is_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("ICUPreprocessor is not fitted. Call fit() first.")


def load_feature_names(path: str) -> list[str]:
    """Load one-column feature name files saved without a header."""
    return pd.read_csv(path, header=None)[0].tolist()


def align_to_features(X: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    """Align a processed frame to an expected feature list."""
    return X.reindex(columns=list(feature_names), fill_value=0)
