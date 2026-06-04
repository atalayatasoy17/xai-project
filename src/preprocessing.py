from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


@dataclass
class ICUPreprocessor:
    """Preprocess raw WiDS ICU rows into the final model feature schema.

    This production wrapper mirrors the model-experiment preprocessing:
    numeric features are median-imputed, missingness indicators are added for
    numeric missing values, binary features are ordinal-encoded, and remaining
    categorical features are one-hot encoded with infrequent-category handling.
    Numeric scaling is intentionally omitted because LightGBM is tree-based and
    evidence packets should keep values clinically readable.
    """

    id_cols: list[str] = field(
        default_factory=lambda: ["encounter_id", "patient_id", "hospital_id", "icu_id"]
    )
    leakage_cols: list[str] = field(
        default_factory=lambda: [
            "apache_4a_hospital_death_prob",
            "apache_4a_icu_death_prob",
        ]
    )
    target_col: str = "hospital_death"
    ohe_max_categories: int = 10

    numeric_cols_: list[str] = field(default_factory=list, init=False)
    binary_cols_: list[str] = field(default_factory=list, init=False)
    categorical_cols_: list[str] = field(default_factory=list, init=False)
    ordinal_cols_: list[str] = field(default_factory=list, init=False)
    missing_indicator_cols_: list[str] = field(default_factory=list, init=False)
    high_missing_cols_: list[str] = field(default_factory=list, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    preprocessor_: ColumnTransformer | None = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, raw_df: pd.DataFrame) -> "ICUPreprocessor":
        """Learn preprocessing decisions from the training split only."""
        X = self._drop_initial_columns(raw_df)
        self._learn_column_groups(X)

        numeric_tf = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ]
        )
        binary_tf = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        ohe_tf = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ohe",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        sparse_output=False,
                        max_categories=self.ohe_max_categories,
                    ),
                ),
            ]
        )

        self.preprocessor_ = ColumnTransformer(
            transformers=[
                ("num", numeric_tf, self.numeric_cols_),
                ("bin", binary_tf, self.binary_cols_),
                ("ohe", ohe_tf, self.categorical_cols_),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )
        self.preprocessor_.fit(X)
        self.feature_names_ = self._clean_feature_names(
            self.preprocessor_.get_feature_names_out()
        )
        self.fitted_ = True
        return self

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing decisions to raw validation/test/new rows."""
        self._check_is_fitted()
        if self.preprocessor_ is None:
            raise RuntimeError("ICUPreprocessor transformer is missing.")

        X = self._drop_initial_columns(raw_df)
        transformed = self.preprocessor_.transform(X)

        return pd.DataFrame(
            transformed,
            columns=self.feature_names_,
            index=raw_df.index,
        )

    def get_display_values(self, raw_df: pd.DataFrame) -> dict[str, object]:
        """Return human-readable values aligned to transformed feature names.

        The model uses scaled/encoded values, but the evidence packet should
        show original patient values whenever the transformed feature still maps
        directly to a raw column.
        """
        self._check_is_fitted()

        if len(raw_df) != 1:
            raise ValueError("raw_df must contain exactly one row.")

        raw_X = self._drop_initial_columns(raw_df)
        transformed_row = self.transform(raw_df).iloc[0]
        raw_row = raw_X.iloc[0]
        display_values: dict[str, object] = {}

        for feature in self.feature_names_:
            if feature in raw_X.columns:
                display_values[feature] = raw_row[feature]
                continue

            if feature.endswith("_missing"):
                raw_feature = feature.removesuffix("_missing")
                if raw_feature in raw_X.columns:
                    display_values[feature] = int(pd.isna(raw_row[raw_feature]))
                    continue

            display_values[feature] = transformed_row[feature]

        return display_values

    def fit_transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on raw training rows and return the processed training matrix."""
        return self.fit(raw_df).transform(raw_df)

    def _drop_initial_columns(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = self.id_cols + self.leakage_cols + [self.target_col]
        return raw_df.drop(columns=drop_cols, errors="ignore").copy()

    def _learn_column_groups(self, X: pd.DataFrame) -> None:
        self.numeric_cols_ = []
        self.binary_cols_ = []
        self.categorical_cols_ = []
        self.ordinal_cols_ = []
        self.high_missing_cols_ = []

        for column in X.columns:
            n_unique = X[column].nunique(dropna=True)

            if n_unique == 2:
                self.binary_cols_.append(column)
            elif np.issubdtype(X[column].dtype, np.number):
                self.numeric_cols_.append(column)
            elif X[column].dtype == "O":
                self.categorical_cols_.append(column)
            else:
                self.categorical_cols_.append(column)

        self.missing_indicator_cols_ = [
            column for column in self.numeric_cols_ if X[column].isna().any()
        ]

    def _clean_feature_names(self, raw_feature_names: Iterable[str]) -> list[str]:
        cleaned = []

        for name in raw_feature_names:
            feature = str(name)
            if "__" in feature:
                transformer_name, feature = feature.split("__", 1)
            else:
                transformer_name = ""

            if feature.startswith("missingindicator_"):
                feature = feature.replace("missingindicator_", "", 1) + "_missing"

            if transformer_name == "ohe":
                feature = self._clean_ohe_feature_name(feature)

            cleaned.append(feature)

        return cleaned

    def _clean_ohe_feature_name(self, feature: str) -> str:
        for column in sorted(self.categorical_cols_, key=len, reverse=True):
            prefix = f"{column}_"
            if feature.startswith(prefix):
                category = feature[len(prefix):]
                return f"{column}_{category}"

        return feature

    def _check_is_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("ICUPreprocessor is not fitted. Call fit() first.")


def load_feature_names(path: str) -> list[str]:
    """Load one-column feature name files saved without a header."""
    return pd.read_csv(path, header=None)[0].tolist()


def align_to_features(X: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    """Align a processed frame to an expected feature list."""
    return X.reindex(columns=list(feature_names), fill_value=0)
