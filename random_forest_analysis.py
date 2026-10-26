from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET_COLUMN = "target"


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


def evaluate(model, x_test: pd.DataFrame, y_test: pd.Series) -> Metrics:
    pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
    else:
        auc = float("nan")

    return Metrics(
        accuracy=accuracy_score(y_test, pred),
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
        roc_auc=auc,
    )


def save_importance_plot(importances: pd.DataFrame, output_path: Path) -> None:
    ordered = importances.sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(ordered["feature"], ordered["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Random Forest feature importance")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_importance_svg(importances: pd.DataFrame, output_path: Path) -> None:
    """Save a compact, dependency-free SVG copy of the importance chart."""
    ordered = importances.sort_values("importance", ascending=False).reset_index(drop=True)
    width, height = 820, 420
    left, top, bar_height, gap = 180, 55, 38, 18
    max_value = float(ordered["importance"].max()) or 1.0
    chart_width = 540
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="410" y="30" text-anchor="middle" font-family="Arial" font-size="22">Random Forest feature importance</text>',
    ]
    for index, row in ordered.iterrows():
        y = top + index * (bar_height + gap)
        value = float(row["importance"])
        bar_width = chart_width * value / max_value
        feature = str(row["feature"])
        parts.append(
            f'<text x="{left - 12}" y="{y + 25}" text-anchor="end" font-family="Arial" font-size="16">{feature}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" fill="#4C78A8"/>'
        )
        parts.append(
            f'<text x="{left + bar_width + 8:.1f}" y="{y + 25}" font-family="Arial" font-size="14">{value:.4f}</text>'
        )
    parts.append('</svg>')
    output_path.write_text('\n'.join(parts), encoding='utf-8')


def run_feature_removal_experiment(
    tuned_model: RandomForestClassifier,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    importance_df: pd.DataFrame,
    cv: StratifiedKFold,
) -> tuple[pd.DataFrame, list[str]]:
    least_to_most = importance_df.sort_values("importance")["feature"].tolist()
    rows: list[dict[str, object]] = []

    # Baseline CV on all features. We use CV on train to decide which features may be removed,
    # then look at test only for final reporting. This avoids selecting features on the test set.
    full_cv = cross_val_score(
        clone(tuned_model), x_train, y_train, scoring="f1", cv=cv, n_jobs=1
    ).mean()

    for remove_count in range(0, len(least_to_most)):
        removed = least_to_most[:remove_count]
        selected = [column for column in x_train.columns if column not in removed]
        model = clone(tuned_model)
        cv_f1 = cross_val_score(
            model,
            x_train[selected],
            y_train,
            scoring="f1",
            cv=cv,
            n_jobs=1,
        ).mean()
        model.fit(x_train[selected], y_train)
        test_metrics = evaluate(model, x_test[selected], y_test)

        rows.append(
            {
                "features_left": len(selected),
                "removed_features": ", ".join(removed) if removed else "-",
                "cv_f1": cv_f1,
                "cv_delta_vs_full": cv_f1 - full_cv,
                "test_accuracy": test_metrics.accuracy,
                "test_f1": test_metrics.f1,
                "test_roc_auc": test_metrics.roc_auc,
            }
        )

    experiment = pd.DataFrame(rows)

    # "Без существенной потери" = падение среднего CV F1 не больше 0.01.
    acceptable = experiment[experiment["cv_f1"] >= full_cv - 0.01]
    best_row = acceptable.sort_values(["features_left", "cv_f1"], ascending=[True, False]).iloc[0]
    removed_features = [] if best_row["removed_features"] == "-" else str(best_row["removed_features"]).split(", ")

    return experiment, removed_features


def main() -> None:
    parser = argparse.ArgumentParser(description="ML-P-7: Random Forest, variant B")
    parser.add_argument("--data", default="data.csv", help="CSV dataset path")
    parser.add_argument("--output-dir", default="results", help="Directory for output files")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Column {TARGET_COLUMN!r} not found in {data_path}")

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 1) Single decision tree on exactly the same split.
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    tree.fit(x_train, y_train)

    # 2) Default Random Forest. In current sklearn the default n_estimators is 100.
    default_forest = RandomForestClassifier(random_state=RANDOM_STATE)
    default_forest.fit(x_train, y_train)

    # 3) Hyperparameter tuning required by variant B.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 4, 8],
        "min_samples_leaf": [1, 2, 4],
    }
    search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        return_train_score=False,
    )
    search.fit(x_train, y_train)
    tuned_forest: RandomForestClassifier = search.best_estimator_

    # Direct metrics comparison on the same untouched test split.
    models = {
        "Decision Tree": tree,
        "Random Forest (default)": default_forest,
        "Random Forest (tuned)": tuned_forest,
    }
    rows = []
    for name, model in models.items():
        metrics = evaluate(model, x_test, y_test)
        rows.append({"model": name, **metrics.as_dict()})

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    cv_results = pd.DataFrame(search.cv_results_)
    columns = [
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_leaf",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
    cv_results[columns].sort_values("rank_test_score").to_csv(
        output_dir / "grid_search_results.csv", index=False
    )

    # 4) Feature importance and chart.
    importance_df = pd.DataFrame(
        {
            "feature": x.columns,
            "importance": tuned_forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    save_importance_plot(importance_df, output_dir / "feature_importance.png")
    save_importance_svg(importance_df, output_dir / "feature_importance.svg")

    # 5) Remove the least important features, deciding from CV on train only.
    removal_df, removable_features = run_feature_removal_experiment(
        tuned_model=tuned_forest,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        importance_df=importance_df,
        cv=cv,
    )
    removal_df.to_csv(output_dir / "feature_removal_experiment.csv", index=False)

    selected_features = [feature for feature in x.columns if feature not in removable_features]
    reduced_model = clone(tuned_forest)
    reduced_model.fit(x_train[selected_features], y_train)
    reduced_metrics = evaluate(reduced_model, x_test[selected_features], y_test)

    summary_lines = [
        f"Rows: {len(df)}",
        f"Train rows: {len(x_train)}",
        f"Test rows: {len(x_test)}",
        f"Features: {', '.join(x.columns)}",
        "",
        "Best hyperparameters:",
        f"  n_estimators={search.best_params_['n_estimators']}",
        f"  max_depth={search.best_params_['max_depth']}",
        f"  min_samples_leaf={search.best_params_['min_samples_leaf']}",
        f"Best CV F1: {search.best_score_:.4f}",
        "",
        "Model comparison:",
        comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "Feature importance:",
        importance_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "Feature removal experiment:",
        removal_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        f"Features removable with CV F1 drop <= 0.01: {', '.join(removable_features) if removable_features else 'none'}",
        f"Reduced model features: {', '.join(selected_features)}",
        f"Reduced model test F1: {reduced_metrics.f1:.4f}",
        f"Reduced model test accuracy: {reduced_metrics.accuracy:.4f}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
