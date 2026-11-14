"""Line up verified gallery models so their methods can be compared side by side.

Splits are seeded, so two runs on the same dataset with the same holdout portion test on the same
holdout samples, and with the same fold count also share their CV folds. Only then are their
scores directly comparable; `comparability_issues` names what differs otherwise.
"""

import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from gtex_explorer.display import METRICS, count
from gtex_explorer.training.data import samples_fingerprint, stable_hash
from gtex_explorer.training.gallery import list_gallery
from gtex_explorer.training.panels import load_run_panel
from gtex_explorer.training.run_labels import (
    L1_OPTION_LABELS,
    NOTEBOOK_DATA,
    XGBOOST_LABELS,
    data_settings,
    method_description,
    model_settings,
    parameter_settings,
    run_created,
    run_name,
)

MAX_MODELS = 4
LETTERS = "ABCD"
BASES = {"holdout": "Evaluasi holdout", "cv": "Evaluasi CV"}
# Settings rows in the order of the Latih Model form. CPU threads do not change what a model learns.
SETTING_ORDER = [
    "Dataset",
    "Sampel",
    "Donor",
    "Jaringan",
    "Data development",
    "Data holdout",
    "Pembagian",
    "Nilai kosong",
    "Filter teknis",
    "Gen kandidat",
    "Pra-pemrosesan",
    "Metode seleksi fitur",
    "Top-k gen",
    "C (regularisasi)",
    "n_neighbors",
    "Evaluasi tambahan",
    *(label for name, label in XGBOOST_LABELS.items() if name != "n_jobs"),
    *L1_OPTION_LABELS.values(),
]


# --- Models ---


def _samples_identity(saved):
    """What the run's samples were, apart from its Gen kandidat count.

    Runs started since this check existed save it. For older runs it is read from the saved dataset when
    that is still the one the run used; otherwise the full dataset fingerprint is used, which is stricter.
    """
    if saved.get("samples_fingerprint"):
        return saved["samples_fingerprint"]
    try:
        manifest = json.loads((Path(saved["data_root"]) / "manifest.json").read_text(encoding="utf-8"))
        if stable_hash(manifest) == saved.get("data_fingerprint"):
            return samples_fingerprint(manifest)
    except (KeyError, OSError, TypeError, ValueError):
        pass
    return saved.get("data_fingerprint")


def gallery_models(state_root, *, issues=None):
    """Verified gallery models with the settings and scores needed to compare them, newest first."""
    models = []
    for entry in list_gallery(state_root, issues=issues):
        root = Path(state_root) / "runs" / entry["run_id"]
        saved = json.loads((root / "config.json").read_text(encoding="utf-8"))
        dataset = saved.get("dataset") or {}
        models.append(
            {
                "run_id": entry["run_id"],
                "name": run_name(root),
                "config": saved["config"],
                "dataset": dataset,
                # Same samples, labels and values; the Gen kandidat count may differ (see candidate_note).
                "fingerprint": _samples_identity(saved),
                "filename": dataset.get("source", {}).get("expression_file", NOTEBOOK_DATA),
                "method": method_description(saved["config"]),
                "created": run_created(root),
                # Receipts written before holdout support have no holdout key.
                "cv": entry.get("cv_summary"),
                "holdout": entry.get("holdout_summary"),
            }
        )
    return models


def _comparable(first, second):
    """Whether two models have a score measured on the same samples."""
    if first["fingerprint"] is None or first["fingerprint"] != second["fingerprint"]:
        return False
    if (first["config"].get("holdout_percent") or 0) != (second["config"].get("holdout_percent") or 0):
        return False
    if first["holdout"] and second["holdout"]:
        return True
    return bool(first["cv"] and second["cv"]) and first["config"]["cv_folds"] == second["config"]["cv_folds"]


def default_selection(models):
    """Up to four of the newest models whose scores are comparable with each other, else the newest two."""
    for anchor in models:
        matching = [model["run_id"] for model in models if model is anchor or _comparable(anchor, model)]
        if len(matching) >= 2:
            return matching[:MAX_MODELS]
    return [model["run_id"] for model in models[:2]]


def available_bases(models):
    """Evaluations that at least two of the models have scores for."""
    return [basis for basis in BASES if sum(bool(model[basis]) for model in models) >= 2]


def candidate_note(models, basis=None, letters=LETTERS):
    """A note when the scored models ranked different numbers of candidate genes, or None.

    This does not stop a comparison: the samples and split are the same. It says what else the scores reflect,
    so with a basis only the models scored on it count.
    """
    scored = [(letter, model) for letter, model in zip(letters, models) if basis is None or model[basis]]
    counts = [model["dataset"].get("candidate_count") for _, model in scored]
    if None in counts or len(set(counts)) < 2:
        return None
    _, described = _grouped(scored, lambda model: f"{count(model['dataset']['candidate_count'])} gen", len(models))
    # The shared samples and split are named by shared_data_note, just above this note.
    return (
        f"Gen kandidat berbeda: {described}, jadi selisih skor juga mencerminkan jumlah gen yang diperingkat "
        "sebelum seleksi fitur, bukan hanya metodenya."
    )


# Every model of a comparison, by how many there are.
ALL_MODELS = {2: "Kedua model", 3: "Ketiga model", 4: "Keempat model"}


def _models_named(letters, total):
    """Name a group of models: all of them, or by letter as an Indonesian list (A, B, dan C)."""
    if len(letters) == total and total in ALL_MODELS:
        return ALL_MODELS[total]
    if len(letters) < 3:
        return "Model " + " dan ".join(letters)
    return "Model " + ", ".join(letters[:-1]) + ", dan " + letters[-1]


def source_name(filename):
    """The dataset as a note names it."""
    return "data eksperimen notebook" if filename == NOTEBOOK_DATA else filename


def shared_data_note(models, basis, letters=LETTERS):
    """The dataset and split that the scored models share, for when comparability_issues finds none."""
    scored = [(letter, model) for letter, model in zip(letters, models) if model[basis]]
    first = scored[0][1]
    dataset, config = first["dataset"], first["config"]
    source = source_name(first["filename"])
    source = source if first["filename"] == NOTEBOOK_DATA else f"dataset {source}"
    sizes = [
        f"{count(dataset[key])} {unit}"
        for key, unit in [("sample_count", "sampel"), ("donor_count", "donor"), ("class_count", "jaringan")]
        if dataset.get(key) is not None
    ]
    if sizes:
        source += f" ({', '.join(sizes)})"
    holdout = config.get("holdout_percent") or 0
    if basis == "holdout":
        summary = first["holdout"]
        held = [f"{count(summary[key])} {unit}" for key, unit in
                [("validation_samples", "sampel"), ("validation_donors", "donor")] if summary.get(key)]
        split = f"data development {100 - holdout}% dan data holdout {holdout}%" + (
            f" ({' dari '.join(held)})" if held else ""
        )
        measured = "Skor diukur pada sampel holdout yang sama"
    else:
        folds = config["cv_folds"]
        split = (
            f"data development {100 - holdout}% dibagi menjadi {folds} fold CV yang sama"
            if holdout
            else f"seluruh sampel dibagi menjadi {folds} fold CV yang sama"
        )
        measured = "Skor diukur pada data validation fold yang sama"
    named = _models_named([letter for letter, _ in scored], len(models))
    return (
        f"{named} dilatih pada {source} dengan pembagian per donor yang sama: {split}. "
        f"{measured}, jadi dapat dibandingkan langsung."
    )


def unscored_note(models, basis, letters=LETTERS):
    """Which models have no score on this basis and why, or None."""
    reasons = {}
    for letter, model in zip(letters, models):
        if model[basis]:
            continue
        config = model["config"]
        if basis == "cv" and not config.get("evaluate_cv"):
            reason = "dilatih tanpa CV"
        elif basis == "holdout" and not config.get("holdout_percent"):
            reason = "dilatih tanpa data holdout"
        else:
            reason = f"tidak menyimpan skor {'CV' if basis == 'cv' else 'holdout'}"
        reasons.setdefault(reason, []).append(letter)
    if not reasons:
        return None
    parts = [f"{_models_named(group, len(models))} {reason}" for reason, group in reasons.items()]
    # At least two models have scores, so at most two of the four have none.
    subject = "jadi keduanya" if len(parts) > 1 else "jadi"
    return (
        f"{' dan '.join(parts)}, {subject} tidak tampil pada skor {BASES[basis]}. "
        "Panel gen dan pengaturannya tetap dibandingkan di bawah."
    )


def _grouped(scored, value, total):
    """The scored models grouped by a value, as "Model A dan C <value>; Model B <value>".

    Semicolons keep the groups apart from the commas inside them (Model A, B, dan C).
    """
    groups = {}
    for letter, model in scored:
        groups.setdefault(value(model), []).append(letter)
    return groups, "; ".join(f"{_models_named(group, total)} {shown}" for shown, group in groups.items())


def comparability_issues(models, basis, letters=LETTERS):
    """What differs between the scored models, naming which model has what; empty when nothing does."""
    scored = [(letter, model) for letter, model in zip(letters, models) if model[basis]]
    total = len(models)
    issues = []
    fingerprints = {}
    for letter, model in scored:
        fingerprints.setdefault(model["fingerprint"], []).append(letter)
    if None in fingerprints or len(fingerprints) > 1:
        # Models from an older run without a fingerprint are each their own dataset.
        groups = [[letter] for letter in fingerprints.pop(None, [])] + list(fingerprints.values())
        names = {letter: source_name(model["filename"]) for letter, model in scored}
        described = [f"{_models_named(group, total)} memakai {names[group[0]]}" for group in groups]
        issue = f"Dataset berbeda: {'; '.join(described)}."
        if len({names[group[0]] for group in groups}) < len(groups):
            issue += " Nama berkasnya sama, tetapi sampel atau nilainya berbeda."
        issues.append(issue)
    holdouts, described = _grouped(
        scored, lambda model: f"{model['config']['holdout_percent']}%" if model["config"].get("holdout_percent")
        else "tanpa holdout", total
    )
    if len(holdouts) > 1:
        issues.append(f"Porsi holdout berbeda: {described}, sehingga data development dan holdout tidak sama.")
    if basis == "cv":
        folds, described = _grouped(scored, lambda model: f"{model['config']['cv_folds']} fold", total)
        if len(folds) > 1:
            issues.append(f"Jumlah fold CV berbeda: {described}.")
    return issues


# --- Scores ---


def metric_frame(models, basis):
    """One row per model and metric: the score and, for CV, its SD across folds. Missing scores are NaN."""
    rows = []
    for letter, model in zip(LETTERS, models):
        summary = model[basis]
        for key, label in METRICS.items():
            if not summary:
                score = sd = None
            elif basis == "cv":
                score, sd = summary["mean_" + key], summary["std_" + key]
            else:
                score, sd = summary[key], None
            rows.append({"model": letter, "metric": label, "score": score, "sd": sd})
    return pd.DataFrame(rows, columns=["model", "metric", "score", "sd"]).astype({"score": float, "sd": float})


def tissue_frame(models, basis):
    """One row per model and tissue with its F1-score, CV SD, and holdout sample count."""
    rows = []
    for letter, model in zip(LETTERS, models):
        summary = model[basis]
        if not summary:
            continue
        if basis == "cv":
            # CV summaries saved before per-tissue scores existed have none.
            for item in summary.get("tissue_f1", []):
                rows.append(
                    {"model": letter, "tissue": item["tissue"], "f1": item["mean_f1"], "sd": item["std_f1"]}
                )
        else:
            for tissue, score in summary["tissue_f1"].items():
                rows.append(
                    {
                        "model": letter,
                        "tissue": tissue,
                        "f1": score,
                        "samples": summary["tissue_support"].get(tissue),
                    }
                )
    return pd.DataFrame(rows, columns=["model", "tissue", "f1", "sd", "samples"]).astype(
        {"f1": float, "sd": float, "samples": float}
    )


# --- Gene panels ---


def final_panels(state_root, models):
    """The final gene panel of each model, in selection order."""
    return {
        letter: load_run_panel(state_root, model["run_id"], "final")[0]["gene"].tolist()
        for letter, model in zip(LETTERS, models)
    }


def panel_overlap(panels):
    """Shared genes and Jaccard index for every pair of panels."""
    rows = []
    for first, second in combinations(panels, 2):
        a, b = set(panels[first]), set(panels[second])
        rows.append(
            {
                "first": first,
                "second": second,
                "shared": len(a & b),
                "jaccard": len(a & b) / len(a | b) if a | b else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=["first", "second", "shared", "jaccard"])


def panel_membership(panels):
    """Every gene picked by any model, with its selection rank in each panel (empty when not picked)."""
    genes = list(dict.fromkeys(gene for panel in panels.values() for gene in panel))
    frame = pd.DataFrame({"gene": genes})
    for letter, panel in panels.items():
        ranks = {gene: rank for rank, gene in enumerate(panel, 1)}
        frame[f"rank_{letter}"] = frame["gene"].map(ranks).astype("Int64")
    frame["models"] = frame.filter(like="rank_").notna().sum(axis=1)
    return frame.sort_values(["models", "gene"], ascending=[False, True], kind="stable").reset_index(drop=True)


# --- Settings ---


def settings_frame(models):
    """One row per setting and one column per model letter, plus whether the models differ.

    Rows follow the Latih Model form: the data, then Pra-pemrosesan, Seleksi fitur, Evaluasi tambahan,
    and the XGBoost parameters, with the same labels. A setting a model does not have reads "-".
    """
    per_model = []
    for m in models:
        settings = data_settings(m["config"], m["dataset"].get("candidate_count"))
        settings += model_settings(m["config"]) + parameter_settings(m["config"])
        per_model.append(
            {
                "Dataset": m["filename"],
                "Sampel": count(m["dataset"].get("sample_count")),
                "Donor": count(m["dataset"].get("donor_count")),
                "Jaringan": count(m["dataset"].get("class_count")),
                **dict(settings),
            }
        )
    # A setting no chosen model has, such as C when none uses L1, is left out.
    rows = {
        label: [values.get(label, "-") for values in per_model]
        for label in SETTING_ORDER
        if any(label in values for values in per_model)
    }
    frame = pd.DataFrame(rows, index=list(LETTERS[: len(models)])).T
    frame.index.name = "Pengaturan"
    frame["differs"] = frame.nunique(axis=1) > 1
    return frame.reset_index()
