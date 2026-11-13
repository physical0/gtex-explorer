"""Final and fold-specific panels from recorded user training runs."""

import json

import pandas as pd
import streamlit as st

from gtex_explorer.display import SCORE_LABELS, count, percent, plain
from gtex_explorer.platform.cloud.media import download_button
from gtex_explorer.training.panels import load_run_panel
from gtex_explorer.training.run_labels import NOTEBOOK_DATA, method_description, run_name
from gtex_explorer.ui.components.training_runs import choose_run, show_diagnostics


def render_training_gene_page(state_root):
    """Show the gene panels of user-trained models."""
    st.title("Panel gen pelatihan")
    st.caption("Gen dari model final dan setiap fold pada pelatihan pengguna.")
    run_id = choose_run(state_root, "training_panel_run", gallery_only=True)
    if run_id is None:
        return
    try:
        # Read the saved run.
        root = state_root / "runs" / run_id
        saved = json.loads((root / "config.json").read_text(encoding="utf-8"))
        config = saved["config"]
        heading, back = st.columns([3, 1], vertical_alignment="center")
        # A named run leads with its name, its method below it.
        name = run_name(root)
        heading.subheader(plain(name) if name else method_description(config))
        if name:
            heading.caption(method_description(config))
        if back.button("Kembali ke hasil", key="training_panel_back", icon=":material/arrow_back:", width="stretch"):
            st.session_state.training_run = run_id
            st.session_state.page = "Hasil Pelatihan"
            st.rerun()

        # Choose the final model or one fold. The final model learns from the development data only.
        development = "data development" if config.get("holdout_percent") else "seluruh data development"
        options = []
        if (state_root / "models" / run_id / "bundle_manifest.json").is_file():
            options.append("final")
        options += [i for i in range(1, config["cv_folds"] + 1) if (root / f"fold_{i}.json").is_file()]
        if not options:
            st.info("Belum ada panel yang selesai disimpan untuk pelatihan ini.")
            return
        scope = st.selectbox(
            "Sumber panel",
            options,
            format_func=lambda x: (
                f"Model final - {development}" if x == "final" else f"Fold {x} - data training"
            ),
            key=f"training_panel_scope_{run_id}",
        )
        genes, info = load_run_panel(state_root, run_id, scope)

        # Sample counts for this panel.
        dataset = saved.get("dataset", {})
        if dataset:
            st.caption(
                dataset.get("source", {}).get("expression_file", NOTEBOOK_DATA)
                + f" - dataset lengkap: {dataset['sample_count']} sampel"
            )
        training_count = info.get("training_samples", dataset.get("sample_count") if scope == "final" else None)
        validation_count = info.get("validation_samples") if scope != "final" else None
        counts = st.columns(4)
        for column, value, label in zip(
            counts,
            [training_count, validation_count, len(genes), info.get("candidate_count", dataset.get("candidate_count"))],
            ["Sampel training", "Sampel validation", "Gen terpilih", "Gen kandidat"],
        ):
            column.markdown(f"**{count(value)}**  \n{label}")
        if scope == "final":
            st.caption(
                f"Panel final dipilih dari {development}. "
                + (
                    "Data holdout hanya dipakai untuk menguji model."
                    if config.get("holdout_percent")
                    else "Tanpa data holdout, model tidak diuji pada data terpisah."
                )
                + (" Setiap fold CV memiliki panelnya sendiri." if config["evaluate_cv"] else "")
            )
        gene_tab, diagnostic_tab, stability_tab = st.tabs(["Daftar gen", "Diagnostik metode", "Stabilitas antar-fold"])
        with gene_tab:
            search, limit = st.columns([3, 1])
            query = search.text_input("Cari ID gen", key="training_panel_search")
            maximum = limit.number_input(
                "Baris ditampilkan", 1, max(1, len(genes)), min(100, len(genes)), key="training_panel_limit"
            )
            visible = genes[genes.gene.str.contains(query, case=False, regex=False)] if query else genes
            labels = {
                "gene": "ID gen",
                "selection_rank": "Peringkat",
                "selection_score": SCORE_LABELS.get(config["selector"], "Skor seleksi"),
                "p_value": "p-value",
                "nonzero_class_count": "Jumlah kelas dengan koefisien nonzero",
            }
            if config["selector"] == "baseline":
                visible = visible.drop(columns="selection_rank", errors="ignore")
                st.caption(
                    "Seluruh gen kandidat digunakan. Daftar ini tidak memiliki skor atau peringkat seleksi fitur."
                )
            st.caption(f"{count(len(visible))} dari {count(len(genes))} gen sesuai pencarian")
            # Format a display copy. The complete CSV keeps the original values and precision.
            display = visible.head(maximum).copy()
            ordered = [name for name in ["selection_rank", "gene", "selection_score", "p_value"] if name in display]
            display = display[ordered + [name for name in display if name not in ordered]]
            st.dataframe(
                display.rename(columns=labels),
                hide_index=True,
                width="stretch",
                height=320,
                column_config={
                    labels["selection_score"]: st.column_config.NumberColumn(format="%.4f"),
                    "p-value": st.column_config.NumberColumn(format="%.2e"),
                },
            )
            download_button(
                "Unduh panel gen lengkap",
                genes.to_csv(index=False),
                f"{run_id}_panel_{scope}.csv",
                key="training_download_genes",
                icon=":material/download:",
            )
            st.caption(f"CSV berisi seluruh {count(len(genes))} gen pada panel ini.")
        with diagnostic_tab:
            show_diagnostics(info)
            if config["selector"] == "anova":
                st.caption(
                    "Skor F dan p-value dihitung dari data training panel ini. p-value tidak mengoreksi ketergantungan donor."
                )
            elif config["selector"] == "mutual_information":
                st.caption("Skor MI memakai n_neighbors tersimpan pada konfigurasi pelatihan ini, dalam satuan nat.")
            elif config["selector"] == "l1":
                st.caption(
                    "Skor seleksi adalah koefisien absolut maksimum antarkelas. Gen nonzero dihitung pada seluruh gen kandidat."
                )
                if config.get("standardize") is False:
                    st.caption(
                        "Model ini dilatih tanpa standardisasi, jadi besar koefisien juga dipengaruhi skala ekspresi "
                        "setiap gen, bukan hanya kekuatan hubungannya dengan jaringan."
                    )
        with stability_tab:
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))
            summary = status.get("cv_summary")
            if summary and summary.get("mean_jaccard") is not None:
                st.metric("Rerata Jaccard antar-fold", percent(summary["mean_jaccard"]))
                pairs = pd.DataFrame(summary["jaccard_pairs"]).rename(
                    columns={"fold_a": "Fold A", "fold_b": "Fold B", "jaccard": "Jaccard"}
                )
                pairs["Jaccard"] = pairs["Jaccard"].map(percent)
                st.caption("Perbandingan gen terpilih antar-fold pada seluruh run ini.")
                st.dataframe(pairs, hide_index=True, width="stretch")
            elif config["selector"] == "baseline":
                st.info("Baseline menggunakan seluruh gen kandidat. Jaccard seleksi fitur tidak dihitung.")
            elif not config["evaluate_cv"]:
                st.info("Run ini dilatih tanpa CV. Stabilitas antar-fold tidak dihitung.")
            else:
                st.info(
                    "Jaccard tersedia setelah seleksi fitur pada seluruh fold selesai. Baseline tidak dihitung sebagai stabilitas seleksi."
                )
    except (ValueError, OSError, KeyError, TypeError) as error:
        st.error(f"Panel tidak dapat dibaca: {error}")
