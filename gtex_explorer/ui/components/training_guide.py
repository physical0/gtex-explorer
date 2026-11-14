"""Cara kerja pelatihan: a user guide to the Latih Model page and what a training run does behind it.

Each fact has one place, so nothing is explained twice:
- Alur proses pelatihan tells the run, including what happens with and without cross-validation.
- Pra-pemrosesan explains its five steps, together with the three settings that belong to them.
- Pengaturan dan pengaruhnya covers the remaining settings and points to the stage they change.
Every control on the form is explained under its own label. Only what the training code does is stated.
"""

from __future__ import annotations

import html

import streamlit as st

from gtex_explorer.display import METHODS
from gtex_explorer.training.data import TECHNICAL_FILTER_RULE
from gtex_explorer.training.run_labels import L1_OPTION_LABELS, XGBOOST_LABELS

# --- Content ---

# How the user works through the page, one card per tab.
USAGE = [
    (
        "Data training",
        "Unggah matriks TPM dan metadata, tetapkan kolom sampel, donor, dan jaringan, lalu periksa data; "
        "atau gunakan data eksperimen notebook.",
    ),
    ("Konfigurasi", "Atur pembagian data, pra-pemrosesan, seleksi fitur, evaluasi, dan hiperparameter XGBoost."),
    (
        "Tinjau dan latih",
        "Periksa ringkasan, beri <em>Nama pelatihan</em> bila perlu, lalu pilih <em>Latih dan simpan model</em>.",
    ),
]

# The stages of a run, as (title, optional, text). The only place that tells the run.
FLOW = [
    (
        "Inisialisasi pelatihan",
        False,
        "Data dan konfigurasi disimpan, lalu pelatihan dijalankan oleh worker di latar belakang, satu pelatihan "
        "pada satu waktu. Kemajuannya tampil di Hasil Pelatihan, tab Status proses.",
    ),
    (
        "Pembagian data utama",
        False,
        "Sampel dipartisi per donor menjadi data development dan data holdout secara deterministik, sehingga satu "
        "donor tidak muncul di kedua partisi.",
    ),
    (
        "Cross-validation",
        True,
        "Bila <em>Evaluasi dengan cross-validation</em> aktif, data development dibagi menjadi <em>k</em> fold per "
        "donor. Pada setiap fold, pra-pemrosesan, seleksi fitur, dan XGBoost dilatih pada data training fold, lalu "
        "dievaluasi pada data validation fold, sehingga daftar gen kandidat dapat sedikit berbeda antar-fold. Hasil "
        "setiap fold disimpan sebagai checkpoint.",
    ),
    (
        "Pelatihan model final",
        False,
        "Dengan atau tanpa cross-validation, proses pra-pemrosesan dilakukan satu kali pada data development "
        "(seluruh sampel bila porsi holdout 0%), lalu XGBoost menggunakan seleksi fitur (jika digunakan) dilatih pada hasilnya. Nilai dari "
        "fold tidak digunakan.",
    ),
    (
        "Evaluasi holdout",
        True,
        "Bila porsi holdout lebih dari 0%, data holdout diproses dengan nilai pra-pemrosesan model final, lalu model "
        "dievaluasi satu kali tanpa diubah.",
    ),
    (
        "Penyimpanan dan verifikasi",
        False,
        "Model, panel gen, serta median, rerata, dan simpangan baku gen terpilih disimpan dan diverifikasi. Model "
        "kemudian tersedia di Gallery model, Panel Gen, dan Prediksi Jaringan; Prediksi Jaringan menerapkan nilai "
        "tersebut pada data input.",
    ),
]

# Pra-pemrosesan: its steps with the settings that belong to them, as (title, text, only when its setting is on).
PREPROCESSING_LEAD = (
    "Dijalankan pada tahap 3 (Cross-validation) jika digunakan dan tahap 4 (Pelatihan model final). Nilai setiap langkah dihitung "
    "dari data yang dilatih pada tahap itu (data training fold atau data development), lalu diterapkan tanpa diubah "
    "pada data evaluasi untuk mencegah kebocoran data."
)
# As (title, text, tag): the tag marks a step the form lets the user choose or switch off.
PREPROCESSING_STEPS = [
    (
        "Nilai kosong",
        "Nilai kosong pada data training diisi median gen tersebut, atau gen yang memiliki nilai kosong dibuang. "
        "Nilai kosong pada data evaluasi tetap diisi median data training.",
        "Pilihan",
    ),
    (
        "Filter teknis",
        "Gen yang ekspresinya terlalu rendah untuk diukur atau tidak bervariasi dipotong dan tidak digunakan. Gen "
        f"dipertahankan hanya jika memenuhi {TECHNICAL_FILTER_RULE}. Bila nonaktif, hanya gen yang tidak bervariasi "
        "yang dibuang.",
        "Bila aktif",
    ),
    (
        "Gen kandidat",
        "Gen diperingkat menurut varians nilai log2(TPM+1), baik transformasi log2(TPM+1) aktif maupun tidak; sejumlah "
        "<em>Gen kandidat</em> teratas dipertahankan sebagai masukan seleksi fitur.",
        None,
    ),
    (
        "Transformasi log2(TPM+1)",
        "Nilai gen kandidat diteruskan sebagai log2(TPM+1) untuk mengompresi distribusi ekspresi yang condong; bila "
        "nonaktif, diteruskan sebagai TPM asli.",
        "Bila aktif",
    ),
    (
        "Standardisasi (z-score)",
        "Menskalakan gen menjadi rerata 0 dan simpangan baku 1; berpengaruh besar pada L1, kecil pada ANOVA, Mutual "
        "Information, dan XGBoost.",
        "Bila aktif",
    ),
]

_xgb = XGBOOST_LABELS
_l1 = L1_OPTION_LABELS

# The remaining controls of the Konfigurasi step, grouped as on the form, as (section, [(labels, text)]).
# Pra-pemrosesan is not here: its settings are explained with its steps.
SETTINGS = [
    (
        "Pembagian data utama",
        [
            (
                ["Porsi data holdout"],
                "Ukuran data holdout pada tahap 2 (Pembagian data utama); 0% meniadakan tahap 5 (Evaluasi holdout).",
            )
        ],
    ),
    (
        "Seleksi fitur",
        [
            (
                ["Metode seleksi fitur", METHODS["l1"], "C (regularisasi)"],
                "Metode embedded berbasis koefisien; C lebih kecil memperkuat regularisasi sehingga gen terpilih "
                "lebih sedikit.",
            ),
            (
                ["Opsi selector", _l1["l1_max_iter"], _l1["l1_tol"], _l1["l1_class_weight"]],
                "Konvergensi solver dan penyeimbangan kelas; pelatihan berhenti bila solver tidak konvergen atau gen "
                "nonzero kurang dari Top-k.",
            ),
            ([METHODS["anova"]], "Metode filter berbasis statistik F antarjaringan."),
            (
                [METHODS["mutual_information"], "n_neighbors"],
                "Metode filter berbasis informasi bersama; n_neighbors menentukan tetangga estimator k-NN.",
            ),
            ([METHODS["baseline"]], "Seluruh gen kandidat digunakan; Top-k tidak berlaku."),
            (["Top-k gen"], "Jumlah gen peringkat teratas yang menjadi masukan XGBoost."),
        ],
    ),
    (
        "Evaluasi tambahan",
        [
            (
                ["Evaluasi dengan cross-validation", "Jumlah fold"],
                "Menjalankan tahap 3 (Cross-validation) dengan <em>k</em> fold; waktu pelatihan bertambah sesuai "
                "jumlah fold.",
            ),
        ],
    ),
    (
        "Parameter XGBoost",
        [
            (
                [_xgb[name] for name in ["n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree"]],
                "Hiperparameter yang sama untuk setiap fold dan model final.",
            ),
            (
                ["Parameter lanjutan", _xgb["reg_lambda"], _xgb["reg_alpha"], _xgb["gamma"], _xgb["n_jobs"]],
                "Regularisasi bobot daun dan ambang pemisahan simpul; Thread CPU hanya memengaruhi kecepatan.",
            ),
        ],
    ),
]


# --- Drawing ---


def _usage_html() -> str:
    """The three tabs of the page as numbered cards."""
    return "".join(
        f'<li class="guide-usage-step"><span class="guide-number" aria-hidden="true">{number}</span>'
        f'<p class="guide-title">{html.escape(title)}</p><p class="guide-text">{text}</p></li>'
        for number, (title, text) in enumerate(USAGE, 1)
    )


def _flow_html() -> str:
    """The stages of a run, the optional ones tagged."""
    return "".join(
        f'<li class="guide-step"><span class="guide-number" aria-hidden="true">{number}</span><div>'
        f'<p class="guide-title">{html.escape(title)}'
        + ('<span class="guide-tag">Opsional</span>' if optional else "")
        + f'</p><p class="guide-text">{text}</p></div></li>'
        for number, (title, optional, text) in enumerate(FLOW, 1)
    )


def _settings_html() -> str:
    """The remaining controls of the Konfigurasi step, by section, with what they change."""
    sections = []
    for section, items in SETTINGS:
        rows = "".join(
            '<div class="guide-setting"><dt>'
            + ", ".join(f"<em>{html.escape(label)}</em>" for label in labels)
            + f"</dt><dd>{text}</dd></div>"
            for labels, text in items
        )
        sections.append(f'<section class="guide-settings-group"><p class="guide-group">{section}</p><dl>{rows}</dl></section>')
    return "".join(sections)


def _preprocessing_html() -> str:
    """The pra-pemrosesan steps in order, each with its own setting where it has one."""
    steps = "".join(
        f'<li class="guide-prep-step"><p class="guide-prep-title"><span class="guide-prep-number">{number}</span>'
        f"{html.escape(title)}"
        + (f'<span class="guide-tag">{tag}</span>' if tag else "")
        + f'</p><p class="guide-text">{text}</p></li>'
        for number, (title, text, tag) in enumerate(PREPROCESSING_STEPS, 1)
    )
    return (
        '<section class="guide-prep" aria-label="Pra-pemrosesan">'
        '<p class="guide-heading">Pra-pemrosesan</p>'
        f'<p class="guide-text guide-prep-lead">{PREPROCESSING_LEAD}</p>'
        f'<ol class="guide-prep-steps">{steps}</ol></section>'
    )


def render_training_guide() -> None:
    """Show the closed Cara kerja pelatihan section: how to use the page and what a run does."""
    with st.expander("Cara kerja pelatihan", icon=":material/account_tree:"):
        st.markdown(
            '<section class="training-guide" aria-label="Cara kerja pelatihan">'
            '<p class="guide-heading">Langkah penggunaan</p>'
            f'<ol class="guide-usage">{_usage_html()}</ol>'
            '<div class="guide-grid">'
            '<div><p class="guide-heading">Alur proses pelatihan</p>'
            f'<ol class="guide-steps">{_flow_html()}</ol></div>'
            '<aside class="guide-recipe" aria-label="Pengaturan dan pengaruhnya">'
            '<p class="guide-heading">Pengaturan dan pengaruhnya</p>'
            f"{_settings_html()}</aside></div>"
            f"{_preprocessing_html()}</section>",
            unsafe_allow_html=True,
        )
