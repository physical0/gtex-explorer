"""Offline biological dictionary with a searchable index and reading panel."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from gtex_explorer.platform.cloud.media import download_button


SOURCES = {
    "gene": ("NHGRI - Gene", "https://www.genome.gov/genetics-glossary/Gene"),
    "expression": ("NHGRI - Gene Expression", "https://www.genome.gov/genetics-glossary/Gene-Expression"),
    "dna": ("NHGRI - DNA", "https://www.genome.gov/genetics-glossary/Deoxyribonucleic-Acid-DNA"),
    "rna": ("NHGRI - RNA", "https://www.genome.gov/genetics-glossary/Ribonucleic-Acid-RNA"),
    "mrna": ("NHGRI - Messenger RNA", "https://www.genome.gov/genetics-glossary/Messenger-RNA-mRNA"),
    "protein": ("NHGRI - Protein", "https://www.genome.gov/genetics-glossary/Protein"),
    "transcription": ("NHGRI - Transcription", "https://www.genome.gov/genetics-glossary/Transcription"),
    "transcriptome": ("NHGRI - Transcriptome Fact Sheet", "https://www.genome.gov/about-genomics/fact-sheets/Transcriptome-Fact-Sheet"),
    "rnaseq": ("NHGRI - RNA-seq", "https://www.genome.gov/genetics-glossary/RNA-seq"),
    "tpm": ("RNA-SeQC - Gene expression in TPM", "https://github.com/getzlab/rnaseqc"),
    "gtex": ("GTEx - Adult GTEx project", "https://gtexportal.org/home/aboutAdultGtex"),
    "tissue": ("OpenStax - Types of Tissues", "https://openstax.org/books/anatomy-and-physiology-2e/pages/4-1-types-of-tissues"),
    "pituitary": ("OpenStax - Pituitary Gland", "https://openstax.org/books/anatomy-and-physiology-2e/pages/17-3-the-pituitary-gland-and-hypothalamus"),
    "thyroid": ("OpenStax - Thyroid Gland", "https://openstax.org/books/anatomy-and-physiology-2e/pages/17-4-the-thyroid-gland"),
    "adrenal": ("OpenStax - Adrenal Glands", "https://openstax.org/books/anatomy-and-physiology-2e/pages/17-6-the-adrenal-glands"),
    "heart": ("OpenStax - Heart Anatomy", "https://openstax.org/books/anatomy-and-physiology-2e/pages/19-1-heart-anatomy"),
    "arteries": ("OpenStax - Circulatory Pathways", "https://openstax.org/books/anatomy-and-physiology-2e/pages/20-5-circulatory-pathways"),
    "digestion": ("OpenStax - Liver and Pancreas", "https://openstax.org/books/anatomy-and-physiology-2e/pages/23-6-accessory-organs-in-digestion-the-liver-pancreas-and-gallbladder"),
    "adipose": ("OpenStax - Connective Tissue", "https://openstax.org/books/anatomy-and-physiology-2e/pages/4-3-connective-tissue-supports-and-protects"),
    "muscle": ("OpenStax - Skeletal Muscle", "https://openstax.org/books/anatomy-and-physiology-2e/pages/10-2-skeletal-muscle"),
}

# Indonesian name, English/GTEx name, category, aliases, definition, context,
# short clarification, related entry names, source. All text is bundled locally.
ENTRIES = [
    ("DNA", "Deoxyribonucleic acid", "Biologi molekuler", "asam deoksiribonukleat",
     "Molekul yang menyimpan informasi genetik pada sel.",
     "Informasi pada DNA menjadi dasar pembentukan RNA. Data ekspresi RNA tidak menunjukkan urutan DNA secara langsung.",
     "Gen merupakan bagian dari DNA, bukan sinonim untuk seluruh DNA.", ("Gen", "RNA", "Transkripsi"), "dna"),
    ("Ekspresi gen", "Gene expression", "Biologi molekuler", "aktivitas gen",
     "Penggunaan informasi pada gen untuk menghasilkan RNA dan, pada gen pengode protein, protein.",
     "Setiap nilai TPM menunjukkan kelimpahan RNA relatif sebuah gen pada satu sampel. Nilai tersebut tidak sama dengan jumlah sel.",
     "Pengukuran RNA menunjukkan kelimpahan transkrip, bukan jumlah protein secara langsung.", ("Gen", "RNA", "Transkripsi", "TPM"), "expression"),
    ("Gen", "Gene", "Biologi molekuler", "gen pengode protein, gen nonpengode",
     "Bagian DNA yang informasinya digunakan untuk menghasilkan RNA. Sebagian gen mengode protein, sebagian menghasilkan RNA nonpengode.",
     "Satu gen dapat menghasilkan beberapa bentuk transkrip. Data tingkat gen merangkum ekspresi pada tingkat gen, bukan setiap transkrip secara terpisah.",
     "Tidak semua gen menghasilkan protein.", ("DNA", "Ekspresi gen", "RNA", "Transkrip"), "gene"),
    ("RNA", "Ribonucleic acid", "Biologi molekuler", "asam ribonukleat",
     "Molekul yang berperan dalam penyampaian informasi genetik, pembentukan protein, dan pengaturan aktivitas sel.",
     "Data ekspresi gen mengukur RNA yang diperoleh dari sampel jaringan.",
     "mRNA adalah salah satu jenis RNA. Jenis lain mencakup rRNA, tRNA, dan RNA pengatur.", ("mRNA", "Transkripsi", "Transkriptom"), "rna"),
    ("mRNA", "Messenger RNA", "Biologi molekuler", "RNA pembawa pesan, RNA duta",
     "Jenis RNA yang membawa informasi dari gen pengode protein untuk dibaca oleh ribosom dalam pembentukan protein.",
     "Kelimpahan mRNA membantu menggambarkan ekspresi gen pengode protein pada jaringan.",
     "Jumlah mRNA tidak selalu sebanding langsung dengan jumlah protein yang terbentuk.", ("RNA", "Gen", "Protein"), "mrna"),
    ("Protein", "Protein", "Biologi molekuler", "polipeptida, asam amino",
     "Molekul yang tersusun dari rantai asam amino dan menjalankan fungsi seperti penyusunan struktur, kerja enzim, dan pengaturan sel.",
     "Ekspresi RNA dapat berkaitan dengan pembentukan protein, tetapi nilai TPM bukan hasil pengukuran protein.",
     "Protein dibentuk berdasarkan informasi pada mRNA dari gen pengode protein.", ("Gen", "mRNA", "Ekspresi gen"), "protein"),
    ("Transkripsi", "Transcription", "Biologi molekuler", "sintesis RNA",
     "Proses pembentukan RNA dengan menggunakan salah satu untai DNA sebagai cetakan.",
     "RNA yang terbentuk dapat menjadi mRNA atau menjalankan fungsi lain tanpa diterjemahkan menjadi protein.",
     "Transkripsi menghasilkan RNA, bukan langsung menghasilkan protein.", ("DNA", "RNA", "Transkrip"), "transcription"),
    ("Transkrip", "RNA transcript", "Biologi molekuler", "transcript, isoform",
     "Molekul RNA yang dihasilkan melalui transkripsi suatu bagian DNA.",
     "Satu gen dapat menghasilkan bentuk transkrip yang berbeda. Nilai ekspresi tingkat gen merangkum informasi transkrip dari gen tersebut.",
     "Transkrip dan gen bukan satuan yang selalu berkorespondensi satu banding satu.", ("Gen", "Transkripsi", "Transkriptom"), "transcriptome"),
    ("Transkriptom", "Transcriptome", "Biologi molekuler", "transcriptomics, transkriptomik",
     "Keseluruhan transkrip RNA yang terdapat pada sel atau jaringan dalam kondisi dan waktu tertentu.",
     "Komposisi transkriptom dapat berbeda antarjaringan meskipun banyak sel memiliki DNA yang sama.",
     "Transkriptom menggambarkan RNA, sedangkan genom merujuk keseluruhan materi genetik.", ("RNA", "Transkrip", "RNA-seq"), "transcriptome"),
    ("Jaringan", "Tissue", "Jaringan dan anatomi", "jenis jaringan, tissue type",
     "Kumpulan sel dan komponen pendukung yang bekerja bersama menjalankan fungsi tertentu dalam tubuh.",
     "Satu sampel jaringan dapat memuat beberapa tipe sel. Perbedaan lokasi dan fungsi jaringan berkaitan dengan pola ekspresi gennya.",
     "Nama jaringan menunjukkan asal biologis sampel, bukan nama penyakit.", ("Sampel", "Bulk RNA-seq"), "tissue"),
    ("Adiposa subkutan", "Adipose - Subcutaneous", "Jaringan dan anatomi", "lemak subkutan, subcutaneous fat",
     "Jaringan lemak di bawah kulit yang berperan dalam penyimpanan energi, isolasi panas, dan perlindungan tubuh.",
     "Pada label GTEx ini, sampel berasal dari jaringan adiposa di bawah kulit.",
     "Berbeda dari adiposa viseral yang berada di rongga perut.", ("Adiposa viseral", "Jaringan"), "adipose"),
    ("Adiposa viseral", "Adipose - Visceral (Omentum)", "Jaringan dan anatomi", "lemak viseral, lemak visceral, omentum",
     "Jaringan lemak di bagian dalam rongga perut. Omentum adalah lipatan jaringan yang berhubungan dengan organ pencernaan.",
     "Label GTEx ini secara khusus merujuk sampel adiposa pada omentum, bukan seluruh lokasi lemak viseral.",
     "Lokasinya berbeda dari adiposa subkutan di bawah kulit.", ("Adiposa subkutan", "Jaringan"), "adipose"),
    ("Aorta", "Artery - Aorta", "Jaringan dan anatomi", "arteri aorta",
     "Arteri besar yang menyalurkan darah dari ventrikel kiri jantung menuju sirkulasi tubuh.",
     "Sampel dengan label ini berasal dari jaringan dinding aorta.",
     "Jaringan pembuluh darah berbeda dari jaringan otot jantung.", ("Ventrikel kiri", "Arteri koroner", "Arteri tibialis"), "arteries"),
    ("Arteri koroner", "Artery - Coronary", "Jaringan dan anatomi", "coronary artery, pembuluh koroner",
     "Arteri yang memasok darah ke jaringan otot jantung.",
     "Label ini merujuk jaringan arteri koroner, bukan jaringan bilik atau serambi jantung.",
     "Arteri koroner berada pada organ jantung, tetapi jenis jaringannya tetap pembuluh darah.", ("Aorta", "Apendiks atrium", "Ventrikel kiri"), "heart"),
    ("Arteri tibialis", "Artery - Tibial", "Jaringan dan anatomi", "tibial artery, arteri tungkai bawah",
     "Arteri di tungkai bawah yang menyalurkan darah ke bagian tungkai dan kaki.",
     "Label GTEx ini menunjukkan lokasi asal sampel jaringan arteri pada tungkai bawah.",
     "Tibial pada label ini tidak berarti sampel jaringan tulang tibia.", ("Aorta", "Arteri koroner", "Jaringan"), "arteries"),
    ("Apendiks atrium", "Heart - Atrial Appendage", "Jaringan dan anatomi", "aurikula, atrial appendage, apendiks serambi",
     "Bagian menyerupai kantong yang menonjol dari atrium atau serambi jantung.",
     "Sampel berasal dari bagian atrial appendage. Nama ini tidak mewakili seluruh jaringan atrium.",
     "Berbeda dari apendiks pada saluran pencernaan.", ("Ventrikel kiri", "Arteri koroner", "Jaringan"), "heart"),
    ("Ventrikel kiri", "Heart - Left Ventricle", "Jaringan dan anatomi", "bilik kiri, left ventricle",
     "Bilik jantung yang memompa darah menuju aorta untuk diedarkan ke tubuh.",
     "Jaringan ventrikel kiri terutama berperan dalam kontraksi pemompaan darah.",
     "Lokasinya berbeda dari apendiks atrium meskipun keduanya merupakan bagian jantung.", ("Aorta", "Apendiks atrium", "Arteri koroner"), "heart"),
    ("Hati", "Liver", "Jaringan dan anatomi", "hepar",
     "Organ yang menjalankan berbagai proses metabolisme, mengolah zat dalam darah, dan menghasilkan empedu.",
     "Sampel hati dapat memuat sel hati serta sel pembuluh darah, imun, dan jaringan pendukung.",
     "Istilah hati menunjukkan organ asal sampel, bukan ukuran fungsi hati.", ("Pankreas", "Jaringan", "Bulk RNA-seq"), "digestion"),
    ("Otot rangka", "Muscle - Skeletal", "Jaringan dan anatomi", "skeletal muscle, otot skeletal",
     "Otot yang umumnya terhubung dengan rangka dan berperan dalam gerakan tubuh serta pemeliharaan postur.",
     "Jaringan ini memiliki pola ekspresi yang berkaitan dengan struktur dan kontraksi serat otot.",
     "Otot rangka berbeda dari otot jantung dan otot polos.", ("Jaringan", "Protein", "Ekspresi gen"), "muscle"),
    ("Pankreas", "Pancreas", "Jaringan dan anatomi", "pulau langerhans, pancreatic tissue",
     "Organ yang menghasilkan enzim pencernaan dan memiliki bagian endokrin penghasil hormon, termasuk insulin dan glukagon.",
     "Sampel jaringan pankreas memuat komponen eksokrin dan endokrin, bukan hanya sel penghasil insulin.",
     "Ekspresi pada sampel gabungan dipengaruhi campuran tipe sel di dalamnya.", ("Hati", "Jaringan", "Bulk RNA-seq"), "digestion"),
    ("Hipofisis", "Pituitary", "Jaringan dan anatomi", "kelenjar pituitari, pituitary gland",
     "Kelenjar di dasar otak yang, bersama hipotalamus, berperan dalam pengaturan berbagai fungsi hormonal tubuh.",
     "Hormon hipofisis berhubungan dengan pertumbuhan, reproduksi, dan pengaturan kelenjar lain.",
     "Hipofisis berbeda dari kelenjar tiroid yang berada di leher.", ("Tiroid", "Kelenjar adrenal", "Jaringan"), "pituitary"),
    ("Kelenjar adrenal", "Adrenal Gland", "Jaringan dan anatomi", "suprarenal, adrenal",
     "Kelenjar di atas ginjal yang menghasilkan hormon untuk respons stres serta pengaturan metabolisme, garam, dan air.",
     "Kelenjar adrenal memiliki bagian korteks dan medula dengan produk hormon yang berbeda.",
     "Letaknya berdekatan dengan ginjal, tetapi merupakan organ endokrin tersendiri.", ("Hipofisis", "Tiroid", "Jaringan"), "adrenal"),
    ("Tiroid", "Thyroid", "Jaringan dan anatomi", "kelenjar tiroid, thyroid gland",
     "Kelenjar di bagian depan leher yang menghasilkan hormon yang membantu mengatur metabolisme tubuh.",
     "Hormon tiroid memengaruhi penggunaan energi dan berbagai proses pertumbuhan serta perkembangan.",
     "Label Thyroid merujuk jaringan kelenjar tiroid, bukan suatu diagnosis.", ("Hipofisis", "Kelenjar adrenal", "Jaringan"), "thyroid"),
    ("RNA-seq", "RNA sequencing", "Pengukuran ekspresi", "sekuensing RNA, RNA sequencing",
     "Teknik untuk mempelajari jenis dan kelimpahan RNA dengan membaca urutan molekul yang berasal dari RNA sampel.",
     "Pada metode yang umum, RNA diubah menjadi DNA komplementer sebelum disekuensing. Hasilnya diolah menjadi ukuran ekspresi gen.",
     "Data TPM merupakan hasil pengolahan, bukan bacaan sekuensing mentah.", ("RNA", "Transkriptom", "TPM", "Bulk RNA-seq"), "rnaseq"),
    ("Bulk RNA-seq", "Bulk RNA sequencing", "Pengukuran ekspresi", "RNA gabungan, komposisi sel, bulk tissue",
     "Pengukuran RNA gabungan dari banyak sel dalam satu sampel jaringan.",
     "Nilai ekspresi dipengaruhi aktivitas gen pada sel dan proporsi tipe sel yang menyusun sampel.",
     "Pengukuran gabungan tidak menunjukkan ekspresi setiap sel secara terpisah.", ("RNA-seq", "Jaringan", "Sampel", "TPM"), "rnaseq"),
    ("TPM", "Transcripts Per Million", "Pengukuran ekspresi", "kelimpahan RNA relatif",
     "Ukuran kelimpahan RNA relatif yang memperhitungkan panjang gen atau transkrip dan diskalakan terhadap total ekspresi dalam sampel.",
     "Pada hasil lengkap, nilai diskalakan per sejuta. Setelah hanya sebagian gen diambil, jumlah TPM subset tidak harus satu juta.",
     "TPM dapat berupa bilangan desimal. Nilainya bukan jumlah molekul RNA absolut atau jumlah sel.", ("RNA-seq", "Ekspresi gen", "Transkrip"), "tpm"),
    ("Sampel", "Biological sample", "Pengukuran ekspresi", "sample, spesimen, SAMPID",
     "Bahan biologis yang diambil dari individu untuk pemeriksaan atau pengukuran, misalnya potongan jaringan.",
     "Satu sampel ekspresi gen memuat pengukuran banyak gen dan memiliki identitas serta asal jaringan.",
     "Sampel, donor, dan gen adalah satuan yang berbeda.", ("Donor", "Jaringan", "Ekspresi gen"), "gtex"),
    ("Donor", "Tissue donor", "Pengukuran ekspresi", "individu, subject, SUBJID",
     "Individu asal bahan biologis yang digunakan dalam penelitian.",
     "Satu donor dapat menyumbangkan beberapa sampel dari lokasi jaringan yang berbeda.",
     "Beberapa sampel tidak selalu berarti beberapa individu yang berbeda.", ("Sampel", "Jaringan"), "gtex"),
]


GLOSSARY_STYLE = """
<style>
[data-testid="stAppViewContainer"]:has(.glossary-page-marker) { background: #fff; }
[data-testid="stAppViewContainer"]:has(.glossary-page-marker) [data-testid="stMain"] .block-container {
    padding: 1.4rem 2rem 1.5rem;
}
[data-testid="stElementContainer"]:has(.glossary-page-marker),
[data-testid="stAppViewContainer"]:has(.glossary-page-marker) [data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] style) { display: none; }
[data-testid="stAppViewContainer"]:has(.glossary-page-marker) h1 { font-size: 2.4rem !important; }
[data-testid="stAppViewContainer"]:has(.glossary-page-marker) [data-testid="stCaptionContainer"] p { font-size: .95rem; }
.st-key-glossary_filters input { font-size: 1rem; }
.st-key-glossary_filters { border-top: 1px solid #e2e8ec; padding-top: .9rem; margin-bottom: .6rem; }
.st-key-glossary_index .stButton > button {
    width: 100%; justify-content: flex-start; border: 0; border-bottom: 1px solid #e2e8ec;
    border-radius: 0; background: transparent; color: #172033; padding: .55rem .8rem; min-height: 2.9rem;
}
.st-key-glossary_index button > div { justify-content: flex-start !important; }
.st-key-glossary_index button p { text-align: left; font-size: 1rem; }
.st-key-glossary_index .stButton > button:hover { background: #f0f8f8; color: #087173; }
.st-key-glossary_index .stButton > button[kind="primary"] {
    background: #e9f5f5 !important; color: #087173 !important; border-left: 4px solid #0f8b8d; font-weight: 650;
}
.st-key-glossary_index .stButton > button[kind="primary"] * { color: #087173 !important; }
.st-key-glossary_index { gap: 0; }
.st-key-glossary_index [data-testid="stVerticalBlock"] { gap: 0; }
.st-key-glossary_detail { border-left: 1px solid #e2e8ec; padding: .25rem 0 0 2rem; min-height: 530px; }
.st-key-glossary_detail h2 { font-size: 2.4rem !important; margin: .1rem 0 .1rem; }
.st-key-glossary_detail [data-testid="stMarkdownContainer"] p { max-width: 64ch; line-height: 1.65; font-size: 1rem; }
.st-key-glossary_detail h3, [data-testid="stAppViewContainer"]:has(.glossary-page-marker) h3 { font-size: 1.2rem !important; }
.st-key-glossary_detail hr { margin: .6rem 0; }
.st-key-glossary_related .stButton > button {
    min-height: 2rem; padding: .1rem .4rem; color: #087173; text-decoration: underline;
}
@media (max-width: 640px) {
    [data-testid="stAppViewContainer"]:has(.glossary-page-marker) [data-testid="stMain"] .block-container { padding: 1rem; }
    .st-key-glossary_detail { border-left: 0; border-top: 1px solid #e2e8ec; padding: 1rem 0 0; min-height: 0; }
}
</style>
"""


def glossary_table() -> pd.DataFrame:
    return pd.DataFrame(ENTRIES, columns=[
        "term", "english", "category", "aliases", "definition", "context", "note", "related", "source",
    ])


def find_terms(query: str = "", category: str = "Semua kategori") -> pd.DataFrame:
    rows = glossary_table()
    if category not in {"Semua", "Semua kategori"}:
        rows = rows.loc[rows["category"].eq(category)]
    query = query.strip().casefold()
    if query and not rows.empty:
        searchable = rows[["term", "english", "aliases", "definition", "context"]].agg(" ".join, axis=1).str.casefold()
        rows = rows.loc[searchable.str.contains(query, regex=False)].copy()
        rows["exact"] = rows.apply(
            lambda row: query in [row["term"].casefold(), row["english"].casefold(),
                                  *[alias.strip().casefold() for alias in row["aliases"].split(",")]],
            axis=1,
        ) if not rows.empty else pd.Series(dtype=bool)
        rows = rows.sort_values(["exact", "term"], ascending=[False, True], key=lambda col: col.str.casefold() if col.dtype == object else col).drop(columns="exact")
    else:
        rows = rows.sort_values("term", key=lambda values: values.str.casefold())
    return rows.reset_index(drop=True)


def _select_term(term: str, reset_filters: bool = False) -> None:
    st.session_state.glossary_selected_term = term
    if reset_filters:
        st.session_state.glossary_search = ""
        st.session_state.glossary_category = "Semua kategori"


def render_glossary_page() -> None:
    st.markdown(GLOSSARY_STYLE, unsafe_allow_html=True)
    st.markdown('<div class="glossary-page-marker"></div>', unsafe_allow_html=True)
    st.title("Kamus Biologi")
    st.caption("Kenali istilah dan jaringan yang digunakan dalam data ekspresi gen.")
    categories = ["Semua kategori", "Biologi molekuler", "Jaringan dan anatomi", "Pengukuran ekspresi"]
    # Drop obsolete category values left in sessions from the previous dictionary.
    if st.session_state.get("glossary_category", categories[0]) not in categories:
        st.session_state.glossary_category = categories[0]
    with st.container(key="glossary_filters"):
        search, category_filter = st.columns([2.4, 1], gap="medium")
        query = search.text_input("Cari istilah", placeholder="Cari nama Indonesia atau label GTEx", key="glossary_search", label_visibility="collapsed")
        category = category_filter.selectbox("Kategori", categories, key="glossary_category", label_visibility="collapsed")
    rows = find_terms(query, category)
    if rows.empty:
        st.info("Istilah tidak ditemukan. Coba nama lain atau pilih Semua kategori.")
        return

    names = rows["term"].tolist()
    selected = st.session_state.get("glossary_selected_term", "Ekspresi gen")
    if selected not in names:
        selected = names[0]
    st.session_state.glossary_selected_term = selected
    row = rows.loc[rows["term"].eq(selected)].iloc[0]

    index, detail = st.columns([.9, 2.15], gap="large")
    with index:
        st.subheader("Daftar istilah")
        with st.container(height=530, border=False, key="glossary_index"):
            for term in names:
                st.button(term, key=f"glossary_term_{term}", type="primary" if term == selected else "secondary",
                          width="stretch", on_click=_select_term, args=(term,))
    with detail, st.container(key="glossary_detail"):
        st.caption(row["category"])
        st.header(row["term"], anchor=False)
        st.caption(row["english"])
        st.write(row["definition"])
        st.write(row["note"])
        st.divider()
        st.subheader("Dalam data ekspresi")
        st.write(row["context"])
        st.divider()
        st.subheader("Istilah terkait")
        with st.container(horizontal=True, key="glossary_related"):
            for term in row["related"]:
                st.button(term, type="tertiary", key=f"glossary_related_{term}",
                          on_click=_select_term, args=(term, True))
        st.divider()
        reference, download = st.columns([1.35, 1], gap="medium")
        with reference, st.expander("Rujukan definisi"):
            title, address = SOURCES[row["source"]]
            st.write(title)
            st.text(address)
        # Export the entire filtered index, not only the open definition.
        export = rows.drop(columns=["related", "source"]).copy()
        export["Rujukan"] = rows["source"].map(lambda key: SOURCES[key][0])
        export["Alamat rujukan"] = rows["source"].map(lambda key: SOURCES[key][1])
        export = export.rename(columns={"term": "Istilah", "english": "Nama Inggris atau GTEx", "category": "Kategori",
                                        "aliases": "Nama terkait", "definition": "Definisi", "context": "Dalam data ekspresi", "note": "Penjelasan"})
        with download:
            download_button("Unduh kamus", export.to_csv(index=False).encode("utf-8-sig"),
                               file_name="kamus_biologi_gtex.csv", mime="text/csv", icon=":material/download:", width="stretch")
