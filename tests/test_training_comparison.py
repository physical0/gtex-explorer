"""Perbandingan Model: saved gallery models side by side, and when their scores are comparable."""
import json
import math
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from gtex_explorer.training.comparison import (
    available_bases, comparability_issues, default_selection, metric_frame, panel_membership,
    panel_overlap, settings_frame, tissue_frame,
)
from ui_helpers import enter_application

APP = Path(__file__).resolve().parents[1] / 'main.py'


def model(run_id, *, fingerprint='data-1', holdout=0, folds=5, cv=None, holdout_scores=None, selector='anova'):
    return {
        'run_id': run_id, 'fingerprint': fingerprint, 'filename': 'tpm.csv', 'method': selector,
        'created': 0.0, 'dataset': {'sample_count': 100, 'donor_count': 40, 'class_count': 3},
        'config': {'selector': selector, 'top_k': 1000, 'l1_c': 0.1, 'mi_neighbors': 3,
                   'holdout_percent': holdout, 'evaluate_cv': cv is not None, 'cv_folds': folds,
                   'xgboost': {'n_estimators': 300, 'learning_rate': 0.1}},
        'cv': cv, 'holdout': holdout_scores,
    }


def scores(f1, *, cv=False, tissues=None):
    keys = ['macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'accuracy']
    if cv:
        result = {**{'mean_' + key: f1 for key in keys}, **{'std_' + key: 0.01 for key in keys}, 'mean_jaccard': 0.5}
        if tissues:
            result['tissue_f1'] = [{'tissue': name, 'mean_f1': value, 'std_f1': 0.02} for name, value in tissues.items()]
        return result
    return {**{key: f1 for key in keys}, 'tissue_f1': tissues or {}, 'tissue_support': {name: 4 for name in tissues or {}}}


def test_same_dataset_and_split_is_comparable_and_each_difference_is_named():
    a = model('a', holdout=20, holdout_scores=scores(0.9))
    b = model('b', holdout=20, holdout_scores=scores(0.8))
    assert comparability_issues([a, b], 'holdout') == []
    issues = comparability_issues([a, model('c', fingerprint='data-2', holdout=30, holdout_scores=scores(0.8))], 'holdout')
    assert len(issues) == 2 and 'Dataset berbeda' in issues[0] and 'Porsi holdout berbeda' in issues[1]
    cv_a, cv_b = model('a', cv=scores(0.9, cv=True)), model('b', folds=3, cv=scores(0.9, cv=True))
    assert comparability_issues([cv_a, cv_b], 'cv') == ['Jumlah fold CV berbeda: Model A 5 fold; Model B 3 fold.']
    # A model without scores on this basis does not count against the others.
    assert comparability_issues([a, b, model('d', fingerprint='other')], 'holdout') == []
    assert 'Dataset berbeda' in comparability_issues(
        [a, {**b, 'fingerprint': None}], 'holdout')[0]


def test_with_more_than_two_models_each_difference_names_which_models_have_what():
    held = scores(0.9)
    other = {**model('b', fingerprint='data-2', holdout=20, holdout_scores=held), 'filename': 'lain.csv'}
    issues = comparability_issues([model('a', holdout=20, holdout_scores=held), other,
                                   model('c', holdout=30, holdout_scores=held)], 'holdout')
    assert issues == [
        'Dataset berbeda: Model A dan C memakai tpm.csv; Model B memakai lain.csv.',
        'Porsi holdout berbeda: Model A dan B 20%; Model C 30%, sehingga data development dan holdout tidak sama.',
    ]
    renamed = model('b', fingerprint='data-2', holdout=20, holdout_scores=held)
    assert comparability_issues([model('a', holdout=20, holdout_scores=held), renamed,
                                 model('c', holdout=20, holdout_scores=held)], 'holdout') == [
        'Dataset berbeda: Model A dan C memakai tpm.csv; Model B memakai tpm.csv. '
        'Nama berkasnya sama, tetapi sampel atau nilainya berbeda.']
    cv = scores(0.9, cv=True)
    four = [model('a', cv=cv), model('b', cv=cv, folds=3), model('c', cv=cv), model('d', cv=cv, holdout=20)]
    assert comparability_issues(four, 'cv') == [
        'Porsi holdout berbeda: Model A, B, dan C tanpa holdout; Model D 20%, sehingga data development dan holdout '
        'tidak sama.',
        'Jumlah fold CV berbeda: Model A, C, dan D 5 fold; Model B 3 fold.',
    ]


def test_notes_name_three_and_four_models_as_an_indonesian_list():
    from gtex_explorer.training.comparison import shared_data_note, unscored_note
    held, cv = scores(0.9), scores(0.9, cv=True)
    three = [model(x, holdout=20, holdout_scores=held) for x in 'abc']
    assert shared_data_note(three, 'holdout').startswith('Ketiga model dilatih pada dataset tpm.csv')
    four = [model(x, holdout=20, cv=cv) for x in 'abcd']
    assert shared_data_note(four, 'cv').startswith('Keempat model dilatih')
    mixed = [model('a', cv=cv), model('b', cv=cv), model('c', cv=cv), model('d', holdout=20, holdout_scores=held)]
    assert shared_data_note(mixed, 'cv').startswith('Model A, B, dan C dilatih')
    # Two models without a score, for two reasons.
    reasons = [model('a', holdout=20, holdout_scores=held), model('b', cv=cv), model('c', holdout=20),
               model('d', holdout=20, holdout_scores=held)]
    assert unscored_note(reasons, 'holdout') == (
        'Model B dilatih tanpa data holdout dan Model C tidak menyimpan skor holdout, jadi keduanya tidak tampil '
        'pada skor Evaluasi holdout. Panel gen dan pengaturannya tetap dibandingkan di bawah.')


def test_bases_need_two_scored_models_and_default_prefers_newest_matching_split():
    holdout_a = model('a', holdout=20, holdout_scores=scores(0.9))
    unscored = model('b')
    holdout_c = model('c', fingerprint='data-2', holdout=20, holdout_scores=scores(0.9))
    holdout_d = model('d', holdout=20, holdout_scores=scores(0.7), cv=scores(0.8, cv=True))
    assert available_bases([holdout_a, unscored]) == []
    assert available_bases([holdout_a, holdout_d]) == ['holdout']
    assert default_selection([holdout_a, unscored, holdout_c, holdout_d]) == ['a', 'd']
    assert default_selection([holdout_a, holdout_c]) == ['a', 'c']
    assert default_selection([unscored, model('e')]) == ['b', 'e']
    many = [model(name, holdout=20, holdout_scores=scores(0.9)) for name in 'pqrstu']
    assert default_selection(many) == list('pqrs')
    # The newest model has only CV; the older holdout models are the comparable group.
    cv_only = model('n', cv=scores(0.9, cv=True))
    assert default_selection([cv_only, holdout_a, unscored, holdout_d]) == ['a', 'd']
    # CV models are comparable only with the same fold count.
    assert default_selection([cv_only, model('m', folds=3, cv=scores(0.9, cv=True)), model('o')]) == ['n', 'm']
    assert default_selection([cv_only, unscored, model('k', cv=scores(0.8, cv=True))]) == ['n', 'k']


def test_score_frames_keep_missing_models_as_gaps():
    a = model('a', cv=scores(0.9, cv=True, tissues={'Lung': 0.8, 'Liver': 1.0}))
    b = model('b')
    metrics = metric_frame([a, b], 'cv')
    assert len(metrics) == 10
    assert metrics.query("model == 'A' and metric == 'Macro-F1'").score.item() == 0.9
    assert metrics.query("model == 'B'").score.isna().all()
    tissues = tissue_frame([a, b], 'cv')
    assert set(tissues.model) == {'A'} and tissues.sd.tolist() == [0.02, 0.02]
    holdout = tissue_frame([model('c', holdout_scores=scores(0.9, tissues={'Lung': 0.5}))], 'holdout')
    assert holdout.samples.tolist() == [4]


def test_panel_overlap_counts_shared_genes_and_ranks_membership():
    panels = {'A': ['g1', 'g2', 'g3'], 'B': ['g3', 'g1', 'g4'], 'C': ['g5']}
    overlap = panel_overlap(panels)
    first = overlap.iloc[0]
    assert (first['first'], first['second'], first['shared']) == ('A', 'B', 2)
    assert math.isclose(first['jaccard'], 2 / 4)
    assert overlap['shared'].tolist() == [2, 0, 0]
    membership = panel_membership(panels)
    assert membership.gene.tolist()[:2] == ['g1', 'g3']
    assert membership.set_index('gene').loc['g1', ['rank_A', 'rank_B']].tolist() == [1, 2]
    assert pd.isna(membership.set_index('gene').loc['g5', 'rank_A'])
    assert membership.models.tolist() == [2, 2, 1, 1, 1]


def test_settings_mark_only_what_differs():
    frame = settings_frame([model('a'), model('b', selector='mutual_information')]).set_index('Pengaturan')
    assert frame.loc['Metode seleksi fitur', 'differs'] and frame.loc['n_neighbors', 'differs']
    assert frame.loc['n_neighbors', 'A'] == '-' and frame.loc['n_neighbors', 'B'] == '3'
    assert not frame.loc['Dataset', 'differs'] and not frame.loc['Putaran boosting', 'differs']
    assert frame.loc['Top-k gen', 'A'] == '1.000'
    # No model uses L1, so its settings are left out.
    assert 'C (regularisasi)' not in frame.index and 'Maksimum iterasi' not in frame.index


def test_settings_use_the_labels_and_order_of_the_training_form():
    l1 = model('b', selector='l1', holdout=20)
    l1['config'].update(l1_max_iter=20000, l1_tol=0.001, l1_class_weight='balanced')
    frame = settings_frame([model('a'), l1]).set_index('Pengaturan')
    rows = frame.index.tolist()
    expected = ['Data holdout', 'Pra-pemrosesan', 'Metode seleksi fitur', 'Top-k gen', 'C (regularisasi)',
                'Evaluasi tambahan', 'Putaran boosting', 'Learning rate', 'Maksimum iterasi']
    assert [row for row in rows if row in expected] == expected
    assert frame.loc['Data holdout', 'A'] == 'Tidak ada' and frame.loc['Data holdout', 'B'] == '20%'
    assert frame.loc['Pembagian', 'A'] == '-' and frame.loc['Pembagian', 'B'] == 'Donor terpisah'
    assert frame.loc['C (regularisasi)', 'B'] == '0,1' and frame.loc['C (regularisasi)', 'A'] == '-'
    assert 'Classifier' not in rows and 'Thread CPU' not in rows


@pytest.fixture
def two_cv_runs(tmp_path, monkeypatch):
    from test_training_gallery import make_run
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import start_job
    from gtex_explorer.training.worker import run_job
    data, state, first = make_run(tmp_path, monkeypatch, cv=True)
    assert run_job(first, data, state) == 0
    second = start_job(validate_training_config({'selector': 'baseline', 'evaluate_cv': True, 'cv_folds': 3,
        'xgboost': {'n_estimators': 2}}, candidate_count=6), data, state).run_id
    assert run_job(second, data, state) == 0
    return state, first, second


def comparison_app(state):
    return AppTest.from_string(
        'from pathlib import Path\n'
        'from gtex_explorer.ui.pages.training_comparison import render_training_comparison_page\n'
        f'render_training_comparison_page(Path({str(state)!r}))', default_timeout=60).run()


def test_page_lines_up_two_cv_models_on_the_same_folds(two_cv_runs):
    state, first, second = two_cv_runs
    app = comparison_app(state)
    assert not app.exception
    # Newest first, both chosen; only CV is available since neither has a holdout.
    assert app.multiselect(key='training_compare_runs').value == [second, first]
    assert app.segmented_control(key='training_compare_basis').value == 'cv'
    assert app.segmented_control(key='training_compare_basis').options == ['Evaluasi CV']
    assert not app.warning
    # The verdict, dataset and evaluation stay in the closed expander's header; the notes are inside it.
    notes = next(item for item in app.expander if 'Data evaluasi sama' in item.label)
    assert notes.label.endswith('· CV 3 fold') and not notes.proto.expanded
    assert any('dengan pembagian per donor yang sama' in item.value for item in notes.caption)
    headings = [item.value for item in app.subheader]
    assert headings == ['Metrik evaluasi', 'F1-score per jaringan', 'Panel gen final', 'Perbandingan pengaturan model']
    # The gene membership file is no longer offered.
    assert not any('keanggotaan' in str(item.proto).lower() for item in app.get('download_button'))
    metrics = app.dataframe[0].value
    assert list(metrics.columns) == ['Metrik', 'Model A', 'Model B']
    assert metrics['Metrik'].tolist()[0] == 'Macro-F1'
    assert all(' ± ' in value for value in metrics['Model A'])
    assert len(app.get('vega_lite_chart')) == 1
    assert [metric.label for metric in app.metric][:2] == ['Macro-F1', 'Macro-F1']
    overlap = next(frame.value for frame in app.dataframe if 'Pasangan' in frame.value.columns)
    assert overlap['Pasangan'].tolist() == ['Model A dan B']
    settings = app.dataframe[-1].value
    assert 'Metode seleksi fitur' in settings['Pengaturan'].tolist()
    assert 'Putaran boosting' not in settings['Pengaturan'].tolist()
    app.toggle(key='training_compare_differences').set_value(False).run()
    assert 'Putaran boosting' in app.dataframe[-1].value['Pengaturan'].tolist()


def test_page_needs_two_models_and_keeps_order_of_choice(two_cv_runs):
    state, first, second = two_cv_runs
    app = comparison_app(state)
    app.multiselect(key='training_compare_runs').set_value([first]).run()
    assert not app.exception
    assert any('minimal dua model' in item.value for item in app.info)
    assert not app.dataframe
    app.multiselect(key='training_compare_runs').set_value([first, second]).run()
    cards = [item.value for item in app.markdown if item.value.startswith('**') and 'gen' in item.value]
    assert 'ANOVA' in cards[0] and 'Baseline' in cards[1]


def test_empty_gallery_points_to_training(tmp_path):
    app = comparison_app(tmp_path)
    assert not app.exception
    assert any('minimal dua model terverifikasi' in item.value for item in app.info)
    app.button(key='training_compare_train').click().run()
    assert app.session_state.page == 'Pelatihan Model'


def test_sidebar_opens_the_comparison_page(tmp_path, monkeypatch):
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    app = enter_application(AppTest.from_file(str(APP), default_timeout=60).run())
    app.button(key='nav_Perbandingan Pelatihan').click().run()
    assert not app.exception
    assert 'Perbandingan model' in [title.value for title in app.title]
    assert next(item for item in app.sidebar.expander if item.label == 'Gallery Hasil Pelatihan').proto.expanded


def test_samples_fingerprint_ignores_only_the_candidate_count():
    from gtex_explorer.training.data import samples_fingerprint
    manifest = {'sample_ids': ['S1', 'S2'], 'candidate_count': 1000, 'files': {'raw_tpm.npy': 'aaa'}}
    assert samples_fingerprint(manifest) == samples_fingerprint({**manifest, 'candidate_count': 5000})
    assert samples_fingerprint(manifest) != samples_fingerprint({**manifest, 'files': {'raw_tpm.npy': 'bbb'}})


def test_a_different_candidate_count_is_comparable_and_explained():
    from gtex_explorer.training.comparison import _comparable, candidate_note
    a, b = model('a', holdout=20, holdout_scores=scores(0.9)), model('b', holdout=20, holdout_scores=scores(0.8))
    a['dataset']['candidate_count'], b['dataset']['candidate_count'] = 1000, 5000
    assert _comparable(a, b) and comparability_issues([a, b], 'holdout') == []
    note = candidate_note([a, b])
    assert note.startswith('Gen kandidat berbeda: Model A 1.000 gen; Model B 5.000 gen, jadi selisih skor')
    # The shared samples and split are said once, in the note above it.
    assert 'pembagian' not in note
    b['dataset']['candidate_count'] = 1000
    assert candidate_note([a, b]) is None


def test_older_runs_read_the_samples_identity_from_their_saved_dataset(tmp_path):
    import json as json_module
    from gtex_explorer.training.comparison import _samples_identity
    from gtex_explorer.training.data import samples_fingerprint, stable_hash
    manifest = {'sample_ids': ['S1'], 'candidate_count': 10, 'files': {'raw_tpm.npy': 'aaa'}}
    (tmp_path / 'manifest.json').write_text(json_module.dumps(manifest), encoding='utf-8')
    old_run = {'data_root': str(tmp_path), 'data_fingerprint': stable_hash(manifest)}
    assert _samples_identity(old_run) == samples_fingerprint(manifest)
    # A dataset folder that no longer matches the run keeps the stricter full fingerprint.
    assert _samples_identity({**old_run, 'data_fingerprint': 'other'}) == 'other'
    assert _samples_identity({'samples_fingerprint': 'saved', 'data_fingerprint': 'x'}) == 'saved'


def test_two_runs_on_one_upload_with_different_candidate_counts_compare_directly(tmp_path, monkeypatch):
    import json as json_module
    from test_training_shapes import train, upload
    from gtex_explorer.training.comparison import candidate_note, gallery_models
    files = upload([20, 20, 20])
    for candidates in [20, 40]:
        assert train(tmp_path, monkeypatch, files, {'holdout_percent': 25}, candidates=candidates)['status'] == 'completed'
    configs = [json_module.loads(path.read_text(encoding='utf-8')) for path in (tmp_path / 'runs').glob('run_*/config.json')]
    assert len({saved['data_fingerprint'] for saved in configs}) == 2
    assert len({saved['samples_fingerprint'] for saved in configs}) == 1
    models = gallery_models(tmp_path)
    assert comparability_issues(models, 'holdout') == []
    assert 'Gen kandidat berbeda' in candidate_note(models)
    app = comparison_app(tmp_path)
    assert not app.exception and not app.warning
    assert any('Data evaluasi sama' in item.label for item in app.expander)
    assert any('Gen kandidat berbeda' in item.value for item in app.caption)


def test_shared_data_note_names_the_dataset_and_the_split_the_scores_share():
    from gtex_explorer.training.comparison import shared_data_note
    held = {**scores(0.9), 'validation_samples': 20, 'validation_donors': 8}
    a = model('a', holdout=20, holdout_scores=held)
    b = model('b', holdout=20, holdout_scores=held)
    assert shared_data_note([a, b], 'holdout') == (
        'Kedua model dilatih pada dataset tpm.csv (100 sampel, 40 donor, 3 jaringan) dengan pembagian per donor '
        'yang sama: data development 80% dan data holdout 20% (20 sampel dari 8 donor). Skor diukur pada sampel '
        'holdout yang sama, jadi dapat dibandingkan langsung.')
    cv_a = model('c', cv=scores(0.9, cv=True))
    cv_b = model('d', cv=scores(0.8, cv=True))
    no_cv = model('e', holdout_scores=scores(0.7))
    note = shared_data_note([cv_a, cv_b, no_cv], 'cv')
    assert note.startswith('Model A dan B dilatih pada dataset tpm.csv')
    assert 'seluruh sampel dibagi menjadi 5 fold CV yang sama. Skor diukur pada data validation fold yang sama' in note
    with_holdout = model('f', holdout=20, cv=scores(0.9, cv=True))
    assert 'data development 80% dibagi menjadi 5 fold CV yang sama' in shared_data_note(
        [with_holdout, {**with_holdout, 'run_id': 'g'}], 'cv')


def test_shared_data_note_names_notebook_data_and_leaves_out_unknown_sizes():
    from gtex_explorer.training.comparison import shared_data_note
    from gtex_explorer.training.run_labels import NOTEBOOK_DATA
    a = {**model('a', holdout=20, holdout_scores=scores(0.9)), 'filename': NOTEBOOK_DATA, 'dataset': {}}
    note = shared_data_note([a, {**a, 'run_id': 'b'}], 'holdout')
    assert note.startswith('Kedua model dilatih pada data eksperimen notebook dengan pembagian')
    assert 'data holdout 20%. Skor' in note


def test_unscored_note_says_why_a_model_has_no_score():
    from gtex_explorer.training.comparison import unscored_note
    a = model('a', cv=scores(0.9, cv=True))
    b = model('b', cv=scores(0.8, cv=True))
    assert unscored_note([a, b], 'cv') is None
    no_cv = model('c', holdout=20, holdout_scores=scores(0.7))
    assert unscored_note([a, b, no_cv], 'cv') == (
        'Model C dilatih tanpa CV, jadi tidak tampil pada skor Evaluasi CV. '
        'Panel gen dan pengaturannya tetap dibandingkan di bawah.')
    assert unscored_note([no_cv, a, b], 'holdout') == (
        'Model B dan C dilatih tanpa data holdout, jadi tidak tampil pada skor Evaluasi holdout. '
        'Panel gen dan pengaturannya tetap dibandingkan di bawah.')


def test_candidate_note_counts_only_the_models_scored_and_groups_equal_counts():
    from gtex_explorer.training.comparison import candidate_note
    held, cv = scores(0.9), scores(0.9, cv=True)
    four = [model('a', holdout=20, holdout_scores=held), model('b', holdout=20, holdout_scores=held),
            model('c', holdout=20, holdout_scores=held), model('d', cv=cv)]
    for each, candidates in zip(four, [20, 40, 20, 40]):
        each['dataset'] = {**each['dataset'], 'candidate_count': candidates}
    assert candidate_note(four, 'holdout').startswith('Gen kandidat berbeda: Model A dan C 20 gen; Model B 40 gen, jadi')
    # Model D has no holdout score, so its count does not explain a holdout difference.
    four[1]['dataset']['candidate_count'] = 20
    assert candidate_note(four, 'holdout') is None


def test_identical_runs_get_different_picker_labels(tmp_path, monkeypatch):
    # The picker sends back its label, so two runs that only differ by id must not share one.
    from gtex_explorer.ui.pages.training_comparison import _option_label
    first, second = model('run_' + '1' * 32), model('run_' + '2' * 32)
    assert _option_label(first) != _option_label(second)
    assert _option_label(first).endswith('111111')


def test_tied_scores_mark_no_cell_or_card_as_the_highest():
    from gtex_explorer.ui.pages.training_comparison import _best
    scores_table = pd.DataFrame({'A': [1.0, 0.9, 0.8], 'B': [1.0, 0.95, None], 'C': [1.0, 0.95, 0.8]})
    best = _best(scores_table)
    # Every model ties on the first row, so nothing stands out; the second has two winners.
    assert best.values.tolist() == [[False, False, False], [False, True, True], [False, False, False]]


def test_tissue_chart_has_whole_percent_ticks_and_room_for_its_axis(two_cv_runs):
    state, _, _ = two_cv_runs
    app = comparison_app(state)
    assert not app.exception
    spec = json.loads(app.get('vega_lite_chart')[-1].proto.spec)
    assert spec['layer'][-1]['encoding']['x']['axis']['tickMinStep'] == 0.01
    tissue_count = len(spec['layer'][-1]['encoding']['y']['sort'])
    assert spec['height'] >= tissue_count * 40 + 56
