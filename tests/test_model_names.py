"""Model names on Prediksi Jaringan match the training pages."""
from gtex_explorer.platform.model_registry import EXPERIMENT_PREFIX, _display_name


def manifest(model_id, config=None, display_name='saved name'):
    return {'model_id': model_id, 'display_name': display_name, **({'config': config} if config else {})}


def test_user_models_are_named_like_their_training_run():
    config = {'selector': 'mutual_information', 'mi_neighbors': 5, 'top_k': 1000, 'log2_transform': False}
    assert _display_name(manifest('run_' + 'a' * 32, config)) == (
        'Mutual Information - n_neighbors = 5 - 1.000 gen - TPM + standardisasi'
    )


def test_bundled_models_are_marked_as_the_notebook_experiment():
    config = {'selector': 'baseline', 'top_k': 5000}
    assert _display_name(manifest('original_baseline_5000', config)) == (
        EXPERIMENT_PREFIX + 'Baseline - tanpa seleksi fitur - 5.000 gen'
    )


def test_a_model_without_a_readable_config_keeps_its_saved_name():
    assert _display_name(manifest('run_' + 'b' * 32)) == 'saved name'
    assert _display_name(manifest('run_' + 'b' * 32, {'selector': 'l1', 'top_k': 'many', 'l1_c': 1})) == 'saved name'


def test_numbers_use_indonesian_separators_everywhere():
    from gtex_explorer.display import count, decimal, method_name, percent, points
    assert count(5000) == '5.000' and count(None) == '-'
    assert decimal(0.05) == '0,05' and decimal(float('nan'), missing='Tidak tersedia') == 'Tidak tersedia'
    assert percent(0.9625) == '96,25%' and points(0.0049) == '0,49'
    assert method_name('l1', l1_c=0.1) == 'Regresi logistik L1 - C = 0,1'


def test_experiment_tables_name_configurations_like_the_training_pages():
    from gtex_explorer.experiments.reference_tables import load_reference_tables
    tables = load_reference_tables()
    labels = set(tables['cv_summary']['configuration_label'])
    assert 'Regresi logistik L1 - C = 0,1 - 1.000 gen' in labels
    assert 'Baseline - tanpa seleksi fitur - 5.000 gen' in labels
    assert not any('genes' in label or '—' in label for label in labels)
    assert set(tables['final_holdout']['configuration_label']) <= labels
