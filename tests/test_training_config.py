import pytest


def test_controls_have_bounds_and_normalization():
    from gtex_explorer.training.config import validate_training_config
    config = validate_training_config({'selector': 'l1', 'l1_c': 1.0, 'top_k': 1000}, cpu_count=2)
    assert config.top_k == 1000 and config.l1_c == 1
    assert config.xgboost_parameters['n_jobs'] == 2
    assert not config.evaluate_cv and config.cv_folds == 5
    baseline = validate_training_config({'selector': 'baseline', 'top_k': 23})
    assert baseline.top_k == 5000
    with pytest.raises(TypeError):
        config.xgboost_parameters['max_depth'] = 99


@pytest.mark.parametrize('values', [
    {'top_k': True}, {'top_k': 5001}, {'l1_c': float('nan')}, {'l1_c': 0},
    {'l1_max_iter': 99}, {'l1_tol': 0}, {'l1_class_weight': 'unknown'},
    {'mi_neighbors': 21}, {'cv_folds': 11}, {'seed': 11},
    {'xgboost': {'max_depth': 0}}, {'xgboost': {'n_estimators': 0}},
    {'xgboost': {'subsample': 0}}, {'xgboost': {'learning_rate': float('inf')}},
    {'xgboost': {'objective': 'binary:logistic'}}, {'xgboost': {'n_jobs': True}},
])
def test_invalid_controls_cannot_create_a_job(values):
    from gtex_explorer.training.config import validate_training_config
    with pytest.raises(ValueError):
        validate_training_config(values)


def test_identity_tracks_effective_controls_and_versions():
    from gtex_explorer.training.config import validate_training_config, configuration_fingerprint
    a = validate_training_config({'selector': 'anova', 'evaluate_cv': True})
    b = validate_training_config({'selector': 'anova', 'l1_c': 0.1, 'evaluate_cv': True})
    identity = lambda config, split='split': configuration_fingerprint(config, 'data', split, {'sklearn': '1'})
    assert identity(a) == identity(b)
    assert identity(a) != identity(a, 'other_split')
    assert identity(a) != identity(validate_training_config({'selector': 'anova', 'top_k': 500}))


def test_holdout_is_optional_and_leaves_older_runs_unchanged():
    from gtex_explorer.training.config import validate_training_config
    plain = validate_training_config({'selector': 'anova'})
    assert plain.holdout_percent is None and 'holdout_percent' not in plain.to_dict()
    chosen = validate_training_config({'selector': 'anova', 'holdout_percent': 20})
    assert chosen.to_dict()['holdout_percent'] == 20
    assert validate_training_config(chosen.to_dict()).holdout_percent == 20
    for bad in [0, 4, 51, 20.5, True]:
        with pytest.raises(ValueError):
            validate_training_config({'selector': 'anova', 'holdout_percent': bad})
    both = validate_training_config({'selector': 'anova', 'holdout_percent': 20, 'evaluate_cv': True})
    assert both.holdout_percent == 20 and both.evaluate_cv
