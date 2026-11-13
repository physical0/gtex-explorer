"""Form values the frame keeps while the user moves between pages."""

import streamlit as st

TRAINING_FORM_KEYS = (
    "training_selector",
    "training_candidates",
    "training_top_k",
    "training_l1_c",
    "training_l1_iterations",
    "training_l1_tol",
    "training_l1_weight",
    "training_mi_neighbors",
    "training_cv",
    "training_folds",
    "training_holdout",
    "training_log2",
    "training_standardize",
    "training_xgb_threads",
    *(
        "training_xgb_" + name
        for name in (
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "reg_alpha",
            "gamma",
        )
    ),
)


def keep_form_values() -> None:
    """Keep the training form filled in while other pages are open."""
    # Streamlit drops widget values for widgets not drawn in a run; re-assigning keeps the training form
    # intact while the user visits other pages. Workspace switches clear these keys separately.
    for key in TRAINING_FORM_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
