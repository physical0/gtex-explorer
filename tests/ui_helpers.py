"""Enter the existing app explicitly before testing an application page."""


def enter_application(app, route='Prediksi Jaringan'):
    assert not app.exception
    app.button(key='workspace_select_default').click().run()
    app.button(key='workspace_open').click().run()
    assert not app.exception
    if route != 'Pelatihan Model':
        app.button(key='nav_' + route).click().run()
        assert not app.exception
    return app
