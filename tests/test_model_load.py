import mlflow

def test_mlflow_model_load():
    model_uri=(
        "models:/"
        "CustomerChurnModel"
        "/latest"
    )

    model=mlflow.pyfunc.load_model(
        model_uri
    )
    assert model is not None