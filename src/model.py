from sklearn.ensemble import RandomForestClassifier


def get_model():
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    return model