from data_loader import load_data, split_data
from preprocessing import KFoldTargetEncoder
from model import get_model
from evaluate import evaluate_model

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "target_encoding_dataset_5lakh.csv")


def main():

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Target Encoding
    encoder = KFoldTargetEncoder(col="user_id", n_splits=5)

    X_train = encoder.fit_transform(X_train, y_train)
    encoder.fit(X_train.assign(user_id=df.loc[X_train.index, "user_id"]), y_train)

    X_test = encoder.transform(X_test)

    # 4. Train Model
    model = get_model()
    model.fit(X_train, y_train)

    # 5. Evaluate
    evaluate_model(model, X_test, y_test)
    print(y_train.mean())


if __name__ == "__main__":
    main()