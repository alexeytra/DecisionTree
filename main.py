from DecisionTree import DecisionTree
import pandas as pd

if __name__ == "__main__":
    df = pd.DataFrame.from_csv('weather-trainer.csv')
    print("-=Выборка=- \n", df)
    question = list(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    tree = DecisionTree(max_deep=3, min_samples_split=2)
    tree.fit(X, y)

    # test
    print("-=Тестовые данные=-")
    df = pd.DataFrame.from_csv('weather-test.csv')
    X = df
    result = tree.predict(X)
    df[question[-1]] = result
    print(df)