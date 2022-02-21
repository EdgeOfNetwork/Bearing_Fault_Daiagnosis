import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # path = "/content/drive/MyDrive/Bearing/bearing_test_1/"
    df = pd.read_csv("bearing_concat.csv")

    features = df.columns[:400]
    target = 'target'
    random_seed = 42

    ## Split the data into train and validation set
    x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.20,
                                                        random_state=random_seed, shuffle=True)

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    x_train = scaling(x_train)
    x_test = scaling(x_test)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    n_classes = len(np.unique(y_train))