import pandas as pd
from sklearn.model_selection import train_test_split

'''
Here is a demo code to create a next day prediction dataset for APPL that you might find useful
'''

# Importing dataset
df_apple = pd.read_csv("AAPL.csv")
# You might not need date information for next day forcasting
remove_features = ["Date"]
# Create the "Prediction" column by shifting the "Close" column down by one row
df_apple["Prediction"] = pd.Series(np.append(df["Close"][1:].to_numpy(), [0]))
df_apple.drop(df.tail(1).index, inplace=True)
df_apple.drop(remove_features, axis=1, inplace=True)
# Create X,y from the dataset
X = np.array(df.drop(["Prediction"], axis=1))
y = np.array(df["Prediction"])
[X_train, X_test, y_train, y_test] = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
# normalization to training dataset comes after here.