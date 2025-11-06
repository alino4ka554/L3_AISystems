import pandas as pd

file_path = "data.csv"
df = pd.read_csv(file_path)

df = df.drop(columns=['Unnamed: 32', 'id'])
# эквивалентно:
# df = df.loc[:, df.columns[10:]]
df.info()
df.describe()
import matplotlib
matplotlib.use('Agg')  # отключаем GUI backend
import matplotlib.pyplot as plt

df.hist(bins=10, figsize=(20, 20))
plt.suptitle("Гистограммы распределений признаков и целевой переменной")
plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# print(y_test.count())
model_gaussianNB = GaussianNB()
model_gaussianNB.fit(X_train, y_train)
y_pred = model_gaussianNB.predict(X_test)
print(accuracy_score(y_test, y_pred))
plt.figure(figsize=(24, 12))
plot_tree(
    model_gaussianNB,
    filled=True,
    feature_names=X.columns,
    class_names=['Malignant', 'Benign'],
    fontsize=10
)
plt.show()

cm = confusion_matrix(y_test, y_pred, labels=['M', 'B'])
