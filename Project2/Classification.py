from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading of the dataset
data = pd.read_csv(r"Dataset\spam_dataset.csv")

df = pd.DataFrame(data)

# print(df.isnull().sum())
fig, ax = plt.subplots(2, 3, figsize=(24, 10))

sns.countplot(
    x="is_spam",
    data=df,
    ax=ax[0][0]
)

ax[0][0].set_title("Spam vs Not Spam Count")
ax[0][0].set_xlabel("Class")
ax[0][0].set_ylabel("Count")
ax[0][0].set_xticks([0, 1])
ax[0][0].set_xticklabels(["Not Spam", "Spam"])

sns.countplot(
    x="has_link",
    hue="is_spam",
    data=df,
    ax=ax[0][1]
)

ax[0][1].set_title("Spam Count vs Link Presence")
ax[0][1].set_xlabel("Contains Link")
ax[0][1].set_ylabel("Count")
ax[0][1].set_xticks([0, 1])
ax[0][1].set_xticklabels(["No Link", "Has Link"])
ax[0][1].legend(title="Is Spam", labels=["Not Spam", "Spam"])

sns.scatterplot(
    data=df,
    x="message_length",
    y="num_keywords",
    hue="is_spam",
    alpha=0.6,
    ax=ax[0][2]
)

ax[0][2].set_title("Message Length vs Keywords")
ax[0][2].set_xlabel("Message Length")
ax[0][2].set_ylabel("Number of Keywords")

sns.boxplot(
    x="is_spam",
    y="message_length",
    data=df,
    ax=ax[1][0]
)

ax[1][0].set_title("Message Length Distribution")
ax[1][0].set_xlabel("Class")
ax[1][0].set_ylabel("Message Length")
ax[1][0].set_xticks([0, 1])
ax[1][0].set_xticklabels(["Not Spam", "Spam"])

features = [
    "num_keywords",
    "has_link",
    "message_length",
    "all_caps_words"
]

sns.heatmap(
    df[features + ["is_spam"]].corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax[1][1]
)

# plt.tight_layout(w_pad=6.0, h_pad=3.0)
# plt.show()


X = df[
    [
        "num_keywords",
        "has_link",
        "message_length",
        "all_caps_words"
    ]
]

y = df["is_spam"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(
    random_state=43
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# Check the model correctness
# print("Accuracy score of the model is:\n", accuracy_score(y_test, y_pred=y_pred))
# print("Confusion Matrix of the model is: \n", confusion_matrix(y_test, y_pred=y_pred))
# print("Classification report of the model is: \n", classification_report(y_test, y_pred=y_pred))

importance = pd.DataFrame(
    {
        "Features": X.columns,
        "Importance": model.feature_importances_
    }
)

importance = importance.sort_values(
    by="Importance",
    ascending=False
)
# print("\nImportance List:", importance)

# Prediction
sample1 = pd.DataFrame(
    {
        "num_keywords": [8],
        "has_link": [1],
        "message_length": [400],
        "all_caps_words": [8]
    }
)

sample2 = pd.DataFrame({
    "num_keywords": [1],
    "has_link": [0],
    "message_length": [150],
    "all_caps_words": [0]
})

prediction1 = model.predict(sample1)
prediction2 = model.predict(sample2)
print(prediction1,"\n", prediction2)