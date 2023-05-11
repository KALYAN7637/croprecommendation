import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
from sklearn.utils import resample
import streamlit as st
# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve
# Validation
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline, make_pipeline
# Feature Extraction
from sklearn.feature_selection import RFE

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Ensembles
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

#pio.templates.default = "plotly_white"


# Split Data to Training and Validation set

def read_in_and_split_data(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

#Spot-Check Algorithms

def GetModel():
    Models = []
    Models.append(('LR'   , LogisticRegression()))
    Models.append(('NB'   , GaussianNB()))
    return Models

#Spot-Check Normalized Models

def NormalizedModel(nameOfScaler):
    
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    pipelines = []
    pipelines.append((nameOfScaler+'LR'  , Pipeline([('Scaler', scaler),('LR'  , LogisticRegression())])))
    pipelines.append((nameOfScaler+'NB'  , Pipeline([('Scaler', scaler),('NB'  , GaussianNB())])))
    return pipelines

#Train Model


def fit_model(X_train, y_train,models):
    # Test options and evaluation metric
    num_folds = 10
    scoring = 'accuracy'
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
    return names, results



# Create a heatmap of the confusion matrix
fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(conf_matrix, cmap='YlGnBu')

# Add annotations to the heatmap
for i in range(len(classes)):
    for j in range(len(classes)):
        text = ax.text(j, i, conf_matrix[i, j],
                       ha="center", va="center", color="black")

# Customize the plot
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion Matrix', fontsize=20, y=1.1)
plt.ylabel('Actual label', fontsize=15)
plt.xlabel('Predicted label', fontsize=15)

# Add a colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Counts', rotation=-90, va="bottom")

# Display the plot using st.pyplot()
st.pyplot(fig)

# Display the code using st.code()
with st.echo():
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create a heatmap of the confusion matrix
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(conf_matrix, cmap='YlGnBu')

    # Add annotations to the heatmap
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, conf_matrix[i, j],
                           ha="center", va="center", color="black")

    # Customize the plot
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion Matrix', fontsize=20, y=1.1)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)

    # Add a colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Counts', rotation=-90, va="bottom")

    # Display the plot using st.pyplot()
    st.pyplot(fig)




'''
# Performance Measure
def classification_metrics(model, conf_matrix):
    print(f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
    print(f"Validation Accuracy Score: {model.score(X_test, y_test) * 100:.1f}%")
    fig,ax = plt.subplots(figsize=(8,6))
    sns.heatmap(pd.DataFrame(conf_matrix), annot = True, cmap = 'YlGnBu',fmt = 'g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion Matrix', fontsize=20, y=1.1)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.show()
    print(classification_report(y_test, y_pred))
    classes = np.unique(y_test)
    '''
# ROC_AUC


def roc_auc(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(8,6))
    print(f"roc_auc score: {auc(fpr, tpr)*100:.1f}%")
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate',fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=20)
    plt.legend()
    plt.show()


df = pd.read_csv('Crop_recommendation.csv')



# All columns contain outliers except for rice and label you can check outliers by using boxplot
numeric_cols = df.select_dtypes(include=[np.number]).columns

Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
df_out = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]




target ='label'
X_train, X_test, y_train, y_test = read_in_and_split_data(df, target)
models = GetModel()
names,results = fit_model(X_train, y_train,models)
ScaledModel = NormalizedModel('standard')
name,results = fit_model(X_train, y_train, ScaledModel)


pipeline = make_pipeline(StandardScaler(),  LogisticRegression())
model = pipeline.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred
conf_matrix = confusion_matrix(y_test,y_pred)
classification_metrics(pipeline, conf_matrix)



st.title('Recommended crop to Grow')

N = st.number_input("Enter Nitrogen value in %")
P = st.number_input("Enter Phosphorus value in %")
K = st.number_input("Enter Potassium value in %")
temperature = st.number_input("Enter Temperature in deg.c")
humidity = st.number_input("Enter Humidity value ")
ph = st.slider("Enter PH value (0-7)", 0, 7, 3)
rainfall = st.number_input("Enter RainFall value in mm")
sample = [N, P, K, temperature, humidity, ph, rainfall]
single_sample = np.array(sample).reshape(1,-1)
pred = model.predict(single_sample)
st.write("Recommended crop to grow is:",pred.item().title())


