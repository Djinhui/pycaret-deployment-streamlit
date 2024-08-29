# https://mp.weixin.qq.com/s/6kG2paGUYE3SR9oHoaQ4sQ

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve
import joblib


df = pd.read_csv('Data.csv')
X = df.drop(['target'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

params_xgb = {
    'learning_rate':0.02,
    'booster':'gbtree',
    'objective':'binary:logistic',
    'max_leaves':127,
    'verbosity':1,
    'seed':42,
    'nthread':-1,
    'colsample_bytree':0.6,
    'subsample':0.7,
    'eval_metric':'logloss'
}

model_xgb = xgb.XGBClassifier(**params_xgb)

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],  # 树的数量
    'max_depth': [3, 4, 5, 6, 7],               # 树的深度
    'learning_rate': [0.01, 0.02, 0.05, 0.1],   # 学习率
}

grid_search = GridSearchCV(
    estimator=model_xgb,
    param_grid=param_grid,
    scoring='neg_log_loss',
    cv=5,
    n_jobs=-1, 
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

y_score = best_model.predict_proba(X_test)[:, 1]

fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_score)
roc_auc_logistic = auc(fpr_logistic, tpr_logistic)

plt.figure()
plt.plot(fpr_logistic, tpr_logistic, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_logistic)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


joblib.dump(best_model, 'xgboost.pkl')

# -------------------------------------------------------------
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


model = joblib.load('xgboost.pkl')

cp_options = {
    1: 'Typical angina (1)',
    2: 'Atypical angina (2)',
    3: 'Non-anginal pain (3)',
    4: 'Asymptomatic (4)'
}

restecg_options = {
    0: 'Normal (0)',
    1: 'ST-T wave abnormality (1)',
    2: 'Left ventricular hypertrophy (2)'
}

slope_options = {
    1: 'Upsloping (1)',
    2: 'Flat (2)',
    3: 'Downsloping (3)'
}

thal_options = {
    1: 'Normal (1)',
    2: 'Fixed defect (2)',
    3: 'Reversible defect (3)'
}

feature_names = [
    "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Serum Cholesterol",
    "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate", "Exercise Induced Angina",
    "ST Depression", "Slope", "Number of Vessels", "Thal"
]

st.title('Heart Disease Prediction')

age = st.number_input('Age', min_value=1, max_value=120, value=50)
sex = st.selectbox('Sex (0=Female, 1=Male):', options=[0,1], format_func=lambda x:'Female (0)' if x==0 else 'Male (1)')
cp = st.selectbox('Chest pain type:', options=list(cp_options.keys()), format_func=lambda x:cp_options[x])
trestbps = st.number_input('Resting blood pressure (trestbps):', min_value=50, max_value=200, value=120)
chol = st.number_input('Serum cholesterol in mg/dl (chol):', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting blood sugar > 120 mg/dl (fbs):', options=[0,1], format_func=lambda x:'False (0)' if x==0 else 'True (1)')
restecg = st.selectbox('Restng electrocardiographic results:', options=list(restecg_options.keys()), format_func=lambda x:restecg_options[x])
thalach = st.number_input('Maximum heart rate achieved (thalach):', min_value=50, max_value=250, value=150)
exang = st.selectbox('Exercise induced angina (exang):', options=[0,1], format_func=lambda x:'No (0)'if x==0 else 'Yes (1)')
oldpeak = st.number_input('ST depression induced by exercise relative to rest (pldpeak):', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of the peak exercise ST segment (slope):', options=list(slope_options.keys()), format_func=lambda x:slope_options[x])
ca = st.number_input('Number of major vessels colored by fluoroscopy (ca):', min_value=0, max_value=4, value=0)
thal = st.selectbox('Thal (thal):', options=list(thal_options.keys()), format_func=lambda x:thal_options[x])

feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
features = np.array([feature_values])

if st.button('Predict'):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_prob(features)[0]

    st.write(f'**Predicted Class:**{predicted_class}')
    st.write(f'**Prediction Probabilities:**{predicted_proba}')

    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )
    st.write(advice)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig('shape_force_plot.png', bbox_inches='tight', dpi=1200)
    st.image('shape_force_plot.png')


# -------------------------------------------------------------------------------------------------------

'''
APP部署

1. 准备项目文件

确保你的项目文件夹中包含以下文件:
Python脚本文件 (your_script.py)：这是包含 Streamlit 代码的文件
requirements.txt：列出所有依赖库的文件
model

2. 上传项目到Github

登录到 GitHub
点击右上角的“New”按钮创建一个新仓库
选择仓库的名字并初始化 README 文件
将项目文件推送到这个仓库中，以下是推送代码的步骤（假设你已经在本地初始化了 Git）
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/your-repo-name.git
git push -u origin main

3. 部署到 Streamlit Cloud
a. 登录 Streamlit Cloud
访问 Streamlit Cloud
使用 GitHub 账户登录（因为你需要访问 GitHub 仓库）

b. 部署应用
登录后，点击页面右上角的 "Create app" 按钮
在弹出的对话框中，选择你刚刚上传到 GitHub 的项目仓库
选择分支（一般为 main）和要运行的 Python 脚本文件（如 your_script.py）
点击 "Deploy" 按钮
Streamlit 会生成一个唯一的 URL，通过这个 URL，你和其他用户可以访问你的应用

这里是一种部署APP的方法读者可以自行选择部署方法
'''