import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, auc)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 1. 数据加载与初步探索
print("步骤1: 数据加载...")
df = pd.read_csv('adult_data.csv', header=None, na_values=' ?')

# 添加列名
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 
           'marital_status', 'occupation', 'relationship', 'race', 'sex', 
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
df.columns = columns

print(f"数据集形状: {df.shape}")
print("\n前5行数据:")
print(df.head())
print("\n数据信息:")
print(df.info())

# 2. 数据预处理
print("\n步骤2: 数据预处理...")

# 处理缺失值
print("处理缺失值...")
df = df.dropna()
print(f"处理后数据集形状: {df.shape}")

# 处理异常值
print("处理异常值...")
df = df[(df['age'] >= 18) & (df['age'] <= 90)]
df = df[(df['hours_per_week'] >= 10) & (df['hours_per_week'] <= 80)]
print(f"处理后数据集形状: {df.shape}")

# 目标变量编码
print("目标变量编码...")
df['income'] = df['income'].map({' <=50K': 0, ' >50K': 1})

# 3. 探索性数据分析 (EDA)
print("\n步骤3: 探索性数据分析 (EDA)...")

# 关键特征分布
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(df['age'], kde=True)
plt.title('年龄分布')

plt.subplot(2, 2, 2)
sns.countplot(x='education', data=df, order=df['education'].value_counts().index)
plt.xticks(rotation=45)
plt.title('教育水平分布')

plt.subplot(2, 2, 3)
sns.boxplot(x='income', y='hours_per_week', data=df)
plt.title('每周工时与收入关系')

plt.subplot(2, 2, 4)
sns.countplot(x='occupation', hue='income', data=df)
plt.xticks(rotation=45)
plt.title('职业与收入关系')

plt.tight_layout()
plt.savefig('eda_results.png', dpi=300)
plt.show()

# 4. 特征工程
print("\n步骤4: 特征工程...")

# 分离特征和目标变量
X = df.drop('income', axis=1)
y = df['income']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 定义预处理管道
numeric_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
categorical_features = ['workclass', 'education', 'marital_status', 'occupation', 
                        'relationship', 'race', 'sex', 'native_country']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 修复点：将 sparse=False 改为 sparse_output=False
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. 模型训练与评估
print("\n步骤5: 模型训练与评估...")
models = {
    "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
    "决策树": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

results = {}
for name, model in tqdm(models.items(), desc="模型训练进度"):
    print(f"\n训练 {name} 模型...")
    start_time = time.time()
    
    # 创建完整管道
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selector', VarianceThreshold(threshold=0.01)),
        ('classifier', model)
    ])
    
    # 训练模型
    pipeline.fit(X_train, y_train)
    
    # 交叉验证
    print(f"进行 {name} 的5折交叉验证...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    
    # 预测
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # 计算指标
    train_time = time.time() - start_time
    results[name] = {
        'model': pipeline,
        'cv_mean_accuracy': np.mean(cv_scores),
        'cv_std_accuracy': np.std(cv_scores),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        'train_time': train_time
    }
    
    # 输出分类报告
    print(f"\n{name} 分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(f'confusion_matrix_{name}.png', dpi=300)
    plt.show()

# 6. 模型结果比较
print("\n步骤6: 模型结果比较...")
results_df = pd.DataFrame(results).T
print("\n模型性能比较:")
print(results_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'train_time']])

# ROC曲线比较
plt.figure(figsize=(10, 8))
for name, res in results.items():
    if res['roc_auc'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res['model'].predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('ROC曲线比较')
plt.legend(loc="lower right")
plt.savefig('roc_curves.png', dpi=300)
plt.show()

# 7. 模型优化
print("\n步骤7: 模型优化...")

# 逻辑回归优化
print("\n优化逻辑回归模型...")
param_grid_lr = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear']
}

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selector', VarianceThreshold(threshold=0.01)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

grid_lr = GridSearchCV(lr_pipeline, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_lr.fit(X_train, y_train)

print(f"最佳参数: {grid_lr.best_params_}")
print(f"最佳交叉验证准确率: {grid_lr.best_score_:.4f}")

# 决策树优化
print("\n优化决策树模型...")
param_grid_dt = {
    'classifier__max_depth': [5, 10, 15, 20, 25, 30, None],
    'classifier__min_samples_split': [2, 5, 10, 20],
    'classifier__min_samples_leaf': [1, 2, 5, 10]
}

dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selector', VarianceThreshold(threshold=0.01)),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

grid_dt = GridSearchCV(dt_pipeline, param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
grid_dt.fit(X_train, y_train)

print(f"最佳参数: {grid_dt.best_params_}")
print(f"最佳交叉验证准确率: {grid_dt.best_score_:.4f}")

# KNN优化
print("\n优化KNN模型...")
param_grid_knn = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__p': [1, 2]
}

knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selector', VarianceThreshold(threshold=0.01)),
    ('classifier', KNeighborsClassifier())
])

grid_knn = GridSearchCV(knn_pipeline, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train, y_train)

print(f"最佳参数: {grid_knn.best_params_}")
print(f"最佳交叉验证准确率: {grid_knn.best_score_:.4f}")

# 8. 优化后模型评估
print("\n步骤8: 优化后模型评估...")
optimized_models = {
    "优化逻辑回归": grid_lr.best_estimator_,
    "优化决策树": grid_dt.best_estimator_,
    "优化KNN": grid_knn.best_estimator_
}

optimized_results = {}
for name, model in tqdm(optimized_models.items(), desc="优化模型评估进度"):
    print(f"\n评估 {name} 模型...")
    start_time = time.time()
    
    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算指标
    train_time = time.time() - start_time
    optimized_results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'train_time': train_time
    }
    
    # 输出分类报告
    print(f"\n{name} 分类报告:")
    print(classification_report(y_test, y_pred))

# 最终结果比较
final_results_df = pd.DataFrame(optimized_results).T
print("\n优化后模型性能比较:")
print(final_results_df)

# 保存结果
results_df.to_csv('model_results.csv')
final_results_df.to_csv('optimized_model_results.csv')
print("\n所有任务完成! 结果已保存到CSV文件。")