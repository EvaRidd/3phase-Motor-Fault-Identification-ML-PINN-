import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from itertools import cycle

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18


def load_and_prepare_data():
    files = [
        'motor_data_NOM.csv',
        'motor_data_A-GROUND.csv', 'motor_data_B-GROUND.csv', 'motor_data_C-GROUND.csv',
        'motor_data_A-B.csv', 'motor_data_B-C.csv', 'motor_data_A-C.csv',
        'motor_data_phase_fault_A.csv', 'motor_data_phase_fault_B.csv', 'motor_data_phase_fault_C.csv'
    ]

    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except FileNotFoundError:
            print(f"Ошибка: файл {file} не найден!")
            return None

    if not dfs:
        print("Ошибка!")
        return None

    return pd.concat(dfs, ignore_index=True)


full_data = load_and_prepare_data()
if full_data is None:
    exit()


def preprocess_data(data):
    """Обработка и нормализация данных"""
    numeric_cols = ['Tn', 'k', 'time', 'Ia', 'Ib', 'Ic', 'Vbc', 'Torque', 'Speed']
    X = data[numeric_cols]
    y = data['Category']


    preprocessor = make_pipeline(
        SimpleImputer(strategy="median"),
        MinMaxScaler()
    )

    X_processed = preprocessor.fit_transform(X)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return pd.DataFrame(X_processed, columns=numeric_cols), y_encoded, encoder


X, y, encoder = preprocess_data(full_data)


def plot_extended_visualization(X, y, encoder):

    plt.close('all')


    phases = ['Ia', 'Ib', 'Ic']
    for phase in phases:
        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        for label in np.unique(y):
            mask = y == label
            plt.plot(X.loc[mask, phase][:100],
                     linewidth=2.5,
                     alpha=0.8,
                     label=encoder.inverse_transform([label])[0])
        plt.title(f'Ток фазы {phase.upper()}', pad=15)
        plt.xlabel('Временные точки', labelpad=10)
        plt.ylabel('Нормализованное значение', labelpad=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        ax.spines[['right', 'top']].set_visible(False)
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    scatter = sns.scatterplot(x=X['Torque'], y=X['Speed'], hue=y, palette="viridis",
                              s=100, alpha=0.7, edgecolor='w', linewidth=0.5)
    plt.title('Зависимость скорости от момента', pad=15)
    plt.xlabel('Момент (норм.)', labelpad=10)
    plt.ylabel('Скорость (норм.)', labelpad=10)
    plt.legend(title='Режим', labels=encoder.classes_, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.show()


plot_extended_visualization(X, y, encoder)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


def plot_multiclass_roc(clf, X_test, y_test, encoder, figsize=(14, 10)):

    if not hasattr(clf, "predict_proba"):
        print(f"Модель {clf.__class__.__name__} не поддерживает predict_proba, ROC-кривые недоступны")
        return

    y_score = clf.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=figsize)
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=3, alpha=0.8,
                 label='ROC класса {0} (AUC = {1:0.2f})'
                       ''.format(encoder.classes_[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', labelpad=10)
    plt.ylabel('True Positive Rate', labelpad=10)
    plt.title('Многоклассовые ROC-кривые', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.show()



def train_and_evaluate_models(X_train, X_test, y_train, y_test, encoder):

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, C=10, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'=' * 50}\nАнализ модели: {name}\n{'=' * 50}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("\nОтчет о классификации:")
        print(classification_report(y_test, y_pred, target_names=encoder.classes_))

        # Матрица ошибок
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=encoder.classes_,
                    yticklabels=encoder.classes_,
                    annot_kws={"size": 12})
        plt.title(f'Матрица ошибок ({name})', pad=15)
        plt.xlabel('Предсказанный класс', labelpad=10)
        plt.ylabel('Истинный класс', labelpad=10)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # ROC-кривые
        if hasattr(model, "predict_proba"):
            plot_multiclass_roc(model, X_test, y_test, encoder)
        else:
            print(f"ROC-кривые недоступны для {name}")

        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred,
                                            target_names=encoder.classes_,
                                            output_dict=True)
        }

    return results


results = train_and_evaluate_models(X_train, X_test, y_train, y_test, encoder)



def compare_models(results, encoder):

    accuracies = {name: res['accuracy'] for name, res in results.items()}

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=list(accuracies.keys()),
                     y=list(accuracies.values()),
                     hue=list(accuracies.keys()),
                     palette="viridis",
                     alpha=0.8,
                     legend=False)

    plt.title('Сравнение точности моделей', pad=20)
    plt.ylabel('Accuracy', labelpad=10)
    plt.xlabel('Модели', labelpad=10)
    plt.ylim(0.7, 1.0)
    plt.grid(axis='y', alpha=0.3)

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10),
                    textcoords='offset points', fontsize=12)

    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.show()

    metrics = ['precision', 'recall', 'f1-score']
    for metric in metrics:
        plt.figure(figsize=(16, 8))
        for model_name, res in results.items():
            values = [res['report'][cls][metric] for cls in encoder.classes_]
            plt.plot(encoder.classes_, values, 'o-', linewidth=3, markersize=8,
                     label=model_name, alpha=0.8)

        plt.title(f'Сравнение {metric} по классам', pad=20)
        plt.xlabel('Класс неисправности', labelpad=10)
        plt.ylabel(metric.capitalize(), labelpad=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        ax = plt.gca()
        ax.spines[['right', 'top']].set_visible(False)
        plt.tight_layout()
        plt.show()


compare_models(results, encoder)


def plot_feature_importance(results):
    """Визуализация важности признаков"""
    rf_model = results['Random Forest']['model']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(14, 8))
    bars = plt.barh(range(len(indices)), importances[indices], align='center',
                    color='#1f77b4', alpha=0.8, edgecolor='w')
    plt.yticks(range(len(indices)), np.array(X.columns)[indices])
    plt.title('Важность признаков (Random Forest)', pad=20)
    plt.xlabel('Относительная важность', labelpad=10)
    plt.grid(axis='x', alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{width:.3f}',
                 va='center', ha='left')

    ax = plt.gca()
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.show()


plot_feature_importance(results)


def analyze_errors(results, X_test, y_test, encoder):

    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    y_pred = best_model.predict(X_test)

    plt.figure(figsize=(14, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_,
                annot_kws={"size": 12})
    plt.title(f'Матрица ошибок ({best_model_name})', pad=20)
    plt.xlabel('Предсказанный класс', labelpad=10)
    plt.ylabel('Истинный класс', labelpad=10)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    error_df = pd.DataFrame({
        'True': encoder.inverse_transform(y_test),
        'Predicted': encoder.inverse_transform(y_pred)
    })

    print("\nТипичные ошибки классификации:")
    print(error_df[error_df['True'] != error_df['Predicted']].value_counts().head(10))


analyze_errors(results, X_test, y_test, encoder)

print("\n :)")

