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
import torch
import torch.nn as nn
import torch.optim as optim

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


def detect_operational_regimes(data, is_nominal=False):
    if is_nominal:
        return {
            'start': data[data['time'] < 0.1],
            'steady': data[data['time'] >= 0.1],
            'fault': pd.DataFrame()
        }
    else:
        return {
            'start': data[data['time'] < 0.1],
            'steady': data[data['time'] >= 0.1],
            'fault': data[data['time'] >= 0.1]
        }


def load_and_prepare_data():
    files = [
        'motor_data_NOM.csv',
        'motor_data_A-GROUND.csv', 'motor_data_B-GROUND.csv', 'motor_data_C-GROUND.csv',
        'motor_data_A-B.csv', 'motor_data_B-C.csv', 'motor_data_A-C.csv',
        'motor_data_phase_fault_A.csv', 'motor_data_phase_fault_B.csv', 'motor_data_phase_fault_C.csv'
    ]

    regimes = {'start': [], 'steady': [], 'fault': []}

    for file in files:
        try:
            df = pd.read_csv(file)
            is_nominal = 'NOM' in file
            regime_data = detect_operational_regimes(df, is_nominal)
            for regime in ['start', 'steady', 'fault']:
                if not regime_data[regime].empty:
                    regime_df = regime_data[regime].copy()
                    regime_df.loc[:, 'source_file'] = file.replace('motor_data_', '').replace('.csv', '')
                    regime_df.loc[:, 'Category'] = 'NOM' if is_nominal else file.replace('motor_data_', '').replace(
                        '.csv', '')
                    regimes[regime].append(regime_df)
        except FileNotFoundError:
            print(f"Ошибка: файл {file} не найден!")
            continue

    combined_data = {
        'start': pd.concat(regimes['start'], ignore_index=True) if regimes['start'] else pd.DataFrame(),
        'steady': pd.concat(regimes['steady'], ignore_index=True) if regimes['steady'] else pd.DataFrame(),
        'fault': pd.concat(regimes['fault'], ignore_index=True) if regimes['fault'] else pd.DataFrame()
    }

    return combined_data


def preprocess_data(data, regime_name):
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

    X_df = pd.DataFrame(X_processed, columns=numeric_cols)
    X_df['regime'] = regime_name

    return X_df, y_encoded, encoder, preprocessor


def plot_extended_visualization(X, y, encoder, regime_name):
    phases = ['Ia', 'Ib', 'Ic']
    for phase in phases:
        plt.figure(figsize=(12, 6))
        for label in np.unique(y):
            mask = y == label
            plt.plot(X.loc[mask, phase][:100],
                     label=f"{encoder.inverse_transform([label])[0]}",
                     linewidth=2)
        plt.title(f'Ток фазы {phase.upper()} (Режим: {regime_name})', pad=15)
        plt.xlabel('Временные точки', labelpad=10)
        plt.ylabel('Нормализованное значение', labelpad=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(12, 6))
    scatter = sns.scatterplot(
        x=X['Torque'],
        y=X['Speed'],
        hue=[encoder.inverse_transform([label])[0] for label in y],
        palette="husl",
        s=100,
        alpha=0.8,
        edgecolor='w',
        linewidth=0.5
    )
    plt.title(f'Зависимость скорости от момента (Режим: {regime_name})', pad=15)
    plt.xlabel('Момент (норм.)', labelpad=10)
    plt.ylabel('Скорость (норм.)', labelpad=10)
    plt.legend(title='Тип режима', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_multiclass_roc(clf, X_test, y_test, encoder, regime_name):
    if not hasattr(clf, "predict_proba"):
        print(f"Модель {clf.__class__.__name__} не поддерживает predict_proba, ROC-кривые недоступны")
        return

    y_score = clf.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(12, 8))
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

    for i, color in zip(range(n_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=3, alpha=0.8,
                 label='{0} (AUC = {1:0.2f})'.format(encoder.classes_[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ложноположительная частота', labelpad=10)
    plt.ylabel('Истинноположительная частота', labelpad=10)
    plt.title(f'ROC-кривые (Режим: {regime_name})', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def train_and_evaluate_models(X_train, X_test, y_train, y_test, encoder, regime_name):
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, C=10, random_state=42, class_weight='balanced')
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'=' * 50}\nАнализ модели: {name} (Режим: {regime_name})\n{'=' * 50}")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("\nОтчет о классификации:")
            print(classification_report(
                y_test, y_pred,
                target_names=encoder.classes_,
                zero_division=0
            ))

            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_,
                annot_kws={"size": 10}
            )
            plt.title(f'Матрица ошибок ({name}, Режим: {regime_name})', pad=20)
            plt.xlabel('Предсказанный класс', labelpad=10)
            plt.ylabel('Истинный класс', labelpad=10)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

            if len(encoder.classes_) > 1:
                plot_multiclass_roc(model, X_test, y_test, encoder, regime_name)

            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'report': classification_report(
                    y_test, y_pred,
                    target_names=encoder.classes_,
                    output_dict=True,
                    zero_division=0
                )
            }

        except Exception as e:
            print(f"Ошибка в модели {name}: {str(e)}")
            continue

    return results


def compare_models(results, encoder, regime_name):
    if not results:
        return

    accuracies = {name: res['accuracy'] for name, res in results.items()}

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=list(accuracies.keys()),
        y=list(accuracies.values()),
        hue=list(accuracies.keys()),
        palette="viridis",
        alpha=0.8,
        legend=False
    )

    plt.title(f'Сравнение точности моделей (Режим: {regime_name})', pad=20)
    plt.ylabel('Точность', labelpad=10)
    plt.xlabel('Модели', labelpad=10)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', alpha=0.3)

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center', xytext=(0, 10),
            textcoords='offset points', fontsize=10
        )

    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.show()

    metrics = ['precision', 'recall', 'f1-score']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for model_name, res in results.items():
            values = []
            for cls in encoder.classes_:
                if cls in res['report']:
                    values.append(res['report'][cls][metric])
                else:
                    values.append(0)

            plt.plot(encoder.classes_, values, 'o-', linewidth=2, markersize=6,
                     label=model_name, alpha=0.8)

        plt.title(f'Сравнение {metric} по классам (Режим: {regime_name})', pad=20)
        plt.xlabel('Класс неисправности', labelpad=10)
        plt.ylabel(metric.capitalize(), labelpad=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_feature_importance(results, regime_name):
    if not results:
        return

    rf_model = results['Random Forest']['model']
    importances = rf_model.feature_importances_
    features = ['Tn', 'k', 'time', 'Ia', 'Ib', 'Ic', 'Vbc', 'Torque', 'Speed', 'pinn_error']
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        range(len(indices)),
        importances[indices],
        align='center',
        color='#1f77b4',
        alpha=0.8,
        edgecolor='w'
    )

    plt.yticks(range(len(indices)), np.array(features)[indices])
    plt.title(f'Важность признаков (Режим: {regime_name})', pad=20)
    plt.xlabel('Относительная важность', labelpad=10)
    plt.grid(axis='x', alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{width:.3f}',
            va='center',
            ha='left',
            fontsize=9
        )

    ax = plt.gca()
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.show()


def analyze_errors(results, X_test, y_test, encoder, regime_name):
    if not results:
        return

    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    y_pred = best_model.predict(X_test)

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Reds',
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_,
        annot_kws={"size": 10}
    )
    plt.title(f'Матрица ошибок лучшей модели ({best_model_name}, Режим: {regime_name})', pad=20)
    plt.xlabel('Предсказанный класс', labelpad=12)
    plt.ylabel('Истинный класс', labelpad=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    error_mask = y_pred != y_test
    error_points = X_test[error_mask]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=error_points['Torque'],
        y=error_points['Speed'],
        hue=[encoder.inverse_transform([label])[0] for label in y_test[error_mask]],
        palette="dark",
        s=100,
        alpha=0.9,
        edgecolor='k'
    )
    plt.title(f'Точки ошибок классификации (Режим: {regime_name})', pad=20)
    plt.xlabel('Момент', labelpad=12)
    plt.ylabel('Скорость', labelpad=12)
    plt.legend(title='Класс ошибки', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


class EnhancedPINN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=100, output_dim=1):
        super(EnhancedPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def pinn_train(model, times, speeds, epochs=2000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)

    # Параметры асинхронного двигателя
    J = 0.0131  # Момент инерции [кг·м²]
    T_nom = 26.71  # Номинальный момент [Н·м]
    B = 0.001  # Коэффициент вязкого трения
    omega_sync = 314.16  # Синхронная скорость (для 50 Гц) [рад/с]
    s_nom = 0.03  # Номинальное скольжение


    times.requires_grad_(True)

    for epoch in range(epochs):
        optimizer.zero_grad()

        omega_pred = model(times)

        d_omega = torch.autograd.grad(
            outputs=omega_pred,
            inputs=times,
            grad_outputs=torch.ones_like(omega_pred),
            create_graph=True,
            retain_graph=True
        )[0]

        # Моделирование электромагнитного момента
        s = (omega_sync - omega_pred) / omega_sync  # Скольжение
        Te = 2 * T_nom / (s / s_nom + s_nom / s)  # Характеристика момента

        # Момент нагрузки
        Tl = T_nom

        # Физическое ограничение скорости
        speed_constraint = torch.relu(omega_pred - omega_sync)

        # Физическое уравнение
        physics_loss = torch.mean((J * d_omega - (Te - Tl - B * omega_pred)) ** 2) + \
                                  torch.mean(speed_constraint ** 2)

        # Ошибка данных
        data_loss = torch.mean((omega_pred - speeds) ** 2)

        # Общая функция потерь
        loss = 0.7 * data_loss + 0.3 * physics_loss

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 500 == 0:
            print(f"Эпоха {epoch}/{epochs} - Ошибка: {loss.item():.6f} - LR: {optimizer.param_groups[0]['lr']:.2e}")

    return model


def get_pinn_predictions(data, train_on_nominal=True):
    if train_on_nominal:
        norm_data = data[data['Category'] == 'NOM']
        if len(norm_data) == 0:
            print("Нет данных нормального режима для обучения PINN")
            return np.zeros(len(data))

        times = torch.tensor(norm_data['time'].values, dtype=torch.float32).unsqueeze(1)
        speeds = torch.tensor(norm_data['Speed'].values, dtype=torch.float32).unsqueeze(1)
    else:
        times = torch.tensor(data['time'].values, dtype=torch.float32).unsqueeze(1)
        speeds = torch.tensor(data['Speed'].values, dtype=torch.float32).unsqueeze(1)


    times.requires_grad_(True)

    model = EnhancedPINN()
    model = pinn_train(model, times, speeds)

    with torch.no_grad():
        all_times = torch.tensor(data['time'].values, dtype=torch.float32).unsqueeze(1)
        preds = model(all_times).numpy().flatten()

    return preds


def plot_pinn_vs_real(data, preds, regime_name):
    categories = data['Category'].unique()

    for category in categories:
        cat_data = data[data['Category'] == category]
        cat_preds = preds[data['Category'] == category]

        plt.figure(figsize=(12, 6))
        plt.plot(cat_data['time'], cat_data['Speed'], 'b-', linewidth=2, label='Реальная скорость')
        plt.plot(cat_data['time'], cat_preds, 'r--', linewidth=2, label='PINN предсказание')
        plt.title(f'Сравнение реальной скорости и PINN ({category}, Режим: {regime_name})', pad=15)
        plt.xlabel('Время, с', labelpad=10)
        plt.ylabel('Скорость, рад/с', labelpad=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(cat_data['time'], np.abs(cat_data['Speed'] - cat_preds), 'g-', linewidth=2)
        plt.title(f'Ошибка PINN по времени ({category}, Режим: {regime_name})', pad=15)
        plt.xlabel('Время, с', labelpad=10)
        plt.ylabel('Абсолютная ошибка, рад/с', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def add_pinn_error_feature(data, train_on_nominal=True):
    preds = get_pinn_predictions(data, train_on_nominal)
    data_with_feature = data.copy()
    data_with_feature['pinn_error'] = np.abs(data_with_feature['Speed'] - preds)
    return data_with_feature, preds


def main():
    combined_data = load_and_prepare_data()
    regimes = ['start', 'steady', 'fault']

    for regime in regimes:
        data = combined_data[regime]
        if data.empty:
            print(f"Данные для режима {regime} отсутствуют, пропускаем.")
            continue

        print(f"\n{'*' * 50}\nОбработка режима: {regime}\n{'*' * 50}")

        data_with_pinn, pinn_preds = add_pinn_error_feature(data, train_on_nominal=True)

        numeric_cols = ['Tn', 'k', 'time', 'Ia', 'Ib', 'Ic', 'Vbc', 'Torque', 'Speed', 'pinn_error']
        X = data_with_pinn[numeric_cols]
        y = data_with_pinn['Category']

        preprocessor = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
        X_processed = preprocessor.fit_transform(X)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        X_df = pd.DataFrame(X_processed, columns=numeric_cols)

        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )

        plot_pinn_vs_real(data_with_pinn, pinn_preds, regime)
        plot_extended_visualization(X_df, y_encoded, encoder, regime)

        results = train_and_evaluate_models(X_train, X_test, y_train, y_test, encoder, regime)
        compare_models(results, encoder, regime)
        plot_feature_importance(results, regime)
        analyze_errors(results, X_test, y_test, encoder, regime)


if __name__ == "__main__":
    main()




