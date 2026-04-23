import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 12

file_descriptions = {
    'motor_data_NOM.csv': 'Нормальный режим работы',
    'motor_data_A-GROUND.csv': 'Замыкание фазы A на землю',
    'motor_data_B-GROUND.csv': 'Замыкание фазы B на землю',
    'motor_data_C-GROUND.csv': 'Замыкание фазы C на землю',
    'motor_data_A-B.csv': 'Межфазное замыкание A-B',
    'motor_data_B-C.csv': 'Межфазное замыкание B-C',
    'motor_data_A-C.csv': 'Межфазное замыкание A-C',
    'motor_data_phase_fault_A.csv': 'Обрыв фазы A',
    'motor_data_phase_fault_B.csv': 'Обрыв фазы B',
    'motor_data_phase_fault_C.csv': 'Обрыв фазы C'
}

phase_colors = {
    'Ia': 'red',
    'Ib': 'green',
    'Ic': 'blue'
}

for filename, description in file_descriptions.items():
    try:

        data = pd.read_csv(filename)


        if not all(col in data.columns for col in ['time', 'Ia', 'Ib', 'Ic']):
            print(f"В файле {filename} отсутствуют необходимые столбцы")
            continue


        plt.figure(figsize=(14, 7))


        for phase, color in phase_colors.items():
            plt.plot(data['time'], data[phase],
                     label=f'Ток {phase} ({phase[1]})',
                     color=color,
                     linewidth=1.5)


        plt.title(f'Графики токов фаз\n{description}', pad=20)
        plt.xlabel('Время, с')
        plt.ylabel('Ток, А')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.xlim(0, 1)

        plt.tight_layout()

        plt.show()

    except FileNotFoundError:
        print(f'Файл {filename} не найден')
    except Exception as e:
        print("ОШИБКА")


