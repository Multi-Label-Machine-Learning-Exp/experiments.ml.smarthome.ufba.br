import os
import pandas as pd
import numpy as np


scenario = os.listdir(os.path.join('dataset/scenarios'))
for i, pastas in enumerate(scenario):
    if not os.listdir(os.path.join('dataset/scenarios', pastas)):
        continue
    else:
        print('{} - {}'.format(i, pastas))
scene = int(input('\nWhich scenario you will choose?\n'))
os.system("clear")

routines = os.listdir(os.path.join('dataset/scenarios', scenario[scene]))
for i, pastas in enumerate(routines):
    if not os.listdir(os.path.join('dataset/scenarios', scenario[scene], pastas)):
        continue
    else:
        print('{} - {}'.format(i, pastas))
routine = int(input('\nWhich routine you will choose?\n'))
os.system("clear")

approaches = os.listdir(os.path.join('dataset/scenarios', scenario[scene], routines[routine]))
for i, pastas in enumerate(approaches):
    if not os.listdir(os.path.join('dataset/scenarios', scenario[scene], routines[routine], pastas)):
        continue
    else:
        print('{} - {}'.format(i, pastas))
approach = int(input('\nWhich approach you will choose?\n'))
os.system("clear")

categories = os.listdir(os.path.join('dataset/scenarios', scenario[scene], routines[routine], approaches[approach]))
for i, pastas in enumerate(categories):
    if not os.listdir(os.path.join('dataset/scenarios', scenario[scene], routines[routine], approaches[approach],
                                   pastas)):
        continue
    else:
        print('{} - {}'.format(i, pastas))
cat = int(input('\nWhich category you will choose?\n'))
os.system("clear")

print('{} - {} - {} - {}'.format(scenario[scene], routines[routine], approaches[approach], categories[cat]))

path = os.path.join('./output_training/', scenario[scene], routines[routine], approaches[approach])
path_2_save = os.path.join('./charts/data/', scenario[scene], routines[routine], approaches[approach], categories[cat])


def find_devices(item, dataframe, device_dataframe, instant):
    devices = dataframe[dataframe['Device'].str.contains(item, case=False, na=False)].copy()
    devices['Instant'] = instant
    if not device_dataframe.empty:
        device_dataframe = pd.concat([device_dataframe, devices], ignore_index=True)
    else:
        device_dataframe = devices.copy()
    return device_dataframe


def generate_file_by_device():
    save = os.path.join(path_2_save, condition)
    instants = os.listdir(os.path.join(path, categories[cat], condition))

    models_list = os.listdir(os.path.join(path, categories[cat], condition, instants[0]))
    devices = pd.read_csv(os.path.join(path, categories[cat], condition, instants[0], models_list[0]))
    df = pd.DataFrame(columns=devices.columns)
    devices = devices['Device']

    if len(devices) == 1:
        pass
    else:
        devices = np.squeeze(np.asarray(devices)).tolist()

    for device in devices:
        device = device.split('.')[0]
        list_files = []
        for model in models_list:
            for instant in instants:
                df2 = pd.read_csv(os.path.join(path, categories[cat], condition, instant, model))
                instant = instant.split('/')[0]
                df3 = find_devices(device, df2, df, instant)
                list_files.append(df3)

        if not os.path.exists(path_2_save):
            os.makedirs(save)

        pd.concat(list_files, ignore_index=True).to_csv('{}/{}.csv'.format(save, device), index=True)


def generate_file_by_environment():
    models_list = os.listdir(os.path.join(path, categories[cat], condition))

    list_files = []
    for model in models_list:
        trained_data = pd.read_csv(os.path.join(path, categories[cat], condition, model),
                                   usecols=['Algorithm', 'General_Accuracy', 'Accuracy_in_Maintenance',
                                            'Accuracy_in_Change', 'Instant'],
                                   index_col=False,
                                   header=0)
        list_files.append(trained_data)
        if not os.path.exists(path_2_save):
            os.makedirs(path_2_save)

        filename = condition.split('/')[0]
        pd.concat(list_files, ignore_index=True).to_csv('{}/{}.csv'.format(path_2_save, filename), index=True)


def generate_file_by_room():
    save = path_2_save + condition

    rooms_list = os.listdir(os.path.join(path, categories[cat], condition))
    instants = os.listdir(os.path.join(path, categories[cat], condition, rooms_list[0]))
    models_list = os.listdir(os.path.join(path, categories[cat], condition, rooms_list[0], instants[0]))

    for room in rooms_list:
        list_files = []
        for algorithm in models_list:
            for instant in instants:
                trained_data = pd.read_csv(os.path.join(path, categories[cat], condition, room, instant, algorithm),
                                           index_col=None, header=0, names=['Algorithm', 'Total_time', 'Predict_time',
                                                                            'Fit_time', 'General_Accuracy'])

                trained_data = trained_data[['Algorithm', 'General_Accuracy']]
                trained_data['Approach'] = instant
                list_files.append(trained_data)

                if not os.path.exists(save):
                    os.makedirs(save)

                pd.concat(list_files, ignore_index=True).to_csv('{}/{}.csv'.format(save, room), index=True)


def get_conditions():
    global conditions, i, condition, label

    conditions = os.listdir(os.path.join(path, categories[cat]))
    for i, condition in enumerate(conditions):
        print('\n{} - {}\n'.format(i, condition))
    label = int(input('Which label you will choose?\n'))
    condition = conditions[label]


if __name__ == "__main__":

    # Devices
    if categories[cat] == 'devices':
        get_conditions()
        generate_file_by_device()

    # Environment
    elif categories[cat] == 'environment':
        get_conditions()
        generate_file_by_environment()

    # Rooms
    elif categories[cat] == 'rooms':
        get_conditions()

        generate_file_by_room()

    print("Done!")
