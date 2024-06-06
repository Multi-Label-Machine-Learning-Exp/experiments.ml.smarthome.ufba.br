import os
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

savefile = int(input('Do you want to save images?'
                     '\nno - 0'
                     '\nyes - 1\n'))
if savefile == 0:
    save = False
else:
    save = True
os.system("clear")

scenario = os.listdir(os.path.join('dataset/scenarios'))
for i, pastas in enumerate(scenario):
    if not os.listdir(os.path.join('dataset/scenarios', pastas)):
        continue
    else:
        print('{} - {}'.format(i, pastas))
scene = int(input('\nWhich scenario you will choose?\n'))
os.system('clear')

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

path = os.path.join('./charts/data/', scenario[scene], routines[routine], approaches[approach])
path_2_save = os.path.join('./charts/data/', scenario[scene], routines[routine], approaches[approach],
                           categories[cat])


def save_files(file_name):
    images_path = os.path.join('./charts/images/', scenario[scene],  routines[routine], approaches[approach],
                               categories[cat])
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    # plt.savefig(images_path + (file_name.split('.')[0] + '.png'))
    plt.savefig(os.path.join(images_path, (file_name + '.png')))


def accuracy_charts_devices():
    device_list = os.listdir(os.path.join(path, categories[cat], condition))

    for device in device_list:
        chart = pd.read_csv(os.path.join(path, categories[cat], condition, device),
                            usecols=['Algorithm', 'General_Accuracy', 'Instant']).sort_values(by='General_Accuracy')
        matplotlib.pyplot.close()
        plt.figure(figsize=(20, 10))
        device_name = device.split('.')[0]
        (sns.barplot(data=chart, x='Algorithm', y='General_Accuracy', hue='Instant').
         set_title('Dispositivo {} treinado com uma rotina de {}'.format(device_name, routines[routine]),
                   fontsize=20))
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(0, 1, 1, 0))
        plt.tight_layout()

        if savefile:
            save_files(device_name)
        else:
            plt.show()


def accuracy_charts_environment():
    data_chart = []

    env_data = pd.read_csv(os.path.join(path, categories[cat], condition.split('/')[0]),
                           usecols=['Algorithm', 'General_Accuracy', 'Instant'],
                           index_col=False,
                           header=0)

    data_chart.append(env_data)
    data_plot = pd.concat(data_chart, ignore_index=True)
    matplotlib.pyplot.close()
    plt.figure(figsize=(20, 10))
    sns.barplot(data=data_plot, x='Algorithm', y='General_Accuracy', hue='Instant').set_title(
        'Assertividade geral dos algoritmos treinado com uma rotina de {} para um imóvel.'.
        format(routines[routine]), fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), fontsize=15)
    plt.tight_layout()

    if save:
        save_files('environment')
    else:
        plt.show()


def accuracy_charts_rooms():
    algorithms_list = os.listdir(os.path.join(path, categories[cat], condition))

    for algorithm in algorithms_list:
        data_chart = pd.read_csv(os.path.join(path, categories[cat], condition, algorithm),
                                 usecols=['Algorithm', 'General_Accuracy', 'Approach']). \
            sort_values(by='General_Accuracy')
        matplotlib.pyplot.close()
        plt.figure(figsize=(20, 10))
        sns.barplot(data=data_chart, x='Algorithm', y='General_Accuracy', hue='Approach', label='Algorithm').set_title(
            'Precisão do ambiente {} treinado com uma rotina de {}'.format(algorithm.split('.')[0], routines[routine]),
            fontsize=20)
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(0, 1, 1, 0))
        plt.tight_layout()

        if save:
            save_files(algorithm)
        else:
            plt.show()


if __name__ == "__main__":

    # Devices
    if categories[cat].lower() == 'devices':
        conditions = os.listdir(os.path.join(path, categories[cat]))

        for i, condition in enumerate(conditions):
            print('\n{} - {}\n'.format(i, condition))

        label = int(input('Which label you will choose?\n'))
        condition = conditions[label]
        accuracy_charts_devices()

    # Environment
    elif categories[cat].lower() == 'environment':
        conditions = os.listdir(os.path.join(path, categories[cat]))

        for i, condition in enumerate(conditions):
            print('\n{} - {}\n'.format(i, condition))

        label = int(input('Which label you will choose?\n'))
        condition = conditions[label]
        accuracy_charts_environment()

    # Rooms
    elif categories[cat].lower() == 'rooms':
        conditions = os.listdir(os.path.join(path, categories[cat]))

        for i, condition in enumerate(conditions):
            print('\n{} - {}\n'.format(i, condition))

        label = int(input('Which label you will choose?\n'))
        condition = conditions[label]
        accuracy_charts_rooms()

    print('Done!')
