import pandas as pd
import numpy as np
import copy
import math
import os
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from datetime import datetime
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV, Perceptron, \
    LogisticRegressionCV, PassiveAggressiveClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier

global path

scenario = os.listdir(os.path.join('./dataset/scenarios'))
for i, pastas in enumerate(scenario):
    if not os.listdir(os.path.join('./dataset/scenarios', pastas)):
        continue
    else:
        print('{} - {}'.format(i, pastas))
scene = int(input('\nWhich scenario you will choose?\n'))
os.system("clear")

routines = os.listdir(os.path.join('./dataset/scenarios', scenario[scene]))
for i, pastas in enumerate(routines):
    if not os.listdir(os.path.join('./dataset/scenarios', scenario[scene], pastas)):
        continue
    else:
        print('{} - {}'.format(i, pastas))
routine = int(input('\nWhich routine you will choose?\n'))
os.system("clear")

approaches = os.listdir(os.path.join('./dataset/scenarios', scenario[scene], routines[routine]))
for i, pastas in enumerate(approaches):
    if not os.listdir(os.path.join('./dataset/scenarios', scenario[scene], routines[routine], pastas)):
        continue
    else:
        print('{} - {}'.format(i, pastas))
approach = int(input('\nWhich approach you will choose?\n'))
os.system("clear")

print('Scenario: {}\nRoutine: {}\nApproach: {}\n\n'.format(scenario[scene], routines[routine], approaches[approach]))

path = os.path.join('./dataset/scenarios', scenario[scene], routines[routine], approaches[approach])
path_2_save = os.path.join('./output_training/', scenario[scene], routines[routine], approaches[approach])
categories = os.listdir(os.path.join('./dataset/scenarios', scenario[scene], routines[routine], approaches[approach]))


class MultiLabelMain:

    def __init__(self, ml_instance, ml_name, filename, n_jobs=4):

        self.n_jobs = n_jobs
        self.filename = filename
        self.ml_name = ml_name
        self.ml_instance = ml_instance
        self.multi_instance_prediction = None
        self.general_accuracy = None
        self.exclusive_accuracy = None
        self.total_time = []
        self.time_to_fit = []
        self.time_to_predict = []
        self.device = []
        self.instant = []
        self.general_accuracy = []
        self.exclusive_accuracy = []
        self.accuracy_in_change = []
        self.accuracy_in_maintenance = []
        self.state_changes = []

        self.dev_df = pd.DataFrame([], columns=["Algorithm",
                                                "Total_time",
                                                "Predict_time",
                                                "Fit_time",
                                                "General_Accuracy",
                                                "Device"])

        self.env_df = pd.DataFrame([], columns=["Algorithm",
                                                "Total_time",
                                                "Predict_time",
                                                "Fit_time",
                                                "General_Accuracy",
                                                "Instant"])

        self.room_df = pd.DataFrame([], columns=["Algorithm",
                                                 "Total_time",
                                                 "Predict_time",
                                                 "Fit_time",
                                                 "General_Accuracy"])

    def training_testing(self, train_x_df, train_y_df, test_x_df, test_y_df):
        print("Training model: {}".format(self.ml_name))

        multi_instance = MultiOutputClassifier(copy.deepcopy(self.ml_instance), n_jobs=self.n_jobs)
        self.ml_name = str(self.ml_instance.__class__).split('.')[-1].split('\'')[0]
        start_time_to_fit = datetime.now().strptime(datetime.now().strftime("%H:%M:%S:%f"), "%H:%M:%S:%f")
        multi_instance.fit(train_x_df, train_y_df)
        final_time_to_fit = datetime.now().strptime(datetime.now().strftime("%H:%M:%S:%f"), "%H:%M:%S:%f")
        self.time_to_fit.append(math.ceil((final_time_to_fit - start_time_to_fit).total_seconds()))

        start_time_to_predict = datetime.now().strptime(datetime.now().strftime("%H:%M:%S:%f"), "%H:%M:%S:%f")
        self.multi_instance_prediction = multi_instance.predict(test_x_df)
        final_time_to_predict = datetime.now().strptime(datetime.now().strftime("%H:%M:%S:%f"), "%H:%M:%S:%f")
        self.time_to_predict.append(math.ceil((final_time_to_predict - start_time_to_predict).total_seconds()))

        general_accuracy = ((test_y_df == self.multi_instance_prediction).sum(axis=1) / test_y_df.shape[1]).mean()
        exclusive_accuracy = ((test_y_df == self.multi_instance_prediction).sum(axis=0) / test_y_df.shape[0]).to_dict()

        columns = [column.split('_Y')[0] for column in test_y_df]

        change_status = ((test_x_df[columns].values != test_y_df.values).sum(axis=1) > 0)
        changes = change_status.sum(axis=0)
        self.state_changes.append(changes)

        accuracy_in_change = ((test_y_df[change_status] == self.multi_instance_prediction[change_status]).sum(axis=1) /
                              test_y_df[change_status].shape[1]).mean()
        change_status = ((test_x_df[columns].values == test_y_df.values).sum(axis=1) > 0)
        accuracy_in_maintenance = ((test_y_df[change_status] == self.multi_instance_prediction[change_status]).
                                   sum(axis=1) / test_y_df[change_status].shape[1]).mean()

        # Acurácia geral
        self.general_accuracy.append(general_accuracy)
        # Acurácia da mudança de estado: Quando for pra mudar o estado, e o modelo não muda!
        self.accuracy_in_change.append(0.0 if np.isnan(accuracy_in_change) else accuracy_in_change)
        # Acurácia da manutenção de estado: Quando for pra manter o estado, e o modelo mantém!
        self.accuracy_in_maintenance.append(0.0 if np.isnan(accuracy_in_maintenance) else accuracy_in_maintenance)
        self.exclusive_accuracy.append(exclusive_accuracy)

    def run_over_ml_models(self):
        global path, path_2_save
        k_fold_numbers = 5
        for category in categories:
            if category == "devices":
                files_path = os.path.join(path, category)
                # Para a mesma categoria, as labels e os devices, assim como os nomes dos conteúdos
                # dentro da pasta, são iguais. Por isso, elas contêm o índice [0] nas listas abaixo.
                labels = os.listdir(files_path)
                instants = os.listdir(os.path.join(files_path, labels[0]))
                list_devices = os.listdir(os.path.join(files_path, labels[0], instants[0]))

                for label in labels:
                    for instant in instants:
                        self.total_time = []
                        self.time_to_predict = []
                        self.instant = []
                        self.general_accuracy = []
                        self.state_changes = []

                        media_list = []
                        device_list = []
                        predict_time = []
                        fit_time = []
                        state_changes = []
                        accuracy_in_change_avg = []
                        accuracy_in_maintenance_avg = []
                        exclusive_accuracy_avg = []
                        for device in list_devices:
                            self.time_to_predict = []
                            self.time_to_fit = []
                            self.state_changes = []
                            self.exclusive_accuracy = []
                            self.accuracy_in_maintenance = []
                            self.accuracy_in_change = []

                            deviceFile = pd.read_csv(os.path.join(files_path, label, instant, device))
                            # pd.set_option('future.no_silent_downcasting', True)
                            deviceFile.replace({'OFF': 0,
                                                'ON': 1,
                                                'AUTO': 2,
                                                'COLL': 3,
                                                'STANDBY': 4,
                                                'WIND': 5}, inplace=True)
                            splited = KFold(n_splits=k_fold_numbers)
                            y_columns = [column for column in deviceFile.columns if '_Y' in column]
                            y = deviceFile[y_columns]
                            X = deviceFile.drop(y_columns, axis=1)

                            start_time = datetime.now().strptime(datetime.now().strftime("%H:%M:%S:%f"),
                                                                 "%H:%M:%S:%f")
                            time_list = []

                            for k, (train_index, test_index) in enumerate(splited.split(X, y), start=1):
                                X_train = X.iloc[train_index].copy()
                                X_test = X.iloc[test_index].copy()
                                y_train = y.iloc[train_index].copy()
                                y_test = y.iloc[test_index].copy()

                                self.training_testing(train_x_df=X_train,
                                                        train_y_df=y_train,
                                                        test_x_df=X_test,
                                                        test_y_df=y_test)

                                final_time = datetime.now().strptime(datetime.now().strftime("%H:%M:%S:%f"),
                                                                     "%H:%M:%S:%f")
                                time_list.append(math.ceil((final_time - start_time).total_seconds()))

                            self.total_time.append(np.mean(time_list))
                            predict_time.append(np.mean(self.time_to_predict))
                            fit_time.append(np.mean(self.time_to_fit))
                            state_changes.append(np.mean(self.state_changes))
                            media_list.append(np.mean(self.general_accuracy))
                            device_list.append(device)
                            accuracy_in_maintenance_avg.append(np.mean(self.accuracy_in_maintenance))
                            accuracy_in_change_avg.append(np.mean(self.accuracy_in_change))
                            exclusive_accuracy_avg.append(pd.DataFrame(self.exclusive_accuracy).mean().to_dict())

                        self.dev_df['Predict_time'] = predict_time
                        self.dev_df['Fit_time'] = fit_time
                        self.dev_df['State_changes'] = state_changes
                        self.dev_df['Total_time'] = self.total_time
                        self.dev_df['Device'] = device_list
                        self.dev_df['Algorithm'] = self.ml_name
                        self.dev_df['General_Accuracy'] = media_list
                        self.dev_df['Accuracy_in_Maintenance'] = accuracy_in_maintenance_avg
                        self.dev_df['Accuracy_in_Change'] = accuracy_in_change_avg
                        self.dev_df['Exclusive_Accuracy'] = exclusive_accuracy_avg

                        output_training = os.path.join(path_2_save, category, label, instant)

                        if not os.path.exists(output_training):
                            os.makedirs(output_training)

                        output_file_path = os.path.join(output_training, self.filename)
                        self.dev_df.to_csv(output_file_path, index=True)

            elif category == "environment":
                files_path = os.path.join(path, category)

                labels = os.listdir(files_path)
                files = os.listdir(os.path.join(files_path, labels[0]))
                for condition in labels:
                    self.total_time = []
                    self.time_to_predict = []
                    self.instant = []
                    self.general_accuracy = []
                    self.state_changes = []

                    media_list = []
                    predict_time = []
                    fit_time = []
                    state_changes = []
                    accuracy_in_change_avg = []
                    accuracy_in_maintenance_avg = []
                    exclusive_accuracy_avg = []

                    for instant in files:
                        self.time_to_predict = []
                        self.time_to_fit = []
                        self.state_changes = []
                        self.exclusive_accuracy = []
                        self.accuracy_in_maintenance = []
                        self.accuracy_in_change = []

                        start_time = datetime.now().strptime(datetime.now().strftime("%H:%M:%S:%f"),
                                                             "%H:%M:%S:%f")

                        dataFile = pd.read_csv(os.path.join(files_path, condition, instant))
                        # pd.set_option('future.no_silent_downcasting', True)
                        dataFile.replace({'OFF': 0, 'ON': 1, 'AUTO': 2, 'COLL': 3, 'STANDBY': 4, 'WIND': 5},
                                         inplace=True)
                        dataFile.fillna(0, inplace=True)

                        splited = KFold(n_splits=k_fold_numbers)
                        y_columns = [column for column in dataFile.columns if '_Y' in column]

                        y = dataFile[y_columns]
                        X = dataFile.drop(y_columns, axis=1)
                        time_list = []
                        for k, (train_index, test_index) in enumerate(splited.split(X, y)):
                            X_train = X.iloc[train_index].copy()
                            X_test = X.iloc[test_index].copy()
                            y_train = y.iloc[train_index].copy()
                            y_test = y.iloc[test_index].copy()

                            self.training_testing(train_x_df=X_train,
                                                    train_y_df=y_train,
                                                    test_x_df=X_test,
                                                    test_y_df=y_test)

                            final_time = datetime.now().strptime(datetime.now().strftime("%H:%M:%S:%f"),
                                                                 "%H:%M:%S:%f")
                            time_list.append(math.ceil((final_time - start_time).total_seconds()))

                        self.total_time.append(np.mean(time_list))
                        self.instant.append(instant)
                        predict_time.append(np.mean(self.time_to_predict))
                        fit_time.append(np.mean(self.time_to_fit))
                        state_changes.append(np.mean(self.state_changes))
                        media_list.append(np.mean(self.general_accuracy))
                        accuracy_in_maintenance_avg.append(np.mean(self.accuracy_in_maintenance))
                        accuracy_in_change_avg.append(np.mean(self.accuracy_in_change))
                        exclusive_accuracy_avg.append(pd.DataFrame(self.exclusive_accuracy).mean().to_dict())

                    self.env_df['Predict_time'] = predict_time
                    self.env_df['Fit_time'] = fit_time
                    self.env_df['State_changes'] = state_changes
                    self.env_df['Total_time'] = self.total_time
                    self.env_df['Instant'] = self.instant
                    self.env_df['Algorithm'] = self.ml_name
                    self.env_df['General_Accuracy'] = media_list
                    self.env_df['Accuracy_in_Maintenance'] = accuracy_in_maintenance_avg
                    self.env_df['Accuracy_in_Change'] = accuracy_in_change_avg
                    self.env_df['Exclusive_Accuracy'] = exclusive_accuracy_avg

                    output_training = os.path.join(path_2_save, category, condition)

                    if not os.path.exists(output_training):
                        os.makedirs(output_training)

                    output_file_path = os.path.join(output_training, self.filename)
                    self.env_df.to_csv(output_file_path, index=True)


all_ml = [
    # BernoulliNB(),
    DecisionTreeClassifier(),
    # ExtraTreesClassifier(),
    # ExtraTreeClassifier(),
    # GaussianNB(),
    # GradientBoostingClassifier(),
    # GaussianProcessClassifier(multi_class="one_vs_one"),
    # KNeighborsClassifier(),
    # LabelPropagation(),
    # LabelSpreading(),
    # LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
    # LinearSVC(tol=1e-5, max_iter=30000),
    # LogisticRegression(max_iter=10000),
    # MLPClassifier(max_iter=5000),
    # NearestCentroid(),
    # NuSVC(gamma='scale'),
    # PassiveAggressiveClassifier(),
    # QuadraticDiscriminantAnalysis(),
    # RidgeClassifier(max_iter=50000),
    # RidgeClassifierCV(),
    # SGDClassifier(),
    # SVC(),
]

ml_names = [
    # "BernoulliNB",
    "DecisionTreeClassifier",
    # "ExtraTreesClassifier",
    # "ExtraTreeClassifier",
    # "GaussianNB",
    # "GradientBoostingClassifier",
    # "GaussianProcessClassifier",
    # "KNeighborsClassifier",
    # "LabelPropagation",
    # "LabelSpreading",
    # "LinearDiscriminantAnalysis",
    # "LinearSVC",
    # "LogisticRegression",
    # "MLPClassifier",
    # "NearestCentroid",
    # "NuSVC"
    # "PassiveAggressiveClassifier",
    # "QuadraticDiscriminantAnalysis",
    # "RidgeClassifier",
    # "RidgeClassifierCV",
    # "SGDClassifier",
    # "SVC",
]

if __name__ == "__main__":

    for instance, name in zip(all_ml, ml_names):
        filename = name + '.csv'
        multi_instance = MultiLabelMain(
            ml_instance=instance, ml_name=name, filename=filename
        )
        multi_instance.run_over_ml_models()
