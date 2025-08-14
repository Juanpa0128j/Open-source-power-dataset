# Created by xunannancy at 2021/9/25
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

external_feature_names = ['DHI', 'DNI', 'GHI', 'Dew Point', 'Solar Zenith Angle', 'Wind Speed', 'Relative Humidity', 'Temperature']
target_features = ['load_power', 'wind_power', 'solar_power']
holidays = USFederalHolidayCalendar().holidays()
task_prediction_horizon = OrderedDict({
    'load': [60, 1440],
    'wind': [5, 30],
    'solar': [5, 30],
})
step_size = {
    'wind': 1, # predict at every minute
    'solar': 1, # predict at every minute
    'load': 60 # predict at every hour
}

class HistoryConcatTrainDataset(Dataset):
    def __init__(self, x, y, flag):
        """
        for training & validation dataset
        :param x: historical y and external features
        :param y: future y
        :param flag: whether future y is to predict or not; for loss computation
        """
        self.x = x
        self.y = y
        self.flag = flag

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.flag[idx]

class HistoryConcatTestDataset(Dataset):
    def __init__(self, ID, x):
        """
        for testing
        :param ID: testing ID
        :param x: historical features and y
        """
        self.ID = ID
        self.x = x
    def __len__(self):
        return len(self.ID)
    def __getitem__(self, idx):
        return self.ID[idx], self.x[idx]

class ForecastingIterableDataset(IterableDataset):
    def __init__(self, csv_path, mode, x_cols, y_cols=None, idx_list=None):
        self.csv_path = csv_path
        self.mode = mode  # 'train' or 'test'
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.idx_list = idx_list

    def __iter__(self):
        import pandas as pd
        chunk_iter = pd.read_csv(self.csv_path, chunksize=1024)
        for chunk in chunk_iter:
            # If idx_list is provided, filter rows
            if self.idx_list is not None:
                chunk = chunk[chunk.index.isin(self.idx_list)]
            x = chunk[self.x_cols].to_numpy()
            if self.mode == 'train' and self.y_cols is not None:
                y = chunk[self.y_cols].to_numpy()
                for i in range(len(x)):
                    yield x[i], y[i]
            else:
                for i in range(len(x)):
                    yield x[i]

class ForecastingDataset:
    def __init__(self, root):
        self.raw_data_folder = os.path.join(root, 'Minute-level Load and Renewable')
        self.data_folder = os.path.join(root, 'processed_dataset', 'forecasting')
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        for loc_index, file in enumerate(sorted(os.listdir(self.raw_data_folder))):
            print(f'{loc_index}/{len(os.listdir(self.raw_data_folder))}')
            if not file.endswith('_.csv'):
                continue
            if not os.path.exists(os.path.join(self.data_folder, file.split('.')[0]+'2018.csv')):
                self.processing(file)
            break

    def processing(self, file):
        data = pd.read_csv(os.path.join(self.raw_data_folder, file))
        time = data['time'].to_numpy()
        time_h = [i.split()[0] for i in time]
        date_pd, date_h = pd.DatetimeIndex(time), pd.DatetimeIndex(time_h)
        month_day = (date_pd.month.astype(float) + date_pd.day.astype(float) / 31).to_numpy().reshape([-1, 1])
        weekday = date_pd.weekday.to_numpy().reshape([-1, 1])
        holiday = (date_h.isin(holidays)).astype(int).reshape([-1, 1])
        date_info = np.concatenate([month_day, weekday, holiday], axis=-1)

        year_2018_index = np.sort(np.argwhere((date_pd >= '2018-01-01 00:00:00') & (date_pd < '2019-01-01 00:00:00')).reshape([-1]))
        year_2019_index = np.sort(np.argwhere((date_pd >= '2019-01-01 00:00:00') & (date_pd < '2020-01-01 00:00:00')).reshape([-1]))
        year_2020_index = np.sort(np.argwhere((date_pd >= '2020-01-01 00:00:00') & (date_pd < '2021-01-01 00:00:00')).reshape([-1]))
        for year in tqdm([2018, 2019, 2020]):
            cur_year_index = locals()[f'year_{year}_index']
            cur_data = data.iloc[cur_year_index]
            cur_date_info = date_info[cur_year_index]
            cur_date_pd = pd.DatetimeIndex(cur_data['time'].to_numpy())
            cur_train_test_split_idx = np.argwhere(cur_date_pd == f'{year}-12-01 00:00:00').reshape([-1])[0]

            cur_processed_data_dict = {
                'task': file.split('.')[0]+str(year),
                'feature_name': ['month_day', 'weekday', 'holiday'],
                'training_data': [cur_date_info[:cur_train_test_split_idx]],
                'testing_data': [cur_date_info[cur_train_test_split_idx:]],
            }
            # external features
            cur_processed_data_dict['feature_name'] += external_feature_names
            cur_processed_data_dict['training_data'].append(cur_data[external_feature_names].to_numpy()[:cur_train_test_split_idx])
            cur_processed_data_dict['testing_data'].append(cur_data[external_feature_names].to_numpy()[cur_train_test_split_idx:])

            #cur target features
            for one_target in target_features:
                cur_processed_data_dict['feature_name'] += [f'y{one_target[0]}_t']
                cur_target_data_training = cur_data[[one_target]].to_numpy()[:cur_train_test_split_idx]
                cur_target_data_testing = cur_data[[one_target]].to_numpy()[cur_train_test_split_idx:]
                cur_processed_data_dict['training_data'].append(cur_target_data_training)
                cur_processed_data_dict['testing_data'].append(cur_target_data_testing)

                # every minute or every hour prediction
                predict_flag_training = (np.arange(cur_train_test_split_idx) % step_size[one_target.split('_')[0]] == 0)
                predict_flag_testing = (np.arange(len(cur_data) - cur_train_test_split_idx) % step_size[one_target.split('_')[0]] == 0)
                for forecast_horizon_index, forecast_horizon_val in enumerate(task_prediction_horizon[one_target.split('_')[0]]):
                    cur_processed_data_dict['feature_name'] += [f'y{one_target[0]}_t+{forecast_horizon_val}(val)', f'y{one_target[0]}_t+{forecast_horizon_val}(flag)']
                    # training
                    cur_target_training = np.concatenate([cur_target_data_training[forecast_horizon_val:], np.repeat([[-1]], forecast_horizon_val, axis=0)], axis=0)
                    horizon_predict_flag = (np.arange(len(cur_target_data_training)) < len(cur_target_data_training) - forecast_horizon_val)
                    time_stamp_flag = (cur_target_data_training.squeeze() > 1e-8) & (cur_target_training.squeeze() > 1e-8)
                    cur_target_predict_flag = predict_flag_training & horizon_predict_flag & time_stamp_flag
                    cur_processed_data_dict['training_data'].append(cur_target_training)
                    cur_processed_data_dict['training_data'].append(np.expand_dims(cur_target_predict_flag, axis=-1))
                    # testing
                    cur_target_testing = np.concatenate([cur_target_data_testing[forecast_horizon_val:], np.repeat([[-1]], forecast_horizon_val, axis=0)], axis=0)
                    horizon_predict_flag = (np.arange(len(cur_target_data_testing)) < len(cur_target_data_testing) - forecast_horizon_val)
                    time_stamp_flag = (cur_target_data_testing.squeeze() > 1e-8) & (cur_target_testing.squeeze() > 1e-8)
                    cur_target_predict_flag = predict_flag_testing & horizon_predict_flag & time_stamp_flag
                    cur_processed_data_dict['testing_data'].append(cur_target_testing)
                    cur_processed_data_dict['testing_data'].append(np.expand_dims(cur_target_predict_flag, axis=-1))
            cur_processed_data_dict['training_data'] = np.concatenate(cur_processed_data_dict['training_data'], axis=-1)
            cur_processed_data_dict['testing_data'] = np.concatenate(cur_processed_data_dict['testing_data'], axis=-1)
            training_frame = pd.DataFrame(cur_processed_data_dict['training_data'], columns=cur_processed_data_dict['feature_name'])
            testing_frame = pd.DataFrame(cur_processed_data_dict['testing_data'], columns=cur_processed_data_dict['feature_name'])
            total_frame = pd.concat([training_frame, testing_frame], ignore_index=True)
            total_frame['train_flag'] = np.concatenate([np.ones(training_frame.shape[0]), np.zeros(testing_frame.shape[0])], axis=0)
            total_frame['ID'] = range(total_frame.shape[0])
            total_frame.to_csv(os.path.join(self.data_folder, file.split('.')[0]+str(year)+'.csv'), index=False, header=True, columns=['ID']+list(total_frame)[:-1])
        return

    def load(self, sliding_window, loc, year, batch_size, shuffle):
        import pandas as pd
        csv_path = os.path.join(self.data_folder, f'{loc}_{year}.csv')
        df = pd.read_csv(csv_path)
        train_flag = df['train_flag'].to_numpy()
        train_idx = df.index[train_flag == 1].tolist()
        test_idx = df.index[train_flag == 0].tolist()

        # Define columns for X and Y
        x_cols = [col for col in df.columns if col not in ['ID', 'train_flag'] and not col.endswith('(val)') and not col.endswith('(flag)')]
        y_cols = [col for col in df.columns if col.endswith('(val)')]

        train_dataset = ForecastingIterableDataset(csv_path, 'train', x_cols, y_cols, train_idx)
        test_dataset = ForecastingIterableDataset(csv_path, 'test', x_cols, None, test_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        return train_loader, test_loader
