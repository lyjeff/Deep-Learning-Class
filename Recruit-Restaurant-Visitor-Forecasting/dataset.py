import pandas as pd
import numpy as np
import glob
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split

class Visitor_Dataset(Dataset):
    def __init__(self, dataset_path):
        files = glob.glob(f"{dataset_path}/*.csv")
        df = {}

        for filename in files:
            df[os.path.basename(filename).replace('.csv', '')] = pd.read_csv(filename)

        df['air_visit_data']['visit_datetime'] = pd.to_datetime(df['air_visit_data']['visit_date'])
        df['air_visit_data']['visit_date'] = df['air_visit_data']['visit_datetime'].dt.date
        df['sample_submission']['visit_datetime'] = df['sample_submission']['id'].map(lambda x: str(x).split('_')[2])
        df['sample_submission']['air_store_id'] = df['sample_submission']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
        df['sample_submission']['visit_datetime'] = pd.to_datetime(df['sample_submission']['visit_datetime'])
        df['sample_submission']['visit_date'] = df['sample_submission']['visit_datetime'].dt.date
        df['sample_submission'].drop(['id'], axis=1, inplace=True)
        total_dataset = pd.concat([df['air_visit_data'], df['sample_submission']], axis = 0)


        df['date_info']['calendar_date'] = pd.to_datetime(df['date_info']['calendar_date'])
        df['date_info']['visit_date'] = df['date_info']['calendar_date'].dt.date
        df['date_info'] = df['date_info'].drop(['calendar_date'], axis = 1)
        total_dataset = pd.merge(total_dataset, df['air_store_info'], how='left', on='air_store_id')
        total_dataset = pd.merge(total_dataset, df['date_info'], how='left', on='visit_date')
        total_dataset['visit_datetime'] = pd.to_datetime(total_dataset['visit_date'])
        total_dataset['year']  = total_dataset['visit_datetime'].dt.year
        total_dataset['month'] = total_dataset['visit_datetime'].dt.month
        total_dataset['day']   = total_dataset['visit_datetime'].dt.day
        total_dataset.drop('visit_datetime', axis=1, inplace=True)

        cat_features = [col for col in ['air_genre_name', 'air_area_name', 'day_of_week', 'year']]
        for column in cat_features:
            temp = pd.get_dummies(pd.Series(total_dataset[column]))
            total_dataset = pd.concat([total_dataset,temp],axis=1)
            total_dataset = total_dataset.drop([column],axis=1)
        total_dataset.drop(['latitude', 'longitude'], axis = 1, inplace = True)
        temp = pd.get_dummies(total_dataset['air_store_id'])
        total_dataset = pd.concat([total_dataset,temp],axis=1)
        sep = len(df['air_visit_data'])
        train = total_dataset[:sep]
        to_predict = total_dataset[sep:]
        col = [c for c in train if c not in ['air_store_id', 'visit_date', 'visitors']]
        X_train, y_train = train[col], train['visitors']
        X_to_predict = to_predict[col]
        value_X = X_train.values
        value_y = y_train.values
        value_X_to_predict = X_to_predict.values
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaled_X = scaler_X.fit_transform(value_X)
        scaled_X_to_predict = scaler_X.transform(value_X_to_predict)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, value_y, test_size=0.3, random_state=42)

    def __len__(self):
        #  no need to rewrite
        return len(self.data)

    def __getitem__(self, index):
        # transform dataframe to numpy array, no need to rewrite
        x = self.data.iloc[index, :].values
        y = self.label.iloc[index, :].values
        return x, y

