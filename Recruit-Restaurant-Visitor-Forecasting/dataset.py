import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.preprocessing import (LabelEncoder, StandardScaler)

class Visitor_Dataset():
    def __init__(self, dataset_path, scaler_flag):
        air_visit = pd.read_csv(f"{dataset_path}air_visit_data.csv")
        air_store = pd.read_csv(f"{dataset_path}air_store_info.csv")
        hpg_store = pd.read_csv(f"{dataset_path}hpg_store_info.csv")
        air_reserve = pd.read_csv(f"{dataset_path}air_reserve.csv")
        hpg_reserve = pd.read_csv(f"{dataset_path}hpg_reserve.csv")
        holidays = pd.read_csv(f"{dataset_path}date_info.csv")
        store_id = pd.read_csv(f"{dataset_path}store_id_relation.csv")

        air_reserve["visit_datetime"] = pd.to_datetime(air_reserve["visit_datetime"])
        air_reserve["reserve_datetime"]=pd.to_datetime(air_reserve["reserve_datetime"])
        air_reserve["timeDiff"]=(air_reserve["visit_datetime"]-air_reserve["reserve_datetime"]).astype('timedelta64[h]')
        air_reserve["days"] = air_reserve["timeDiff"]/24

        hpg_reserve["visit_datetime"] = pd.to_datetime(hpg_reserve["visit_datetime"])
        hpg_reserve["reserve_datetime"]=pd.to_datetime(hpg_reserve["reserve_datetime"])
        hpg_reserve["timeDiff"]=(hpg_reserve["visit_datetime"]-hpg_reserve["reserve_datetime"]).astype('timedelta64[h]')
        hpg_reserve["days"] = hpg_reserve["timeDiff"]/24

        air_store_genreWise = pd.DataFrame(air_store.groupby("air_genre_name")["air_store_id"].count()).reset_index()
        air_store_genreWise.index+=1
        air_store_genreWise = air_store_genreWise.rename(columns={"air_store_id":"noOfRest"})
        air_store_genreWise = air_store_genreWise.sort_values(by="noOfRest",ascending=False)

        air_store_areaWise = pd.DataFrame(air_store.groupby("air_area_name")["air_store_id"].count()).reset_index()
        air_store_areaWise.index+=1
        air_store_areaWise = air_store_areaWise.rename(columns={"air_store_id":"noOfRest"})
        air_store_areaWise = air_store_areaWise.sort_values(by="noOfRest", ascending=False)[:16]

        hpg_store_genreWise = pd.DataFrame(hpg_store.groupby("hpg_genre_name")["hpg_store_id"].count()).reset_index()
        hpg_store_genreWise.index+=1
        hpg_store_genreWise = hpg_store_genreWise.rename(columns={"hpg_store_id":"noOfRest"})
        hpg_store_genreWise = hpg_store_genreWise.sort_values(by="noOfRest", ascending=False)[:15]

        hpg_store_areaWise = pd.DataFrame(hpg_store.groupby("hpg_area_name")["hpg_store_id"].count()).reset_index()
        hpg_store_areaWise.index+=1
        hpg_store_areaWise = hpg_store_areaWise.rename(columns={"hpg_store_id":"noOfRest"})
        hpg_store_areaWise=hpg_store_areaWise.sort_values(by="noOfRest",ascending=False)[:20]

        hpg_reserve = pd.merge(hpg_reserve, store_id, how='inner', on=['hpg_store_id'])
        air_reserve["visit_datetime"]= pd.to_datetime(air_reserve["visit_datetime"])
        air_reserve["visit_datetime"] = air_reserve["visit_datetime"].dt.date
        air_reserve["reserve_datetime"]= pd.to_datetime(air_reserve["reserve_datetime"])
        air_reserve["reserve_datetime"] = air_reserve["reserve_datetime"].dt.date
        air_reserve["reserve_datetime_diff"]= air_reserve.apply(lambda r: (r["visit_datetime"]-r["reserve_datetime"]).days,axis=1)
        t1 = air_reserve.groupby(["air_store_id","visit_datetime"],as_index=False)[["reserve_datetime_diff","reserve_visitors"]].sum().rename(columns={"visit_datetime":"visit_date","reserve_datetime_diff":"rs1","reserve_visitors":"rv1"})
        t2 = air_reserve.groupby(["air_store_id","visit_datetime"],as_index=False)[["reserve_datetime_diff","reserve_visitors"]].mean().rename(columns={"visit_datetime":"visit_date","reserve_datetime_diff":"rs2","reserve_visitors":"rv2"})
        air_reserve = pd.merge(t1,t2,how="inner",on=["air_store_id","visit_date"])

        hpg_reserve["visit_datetime"]= pd.to_datetime(hpg_reserve["visit_datetime"])
        hpg_reserve["visit_datetime"] = hpg_reserve["visit_datetime"].dt.date
        hpg_reserve["reserve_datetime"]= pd.to_datetime(hpg_reserve["reserve_datetime"])
        hpg_reserve["reserve_datetime"] = hpg_reserve["reserve_datetime"].dt.date
        hpg_reserve["reserve_datetime_diff"]= hpg_reserve.apply(lambda r: (r["visit_datetime"]-r["reserve_datetime"]).days,axis=1)
        t11 = hpg_reserve.groupby(["air_store_id","visit_datetime"],as_index=False)[["reserve_datetime_diff","reserve_visitors"]].sum().rename(columns={"visit_datetime":"visit_date","reserve_datetime_diff":"rs1","reserve_visitors":"rv1"})
        t22 = hpg_reserve.groupby(["air_store_id","visit_datetime"],as_index=False)[["reserve_datetime_diff","reserve_visitors"]].mean().rename(columns={"visit_datetime":"visit_date","reserve_datetime_diff":"rs2","reserve_visitors":"rv2"})
        hpg_reserve = pd.merge(t11,t22,how="inner",on=["air_store_id","visit_date"])

        air_visit_1 = pd.read_csv(f"{dataset_path}air_visit_data.csv")
        air_visit_1["visit_date"] = pd.to_datetime(air_visit_1["visit_date"])
        air_visit_1['DayofWeek'] = air_visit_1['visit_date'].dt.dayofweek
        air_visit_1['year'] = air_visit_1['visit_date'].dt.year
        air_visit_1['month'] = air_visit_1['visit_date'].dt.month
        air_visit_1['visit_date'] = air_visit_1['visit_date'].dt.date

        distinct_stores=air_visit_1["air_store_id"].unique()
        stores = pd.concat(
            [
                pd.DataFrame({"air_store_id": distinct_stores, "DayofWeek": [i] * len(distinct_stores)})
                for i in range(7)
            ],
            axis=0,
            ignore_index=True
        ).reset_index(drop=True)

        t = air_visit_1.groupby(["air_store_id","DayofWeek"],as_index=False)["visitors"].min().rename(columns={"visitors":"min_visitors"})
        stores = pd.merge(stores, t, how="left", on=["air_store_id", "DayofWeek"])

        t = air_visit_1.groupby(["air_store_id","DayofWeek"],as_index=False)["visitors"].mean().rename(columns={"visitors":"mean_visitors"})
        stores= pd.merge(stores,t,how="left",on=["air_store_id","DayofWeek"])

        t = air_visit_1.groupby(["air_store_id","DayofWeek"],as_index=False)["visitors"].median().rename(columns={"visitors":"median_visitors"})
        stores= pd.merge(stores,t,how="left",on=["air_store_id","DayofWeek"])

        t = air_visit_1.groupby(["air_store_id","DayofWeek"],as_index=False)["visitors"].max().rename(columns={"visitors":"max_visitors"})
        stores= pd.merge(stores,t,how="left",on=["air_store_id","DayofWeek"])

        t = air_visit_1.groupby(["air_store_id","DayofWeek"],as_index=False)["visitors"].count().rename(columns={"visitors":"count_visitors"})
        stores= pd.merge(stores,t,how="left",on=["air_store_id","DayofWeek"])

        k2 = air_visit_1.groupby(["air_store_id"]).agg({"visitors":[np.mean,np.std]}).reset_index()
        k2.columns=["air_store_id","mean_visitorso","std_devo"]
        stores = pd.merge(stores, k2, how="left", on=["air_store_id"])
        stores=pd.merge(stores,air_store,how="left",on=["air_store_id"])

        stores["air_genre_name"]=stores["air_genre_name"].map(lambda x: str(str(x).replace("/"," ")))
        stores["air_area_name"]= stores["air_area_name"].map(lambda x: str(str(x).replace("-"," ")))

        labelEncode = LabelEncoder()

        for i in range(10):
            stores["air_genre_name"+str(i)] = labelEncode.fit_transform(stores["air_genre_name"].map(lambda x: str(str(x).split(" ")[i]) if len(str(x).split(" ")) >i else ""))
            stores["air_area_name"+str(i)] = labelEncode.fit_transform(stores["air_area_name"].map(lambda x: str(str(x).split(" ")[i]) if len(str(x).split(" ")) >i else ""))

        stores["air_genre_name"] = labelEncode.fit_transform(stores["air_genre_name"])
        stores["air_area_name"] = labelEncode.fit_transform(stores["air_area_name"])
        stores['area']=stores['air_area_name'].map(lambda x: str(str(x).split(' ')[:2]))
        stores['area']=labelEncode.fit_transform(stores['area'])
        stores['area']=stores['air_area_name'].map(lambda x: str(str(x).split(' ')[:2]))
        stores['area'] = labelEncode.fit_transform(stores['area'])

        holidays = holidays.rename(columns={"calendar_date":"visit_date"})
        holidays["visit_date"] = pd.to_datetime(holidays["visit_date"])
        holidays["day_of_week"] = labelEncode.fit_transform(holidays["day_of_week"])
        holidays["visit_date"] = holidays["visit_date"].dt.date

        train = pd.merge(air_visit_1,holidays,how="left",on=["visit_date"])
        train = pd.merge(train,stores,how="left",on=["air_store_id","DayofWeek"])

        for df in [air_reserve,hpg_reserve]:
            train = pd.merge(train, df, how='left', on=['air_store_id','visit_date'])

        train["id"] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)
        train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
        train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
        train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

        train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

        train['var_max_lat'] = train['latitude'].max() - train['latitude']
        train['var_max_long'] = train['longitude'].max() - train['longitude']
        train['lon_plus_lat'] = train['longitude'] + train['latitude']

        train['air_store_id2'] = labelEncode.fit_transform(train['air_store_id'])

        self.col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors']]

        train = train.fillna(-1)

        test = train.groupby('year').get_group(2017.0).reset_index(drop=True)
        train = train.groupby('year').get_group(2016.0).reset_index(drop=True)

        self.train_target = train['visitors'].to_frame().astype(float)
        self.test_target = test['visitors'].to_frame().astype(float)

        if scaler_flag == True:
            scaler = StandardScaler()
            train_tmp = train[self.col]
            train_scaler = scaler.fit(train_tmp)
            train_tmp = train_scaler.transform(train_tmp)
            train[self.col] = pd.DataFrame(data=train_tmp, columns=self.col)
            test_tmp = test[self.col]
            test_scaler = scaler.fit(test_tmp)
            test_tmp = test_scaler.transform(test_tmp)
            test[self.col] = pd.DataFrame(data=test_tmp, columns=self.col)

        self.train = train[self.col].astype(float)
        self.test = test

    def __len__(self):
        return len(self.train)

    def __getitem__(self, index):
        # transform dataframe to numpy array
        x = self.train.iloc[index, :].values
        y = self.train_target.iloc[index, :].values
        return x, y

