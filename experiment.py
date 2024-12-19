import numpy as np
import pandas as pd
from typing import Union
import warnings
from dotted_dict import DottedDict
from db import log
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass

class TestInputData(pd.DataFrame):
    _metadata = [
        'groups',
        'store',
        'metrics',
        'slices',
        'test_flg',
        'train_flg',
        'time_label',
        'metrics_renamed',
        'slices_renamed',
        'ls',
        'metrics_mapping',
        'slices_mapping',
    ]
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.groups = [
            'time_label',
            'store',
            'slices',
            'metrics',
            'test_flg',
            'train_flg',
        ]

        # Определение колонок
        # self.name = None
        # self.whs_id = None
        self.store = None
        self.metrics = None
        self.slices = None
        self.test_flg = None
        self.train_flg = None
        self.time_label = None

        self.metrics_renamed = False
        self.slices_renamed = False
        self.metrics_mapping = None
        self.slices_mapping = None

    @property
    def _constructor(self):
        return TestInputData
    
    def lower_columns(self):
        self.columns = self.columns.str.lower()
        return self

    def define(self, **mapping):
        for group, columns in mapping.items():
            if not isinstance(columns, list):
                columns = [columns]
            self.__setattr__(group, self[columns])
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                setattr(getattr(self, group), 'ls', getattr(self,group).columns.tolist())

        return self

    @property
    def info(self):
        info_df =  pd.DataFrame(
            dict(
                group = self.groups,
                is_defined = [True if isinstance(self.__getattr__(group), pd.DataFrame) else False for group in self.groups],
                cols = [getattr(self,group).columns.tolist() if isinstance(getattr(self, group), pd.DataFrame) else [] for group in self.groups],
            )
        )
        return info_df

    def drop_undefined(self):
        undefined = set(self.columns) ^ set(self.info.cols.sum())
        self = self.drop(undefined, axis=1)
        return self
    
    def rename_metrics(self):
        if isinstance(self.metrics, pd.DataFrame):
            if not self.metrics_renamed:
                mapper = {col: f'metric{i}' for i, col in enumerate(self.metrics, start=1)}
                reversed_mapper = {v:k for k,v in mapper.items()}
                self.rename(mapper=mapper, axis=1, inplace=True)
                self.metrics = self[mapper.values()]
                setattr(self, 'metrics_mapping', reversed_mapper)
                setattr(getattr(self, 'metrics'), 'ls', getattr(self, 'metrics').columns.tolist())
                self.metrics_renamed = True
        else:
            print('Metric cols are not defined')
        return self
    
    def rename_slice_categories(self):
        if isinstance(self.slices, pd.DataFrame):
            if not self.slices_renamed:
                slices_mapper = dict()
                for col in pd.DataFrame(self.slices):
                    cats = self[col].unique()
                    mapper = {cat:f'{col}{i}'for i, cat in enumerate(cats, start=1)}
                    self.replace({col:mapper}, inplace=True)
                    slices_mapper[col] = mapper
                reversed_mapper = {col: {v:k for k,v in mapper_.items()} for col, mapper_ in slices_mapper.items()}
                self.slices_mapping = reversed_mapper
                self.slices_renamed = True
        else:
            print('Section cols are not defined')
        return self

    def rename_flags(self):
        for flg in ('test_flg', 'train_flg'):
            flg_data = getattr(self, flg)
            if isinstance(flg_data, pd.DataFrame):
                flg_name, = flg_data.columns.tolist()
                self.rename(mapper={flg_name: flg}, errors='ignore', axis=1, inplace=True)
                setattr(self, flg, pd.DataFrame(self[flg]))
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    setattr(getattr(self, flg), 'ls', getattr(self,flg).columns.tolist())
        return self
    
    def rename_time_label(self):
        
        current_time_label, = self.time_label.ls
        self.rename(mapper={current_time_label:'time_label'}, axis=1, inplace=True)
        setattr(self, 'time_label', self['time_label'])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            setattr(getattr(self, 'time_label'), 'ls', ['time_label'])

        return self
    
    def __finalize__(self, other, method=None, **kwargs):
        if isinstance(other, TestInputData):
            for attr in self._metadata:
                setattr(self, attr, getattr(other, attr, None))
            return self
        

@dataclass
class Store:
    name: str
    whs_id: int
    time_labels: list
    metrics: list
    is_test: bool


class Test:

    def __init__(self, input:TestInputData):
        self.input = input
        self.data = self._prepare_data(input)
        self.mapping = self.data.slices_mapping|dict(metric=self.data.metrics_mapping)
        self.reversed_mapping = {col:{v:k for k, v in mapper.items()} for col, mapper in self.mapping.items()}

    def _prepare_data(self, input:TestInputData):
        data = input.copy()
        data = (
            data
            .drop_undefined()
            .rename_slice_categories()
            .rename_metrics()
            .rename_flags()
            .rename_time_label()
        )


        
        self.metrics_mapping = data.metrics_mapping
        self.slices_mapping = data.slices_mapping
        

        # Плохо можно сразу передавать имена
        self.test_flg = data.test_flg.ls
        self.metrics = data.metrics.ls
        self.slices = data.slices.ls if isinstance(data.slices, pd.DataFrame) else []
        self.time_label = data.time_label.ls
        self.train_flg = data.train_flg.ls
        self.store = data.store.ls

        self.samples = DottedDict()
        self.samples_list = DottedDict()

        return data

    @staticmethod
    def calculate_rmse(fact, pred, normed=True, how='minmax'):
        
        if how not in ('mean', 'minmax'):
            raise ValueError('`how` parameter takes only two values: "mean", "minmax"')

        rmse = np.sqrt(mean_squared_error(fact, pred))
        if normed:
            if how == 'mean':
                nrmse = rmse / np.mean(fact) 
            if how == 'minmax':
                nrmse = rmse / (np.max(fact) - np.min(fact))
            return nrmse
        else:
            return rmse

    def find_analogues(self, n_analogues:int=5, verbose=True, debug=False):

        data = self.data.copy()
        
        test_flg = data.test_flg.ls
        metrics = data.metrics.ls
        slices = data.slices.ls if isinstance(data.slices, pd.DataFrame) else []
        time_label = data.time_label.ls
        train_flg = data.train_flg.ls
        store = data.store.ls

        # Выносим из колонок метрики и создаем два новых столбца - название метрики и значение. Массив должен увеличиться в два раза
        if verbose or verbose: log('Транспонирование массива по метрике...')
        
        id_vars = (
            time_label
            + store
            + slices
            + test_flg
            + train_flg
        )
        
        data_melted = pd.melt(
            data,
            id_vars = id_vars,
            value_vars = metrics,
            var_name = 'metric',
            value_name = 'metric_value',
        )
        # if debug: display(data_melted)
        if verbose or verbose: log('Массив транспонирован')

        if verbose or verbose: log('Идет создание словаря с данными разделенными по групперам и метрикам')
        
        grouper = slices+['metric']

        # Разбиение массива на 4 отдельных набора данных по комбинациями региона и метрики
        samples = DottedDict({'_'.join(g): sample for g, sample in data_melted.groupby(grouper) if sample['test_flg'].sum()}) # Создаем подвыборки только там где есть тестовые магазины        self.samples = samples

        if verbose or verbose: log(f'Создано {len(samples)} подвыборок. Ключи доступа: {', '.join(list(samples.keys()))}')

        info_cols = (
            store
            + slices
            + test_flg
        )

        if verbose or verbose: log(f'Информационные колонки: {info_cols}')


        # Параметры k-NN
        knn_params = dict(
            n_neighbors=n_analogues+20, # c запасом, чтобы отфлировать, если попадуться тестовые
            algorithm='auto',
            metric='cosine',
        )

        if verbose or verbose: log('Идет итерирование по созданным подвыборкам ...')
        analogues_samples = []
        analogues_samples_dim = []
        for n, (name, sample) in enumerate(samples.items(), start=1):
            
            if debug: display(sample)

            full_sample = sample.copy() # Нужно оставить для джоина в конце, чтобы примапились периоды проведения теста
            if verbose or verbose: log(f'Выборка {n}. Создан объект full_sample')
            # if debug: display(sample[train_flg])
            train_sample = sample.loc[sample[train_flg[0]] == 1]#.drop(train_flg, axis=1) # Оставляем только тренировочные данные
            if verbose or verbose: log(f'Выборка {n}. Создан объект train_sample')
            if debug: display(train_sample)

            # Перенесение time_label в столбцы
            train_sample_pivoted = (
                train_sample.pivot_table(
                    index = store + slices + test_flg + ['metric'],
                    columns = time_label,
                    values = ['metric_value'],
                )
            )
            if debug: display('TRAIN_SAMPLE_PIVOTED1', train_sample_pivoted)
            # Схлопывание возникающего мультииндекса в колонках
            if isinstance(train_sample_pivoted.columns, pd.MultiIndex): 
                train_sample_pivoted.columns = [str(lvl1) if not lvl2 else 't'+str(int(lvl2)) for lvl1, lvl2 in train_sample_pivoted]
            if verbose or verbose: log(f'Выборка {n}. Создан объект train_sample_pivoted')
            if debug: display('TRAIN_SAMPLE_PIVOTED2', train_sample_pivoted)

            #Скейлинг
            scaler = MinMaxScaler()
            train_sample_pivoted_scaled = pd.DataFrame(scaler.fit_transform(train_sample_pivoted), index=train_sample_pivoted.index, columns=train_sample_pivoted.columns)
            train_sample_pivoted = train_sample_pivoted.reset_index()

            # k-NN по косинусной близости
            knn = NearestNeighbors(**knn_params)
            knn.fit(train_sample_pivoted_scaled)

            # Получаем индексы соседей
            _, idxs = knn.kneighbors()

            # Добавляем к индексам аналогов, индекс самого магазина
            idxs_w_mm = [[i]+list(int(val) for val in row) for i, row in enumerate(idxs)]
        
            # Убираем ненужные данные
            analogues_sample = train_sample_pivoted[info_cols]

        
            # Кракозябра, которая позволяет подтянуть данные аналогов и фильтрануть их от тестовых магазинов
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
                analogue_cols = store + test_flg
                for col in analogue_cols:
                    array = []
                    for x in idxs_w_mm:
                        ls = []
                        cnt = 0
                        for k, idx in enumerate(x):
                            if (k == 0 or analogues_sample.loc[idx, test_flg[0]] != 1): #to_exclude
                                ls.append(analogues_sample.loc[idx, col])
                                cnt += 1
                                if cnt == n_analogues + 1:
                                    break
                        array.append(ls)
                    analogues_sample.loc[:,f'{col}_analogue'] = array
            if verbose or verbose: log(f'Выборка {n}. Создан объект analogues_sample')
            # Разворачиваем списки в табличный вид
            
            # Оставляем только тестовые магазины и разворачиваем списки в строки
            analogues_sample_exploded = (
                pd.DataFrame(analogues_sample[analogues_sample[test_flg[0]]==1])
                .explode(['name_analogue', 'whs_id_analogue', 'test_flg_analogue'])
            )
            if verbose or verbose: log(f'Выборка {n}. Создан объект analogues_sample_exploded')

            merged_sample = analogues_sample_exploded.merge(
                right = full_sample,
                how = 'left',
                left_on = 'whs_id_analogue',
                right_on = 'whs_id',
                suffixes = ('','_todrop')
            )
            merged_sample = merged_sample[[col for col in merged_sample.columns if not col.endswith('_todrop')]] #Убираем дублирующиеся колонки
            if verbose or verbose: log(f'Выборка {n}. Создан объект analogues_sample_exploded')

            # Плохо. Нужно 
            self.samples_list[name] = DottedDict(
                dict(
                    full_sample = full_sample,
                    train_sample_pivoted = train_sample_pivoted,
                    train_sample_pivoted_scaled = train_sample_pivoted_scaled,
                    analogues_sample = analogues_sample,
                    analogues_sample_exploded = analogues_sample_exploded,
                    merged_sample=merged_sample,
                )
            )
            analogues_samples.append(merged_sample)
            analogues_samples_dim.append(pd.DataFrame(analogues_sample))
            
        self.analogues_calc = pd.concat(analogues_samples)
        self.analogues = self.analogues_calc.replace(self.mapping)
        self.analogues_dim = pd.concat(analogues_samples_dim)
    
    def forecast(self, slices_for_agg:Union[list, str]=None):

        analogues = self.analogues.replace(self.reversed_mapping)

        # samples = DottedDict({'_'.join(g): sample for g, sample in analogues.groupby(grouper) if sample['test_flg'].sum()}) # Создаем подвыборки только там где есть тестовые магазины        self.samples = samples

        # forecast_aggregated_list = []
        
        # samples = self.samples_list.copy()
            
        forecast_detailed_list = []
        # for slices_metric, data_dict in samples.items():

        # if analogues_df:
        #     def check_mapping(analogues_df):

        # for self.analogues_calc
        
        grouper = self.slices+['metric']
        for slices_metric, sample in analogues.groupby(grouper):
            
            # Расчет средних по аналогам
            subsamples = [subsample for _, subsample in sample.groupby('test_flg_analogue')]
            analogues_part, test_part = subsamples
            
            if self.slices:
                *slices, metric = slices_metric.split('_')
            else:
                metric = slices_metric[0]


            # Данные по средним
            mean = pd.DataFrame(
                analogues_part
                .groupby(['name','time_label','train_flg'])['metric_value']
                .mean()
                .rename('analogue_mean')
            )
            names = pd.DataFrame(
                analogues_part
                .groupby(['name','time_label','train_flg'])['name_analogue']
                .agg(list)
                .rename('analogue_names')
            )
            analogue_means = (
                mean
                .merge(
                    right=names,
                    how='outer',
                    left_index=True,
                    right_index=True
                )
            )

            len_series = len(analogue_means)
            if self.slices:
                for slice_col, slice_val in zip(self.slices, slices):
                    analogue_means[slice_col] = [slice_val]*len_series
            analogue_means['metric'] = [metric]*len_series
            index = self.slices + ['metric']
            analogue_means.set_index(index, append=True, inplace=True)
            

            # Данные по тестовым магазинам
            test_part = (
                test_part
                .loc[:,['name', 'time_label']+self.slices+['metric','metric_value']]
                .drop_duplicates()
                .set_index(['name', 'time_label']+self.slices+['metric'])
            )
            test_part = test_part.rename({'metric_value':'test'}, axis=1)

            # Объединяем
            forecast_detailed = (
                test_part
                .merge(
                    right=analogue_means,
                    how='outer',
                    left_index=True,
                    right_index=True
                )
                .reset_index()
            )

            # Расчет K
            forecast_detailed['k'] = np.where(forecast_detailed.train_flg==1, forecast_detailed.test / forecast_detailed.analogue_mean, np.nan)
            
            # Расчет medK для деталки
            forecast_detailed['medk'] = forecast_detailed.groupby('name')['k'].transform('median')

            forecast_detailed['forecast'] = forecast_detailed.analogue_mean*forecast_detailed.medk
            forecast_detailed['forecast_'] = forecast_detailed.forecast
            forecast_detailed['forecast'] = np.where(forecast_detailed.train_flg==0, forecast_detailed.forecast, np.nan)

            forecast_detailed_list.append(forecast_detailed)

        self.forecast_detailed = pd.concat(forecast_detailed_list, ignore_index=True)

        # ----------------------------------------------------------
        # Расчет агрегата

        if slices_for_agg:
            if not isinstance(slices, list):
                slices_for_agg = [slices_for_agg]
        else:
            slices_for_agg = self.slices

        forecast_aggregated = (
            self.forecast_detailed
            .groupby(['time_label', 'train_flg']+slices_for_agg+['metric'], as_index=False)[['test', 'analogue_mean', 'forecast','forecast_']] # Добавил forecast, чтобы можно было декомпозировать прогноз
            .mean()
        )
        self.forecast_aggregated = forecast_aggregated

        #-----------------------------------

