import numpy as np
import pandas as pd


class ResultsEvaluator:

    @staticmethod
    def generate_relative_improvement_to_baseline_performance(dataframe: pd.DataFrame, baseline_encoder: str):
        baseline = dataframe[baseline_encoder]
        dataframe_relative_to_baseline = dataframe.drop(baseline_encoder, axis=1)
        for row in range(dataframe_relative_to_baseline.index.size):
            baseline_performance = baseline[row]
            for column in range(dataframe_relative_to_baseline.columns.size):
                dataframe_relative_to_baseline.iloc[row, column] = np.round((dataframe_relative_to_baseline.iloc[row, column] - baseline_performance) / baseline_performance, 10)
        return dataframe_relative_to_baseline        
    

    def generate_mean_value_per_column(dataframe: pd.DataFrame):
        mean_values_dict = {}
        for encoder_cominbation in dataframe.columns:
            performance_per_dataset = dataframe[encoder_cominbation]
            mean = performance_per_dataset.mean()
            mean_values_dict[encoder_cominbation] = mean
        return mean_values_dict