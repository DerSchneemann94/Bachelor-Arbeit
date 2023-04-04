import pandas as pd


class ResultsEvaluator:

    @staticmethod
    def generate_relative_improvement_to_baseline_performance(dataframe: pd.DataFrame, baseline_encoder: str):
        baseline = dataframe[baseline_encoder]
        dataframe_relative_to_baseline = dataframe.drop(baseline_encoder, axis=1)
        for row in range(dataframe_relative_to_baseline.index.size):
            baseline_performance = baseline[row]
            for column in range(dataframe_relative_to_baseline.columns.size):
                dataframe_relative_to_baseline.iloc[row, column] = (dataframe_relative_to_baseline.iloc[row, column] - baseline_performance) / baseline_performance
        return dataframe_relative_to_baseline        