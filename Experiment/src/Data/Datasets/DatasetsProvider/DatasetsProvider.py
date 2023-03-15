from Data.Datasets.DatasetsProvider.DataBaseProviderNotFroundError import DataBaseProviderNotFroundError

class DatasetsProvider():
    
    @staticmethod
    def get_datasets_provider(type: str):
        try:
            return DatasetsProvider.database_provider_dict[type]
        except:
            raise DataBaseProviderNotFroundError(type) 