import pandas as pd


class DatasetFactory:
    @staticmethod
    def get_code_dataset() -> pd.DataFrame:
        """
        Return code dataset in the form of a pandas DataFrame.

        Returns:
            pd.DataFrame
        """
        return pd.read_parquet("datasets/human_eval/data.parquet")

    @staticmethod
    def get_math_dataset() -> pd.DataFrame:
        """
        Dataset is quite large, we will only use a subset of it, the first 300 rows.

        Returns:
            pd.DataFrame
        """
        return pd.read_parquet("datasets/gsm8k/data.parquet").iloc[0: 300]
    
