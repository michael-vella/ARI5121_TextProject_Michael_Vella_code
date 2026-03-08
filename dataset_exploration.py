
import pandas as pd

pd_df: pd.DataFrame = pd.read_parquet("datasets/human_eval/data.parquet")

print("DataFrame info:")
print(pd_df.info())

print("\nDataFrame describe:")
print(pd_df.describe())

print("\nIterate over DataFrame and display row by row")
for index, row in pd_df.iterrows():
    print("\nIndex:")
    print("Task ID:", row["task_id"])
    print("Prompt:", row["prompt"])
    print("Solution:", row["canonical_solution"])
    print("Test:", row["test"])
    print("Entry point:", row["entry_point"])