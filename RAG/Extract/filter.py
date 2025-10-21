import pandas as pd

df = pd.read_csv("Extract/data/knowledge_base_output.csv", on_bad_lines='skip')

print(df)