import pandas as pd
import numpy as np

file_name = '/import/cogsci/ravi/datasets/Embeddia/STY_24sata_comments_hr_001.csv'

df = pd.read_csv(file_name, parse_dates=['created_date','last_change'],low_memory=False)
df = df.fillna(0)
print('Reading Done')
rules = [0,1,2,3,4,5,6,7,8]

data ={}
for rule in rules:
    sel_data = df[df.infringed_on_rule == rule]
    print(rule, len(sel_data))

    data[rule]=len(sel_data)

print(data)
df.to_csv('rule_stat.csv', index=False)