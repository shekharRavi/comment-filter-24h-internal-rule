import pandas as pd
import numpy as np

file_name = '/import/cogsci/ravi/datasets/Embeddia/STY_24sata_comments_hr_001.csv'

df = pd.read_csv(file_name, parse_dates=['created_date','last_change'],low_memory=False,lineterminator='\n')
df = df.fillna(0)
print('Reading Done')
rules = [0,1,2,3,4,5,6,7,8]

data ={}
for rule in rules:
    sel_data = df[df.infringed_on_rule == rule]
    print(rule, len(sel_data))

    data[rule]=len(sel_data)

print(data)

data_df= pd.DataFrame(data.items(), columns=['rule', 'count']) 

data_df.to_csv('rule_stat.csv', index=False)

rules = [2,3,4,5,6,7,1,8,0]

data ={}
for idx, rule in enumerate(rules):
    sel_data = df[df.infringed_on_rule == rule]
    # print('s',rule, len(sel_data), len(save_df))
    if idx == 0:
        train_df = sel_data.sample(frac=0.9)
        val_df = sel_data.drop(train_df.index)

    else:
        if rule in [8,0,1]:
            if len(save_df) < len(sel_data):
                n = int(len(save_df) * 1.5)
                if n < len(sel_data):
                    # print('n.',n, 'sel',len(sel_data))
                    sel_data = sel_data.sample(n = n)
        
        train = sel_data.sample(frac=0.9)
        val = sel_data.drop(train.index)
        
        train_df = pd.concat([train_df, train])
        val_df = pd.concat([val_df, val])
    
    data[rule] = len(sel_data)

    # print(rule, len(sel_data), len(save_df))
print(data)
train_df['label'] = train_df['infringed_on_rule']
val_df['label'] = val_df['infringed_on_rule']
data_df= pd.DataFrame(data.items(), columns=['rule', 'count']) 
data_df.to_csv('rule_stat_selected.csv', index=False)
train_df.to_csv('train_rule_selected_24sata.csv', index=False)
val_df.to_csv('val_rule_selected_24sata.csv', index=False)

print(len(train_df), len(val_df))