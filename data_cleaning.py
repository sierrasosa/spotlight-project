import pandas as pd

#unwrap function to use in the unwrap lambda function
def unest(data, key):
    if key in data.keys():
        return data.get(key)
    else:
        for dkey in data.keys():
            if isinstance(data.get(dkey), dict):
                return unest(data.get(dkey), key)
            else:
                continue
            
#when mongo retrieves data the columns are the top values of the each nest (context and properties for volley)
#make a list of the lowest level names you want as columns (eg. if you want context.app.version just put 'version')
#then use them in this unwrap function with the corresponding nest (eg. context)
def unwrap_df(df, nest, col):
    for col in col:
        df[col] = df.apply(lambda row: unest(row[nest],col), axis = 1)
    df = df.drop(columns = [nest])    
    return df


#some of the variables include lists that need to be unraveled
#this is used in get_all_dummies to do that
def list_dummies(df, col):
    s = pd.Series(df[col])
    s = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)
    return s

#the model can't handle columns that aren't coded (like timezone) so this codes them correctly
def get_all_dummies(df, normal_cols, list_cols):
    all_dummies = df
    for col in list_cols:
        col = list_dummies(df,col)
        all_dummies = all_dummies.merge(col, 'left', left_index=True, right_index=True)
    norm_dummies = pd.get_dummies(df, prefix=list_cols)
    df = all_dummies.merge(norm_dummies, 'left', left_index=True, right_index=True)
    return 