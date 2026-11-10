import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
def l_data(path):
    data = pd.read_csv(path,sep=r"\s+", header=None)
    x=data.iloc[:,2:].copy()
    x=x.loc[:,x.nunique()>1]
    x.columns=range(x.shape[1])
    y=data.iloc[:,1]
    return x,y
def check_data(x,y,name):
    print(f"\n{name} Data Information:")
    print("No of samples:",len(x))
    print("No of features:",x.shape[1])
    print("\n feature types:")
    print(x.dtypes)
    print("\n missing values:")
    print(x.isnull().sum().sum())
    print("\n class distribution:")
    print(y.value_counts().sort_index())
def p_data(x, scale=True):
    categorical_col= x.select_dtypes(include=['object','string']).columns.tolist()
    numerical_col= x.select_dtypes(include=['int64','float64']).columns.tolist()
    transform=[]
    if numerical_col:
        if scale:
            transform.append(('num', StandardScaler(),numerical_col))
        else:
            transform.append(('num', 'passthrough', numerical_col))
    if categorical_col:
        transform.append(('cat', OrdinalEncoder(), categorical_col))
    return ColumnTransformer(transform)
if __name__ == "__main__":
    for n in ["cho","iyer"]:
        path= f"data/{n}.txt"
        x,y = l_data(path)
        check_data(x,y,n)
        preprocessor = p_data(x, scale=True)
        process = preprocessor.fit_transform(x)
        print(f"\npresscessing Done")
        print("Processed shape:", process.shape)
        print("Mean:", process.mean(axis=0).round(2))
        print("Standard deviation:", process.std(axis=0).round(2))

    



    