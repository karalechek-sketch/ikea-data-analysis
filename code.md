# Import Libraries


```python
import os
import re
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import (train_test_split, 
                                     cross_val_score, 
                                     GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, 
                                   TargetEncoder, 
                                   FunctionTransformer)
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.ensemble import (GradientBoostingRegressor, 
                              BaggingRegressor, 
                              HistGradientBoostingRegressor, 
                              RandomForestRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR  
from xgboost import XGBRegressor 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

```

# Load Dataset


```python
def download_document(file_name, document_url):
    if os.path.exists(file_name):
        pass
    else:
        response = requests.get(document_url)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
        else:
            print(f'Failed to download the document. Status code: {response.status_code}')

file_name = 'ikea.csv'
document_url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv'
download_document(file_name, document_url)
```

# Exploratory Data Analysis (EDA)


```python
df = pd.read_csv('ikea.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>item_id</th>
      <th>name</th>
      <th>category</th>
      <th>price</th>
      <th>old_price</th>
      <th>sellable_online</th>
      <th>link</th>
      <th>other_colors</th>
      <th>short_description</th>
      <th>designer</th>
      <th>depth</th>
      <th>height</th>
      <th>width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>90420332</td>
      <td>FREKVENS</td>
      <td>Bar furniture</td>
      <td>265.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>https://www.ikea.com/sa/en/p/frekvens-bar-tabl...</td>
      <td>No</td>
      <td>Bar table, in/outdoor,          51x51 cm</td>
      <td>Nicholai Wiig Hansen</td>
      <td>NaN</td>
      <td>99.0</td>
      <td>51.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>368814</td>
      <td>NORDVIKEN</td>
      <td>Bar furniture</td>
      <td>995.0</td>
      <td>No old price</td>
      <td>False</td>
      <td>https://www.ikea.com/sa/en/p/nordviken-bar-tab...</td>
      <td>No</td>
      <td>Bar table,          140x80 cm</td>
      <td>Francis Cayouette</td>
      <td>NaN</td>
      <td>105.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>9333523</td>
      <td>NORDVIKEN / NORDVIKEN</td>
      <td>Bar furniture</td>
      <td>2095.0</td>
      <td>No old price</td>
      <td>False</td>
      <td>https://www.ikea.com/sa/en/p/nordviken-nordvik...</td>
      <td>No</td>
      <td>Bar table and 4 bar stools</td>
      <td>Francis Cayouette</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>80155205</td>
      <td>STIG</td>
      <td>Bar furniture</td>
      <td>69.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>https://www.ikea.com/sa/en/p/stig-bar-stool-wi...</td>
      <td>Yes</td>
      <td>Bar stool with backrest,          74 cm</td>
      <td>Henrik Preutz</td>
      <td>50.0</td>
      <td>100.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>30180504</td>
      <td>NORBERG</td>
      <td>Bar furniture</td>
      <td>225.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>https://www.ikea.com/sa/en/p/norberg-wall-moun...</td>
      <td>No</td>
      <td>Wall-mounted drop-leaf table,         ...</td>
      <td>Marcus Arvonen</td>
      <td>60.0</td>
      <td>43.0</td>
      <td>74.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.drop(columns=['Unnamed: 0'])
```


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_id</th>
      <th>price</th>
      <th>depth</th>
      <th>height</th>
      <th>width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.694000e+03</td>
      <td>3694.000000</td>
      <td>2231.000000</td>
      <td>2706.000000</td>
      <td>3105.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.863240e+07</td>
      <td>1078.208419</td>
      <td>54.379202</td>
      <td>101.679970</td>
      <td>104.470853</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.888709e+07</td>
      <td>1374.652494</td>
      <td>29.958351</td>
      <td>61.097585</td>
      <td>71.133771</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.848700e+04</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.039057e+07</td>
      <td>180.900000</td>
      <td>38.000000</td>
      <td>67.000000</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.928808e+07</td>
      <td>544.700000</td>
      <td>47.000000</td>
      <td>83.000000</td>
      <td>80.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.040357e+07</td>
      <td>1429.500000</td>
      <td>60.000000</td>
      <td>124.000000</td>
      <td>140.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.993262e+07</td>
      <td>9585.000000</td>
      <td>257.000000</td>
      <td>700.000000</td>
      <td>420.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (3694, 13)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3694 entries, 0 to 3693
    Data columns (total 13 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   item_id            3694 non-null   int64  
     1   name               3694 non-null   object 
     2   category           3694 non-null   object 
     3   price              3694 non-null   float64
     4   old_price          3694 non-null   object 
     5   sellable_online    3694 non-null   bool   
     6   link               3694 non-null   object 
     7   other_colors       3694 non-null   object 
     8   short_description  3694 non-null   object 
     9   designer           3694 non-null   object 
     10  depth              2231 non-null   float64
     11  height             2706 non-null   float64
     12  width              3105 non-null   float64
    dtypes: bool(1), float64(4), int64(1), object(7)
    memory usage: 350.0+ KB
    


```python
df.isnull().sum()
```




    item_id                 0
    name                    0
    category                0
    price                   0
    old_price               0
    sellable_online         0
    link                    0
    other_colors            0
    short_description       0
    designer                0
    depth                1463
    height                988
    width                 589
    dtype: int64




```python
df.isnull().sum() * 100 / len(df)
```




    item_id               0.000000
    name                  0.000000
    category              0.000000
    price                 0.000000
    old_price             0.000000
    sellable_online       0.000000
    link                  0.000000
    other_colors          0.000000
    short_description     0.000000
    designer              0.000000
    depth                39.604764
    height               26.746075
    width                15.944775
    dtype: float64




```python
df.columns
```




    Index(['item_id', 'name', 'category', 'price', 'old_price', 'sellable_online',
           'link', 'other_colors', 'short_description', 'designer', 'depth',
           'height', 'width'],
          dtype='object')




```python
df.duplicated().sum()
```




    np.int64(0)




```python
print(df.nunique().to_markdown())
```

    |                   |    0 |
    |:------------------|-----:|
    | item_id           | 2962 |
    | name              |  607 |
    | category          |   17 |
    | price             |  979 |
    | old_price         |  365 |
    | sellable_online   |    2 |
    | link              | 2962 |
    | other_colors      |    2 |
    | short_description | 1706 |
    | designer          |  381 |
    | depth             |  114 |
    | height            |  193 |
    | width             |  263 |
    


```python
df.duplicated(subset=['item_id']).sum()
```




    np.int64(732)




```python
df[df["item_id"].duplicated(keep=False)].sort_values(by="item_id")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_id</th>
      <th>name</th>
      <th>category</th>
      <th>price</th>
      <th>old_price</th>
      <th>sellable_online</th>
      <th>link</th>
      <th>other_colors</th>
      <th>short_description</th>
      <th>designer</th>
      <th>depth</th>
      <th>height</th>
      <th>width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1918</th>
      <td>91415</td>
      <td>TROFAST</td>
      <td>Nursery furniture</td>
      <td>5.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>https://www.ikea.com/sa/en/p/trofast-lid-white...</td>
      <td>No</td>
      <td>Lid,          20x28 cm</td>
      <td>Studio Copenhagen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>1834</th>
      <td>91415</td>
      <td>TROFAST</td>
      <td>Children's furniture</td>
      <td>5.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>https://www.ikea.com/sa/en/p/trofast-lid-white...</td>
      <td>No</td>
      <td>Lid,          20x28 cm</td>
      <td>Studio Copenhagen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>2560</th>
      <td>102065</td>
      <td>LYCKSELE LÖVÅS</td>
      <td>Sofas &amp; armchairs</td>
      <td>495.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>https://www.ikea.com/sa/en/p/lycksele-loevas-m...</td>
      <td>No</td>
      <td>Mattress,          140x188 cm</td>
      <td>IKEA of Sweden</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>151</th>
      <td>102065</td>
      <td>LYCKSELE LÖVÅS</td>
      <td>Beds</td>
      <td>495.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>https://www.ikea.com/sa/en/p/lycksele-loevas-m...</td>
      <td>No</td>
      <td>Mattress,          140x188 cm</td>
      <td>IKEA of Sweden</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2178</th>
      <td>105064</td>
      <td>LIATORP</td>
      <td>Sideboards, buffets &amp; console tables</td>
      <td>445.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>https://www.ikea.com/sa/en/p/liatorp-console-t...</td>
      <td>No</td>
      <td>Console table,          133x37 cm</td>
      <td>Carina Bengs</td>
      <td>NaN</td>
      <td>75.0</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2860</th>
      <td>99323614</td>
      <td>SMÅGÖRA</td>
      <td>Tables &amp; desks</td>
      <td>370.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>https://www.ikea.com/sa/en/p/smagoera-changing...</td>
      <td>No</td>
      <td>Changing tbl/bookshelf w 1 shlf ut</td>
      <td>IKEA of Sweden</td>
      <td>40.0</td>
      <td>91.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>99323925</td>
      <td>STENSELE</td>
      <td>Bar furniture</td>
      <td>550.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>https://www.ikea.com/sa/en/p/stensele-bar-tabl...</td>
      <td>No</td>
      <td>Bar table,          70x70 cm</td>
      <td>Maja Ganszyniec</td>
      <td>NaN</td>
      <td>104.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>3028</th>
      <td>99323925</td>
      <td>STENSELE</td>
      <td>Tables &amp; desks</td>
      <td>550.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>https://www.ikea.com/sa/en/p/stensele-bar-tabl...</td>
      <td>No</td>
      <td>Bar table,          70x70 cm</td>
      <td>Maja Ganszyniec</td>
      <td>NaN</td>
      <td>104.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>409</th>
      <td>99902661</td>
      <td>VITTSJÖ</td>
      <td>Bookcases &amp; shelving units</td>
      <td>609.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>https://www.ikea.com/sa/en/p/vittsjoe-shelving...</td>
      <td>No</td>
      <td>Shelving unit with laptop table,      ...</td>
      <td>Johan Kroon</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>202.0</td>
    </tr>
    <tr>
      <th>2737</th>
      <td>99902661</td>
      <td>VITTSJÖ</td>
      <td>Tables &amp; desks</td>
      <td>609.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>https://www.ikea.com/sa/en/p/vittsjoe-shelving...</td>
      <td>No</td>
      <td>Shelving unit with laptop table,      ...</td>
      <td>Johan Kroon</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>202.0</td>
    </tr>
  </tbody>
</table>
<p>1352 rows × 13 columns</p>
</div>



# Data Cleaning & Data Visualization


```python
df_base = df.copy()
```


```python
df_base = df_base.drop_duplicates(subset=['item_id']).reset_index(drop=True)
df_base.drop(columns=['link'], inplace=True)
```


```python
df_base.duplicated(subset=['item_id']).sum()
```




    np.int64(0)




```python
df_base.drop(columns=['item_id'], inplace=True)
```


```python
df_base.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>category</th>
      <th>price</th>
      <th>old_price</th>
      <th>sellable_online</th>
      <th>other_colors</th>
      <th>short_description</th>
      <th>designer</th>
      <th>depth</th>
      <th>height</th>
      <th>width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FREKVENS</td>
      <td>Bar furniture</td>
      <td>265.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>No</td>
      <td>Bar table, in/outdoor,          51x51 cm</td>
      <td>Nicholai Wiig Hansen</td>
      <td>NaN</td>
      <td>99.0</td>
      <td>51.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NORDVIKEN</td>
      <td>Bar furniture</td>
      <td>995.0</td>
      <td>No old price</td>
      <td>False</td>
      <td>No</td>
      <td>Bar table,          140x80 cm</td>
      <td>Francis Cayouette</td>
      <td>NaN</td>
      <td>105.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NORDVIKEN / NORDVIKEN</td>
      <td>Bar furniture</td>
      <td>2095.0</td>
      <td>No old price</td>
      <td>False</td>
      <td>No</td>
      <td>Bar table and 4 bar stools</td>
      <td>Francis Cayouette</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>STIG</td>
      <td>Bar furniture</td>
      <td>69.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>Yes</td>
      <td>Bar stool with backrest,          74 cm</td>
      <td>Henrik Preutz</td>
      <td>50.0</td>
      <td>100.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NORBERG</td>
      <td>Bar furniture</td>
      <td>225.0</td>
      <td>No old price</td>
      <td>True</td>
      <td>No</td>
      <td>Wall-mounted drop-leaf table,         ...</td>
      <td>Marcus Arvonen</td>
      <td>60.0</td>
      <td>43.0</td>
      <td>74.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df_base["category"].value_counts().to_markdown())        
```

    | category                             |   count |
    |:-------------------------------------|--------:|
    | Bookcases & shelving units           |     548 |
    | Chairs                               |     438 |
    | Sofas & armchairs                    |     380 |
    | Tables & desks                       |     370 |
    | Wardrobes                            |     220 |
    | Beds                                 |     208 |
    | Outdoor furniture                    |     197 |
    | Cabinets & cupboards                 |     187 |
    | Chests of drawers & drawer units     |     111 |
    | TV & media furniture                 |      89 |
    | Children's furniture                 |      84 |
    | Bar furniture                        |      47 |
    | Trolleys                             |      23 |
    | Nursery furniture                    |      22 |
    | Café furniture                       |      18 |
    | Sideboards, buffets & console tables |      10 |
    | Room dividers                        |      10 |
    


```python
plt.figure(figsize=(10, 6))
order = df_base['category'].value_counts().index
ax = sns.countplot(data=df_base, y='category', order=order, palette='viridis')  

for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.title('Кількість товарів у кожній категорії')
plt.xlabel('Кількість')
plt.ylabel('Категорія')
plt.tight_layout() 
plt.show()

```

    C:\Users\Anna\AppData\Local\Temp\ipykernel_17484\2022251682.py:3: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      ax = sns.countplot(data=df_base, y='category', order=order, palette='viridis')
    


    
![png](output_24_1.png)
    



```python
print((df_base.groupby('category')['price'].median().sort_values(ascending=False)).to_markdown())
```

    | category                             |   price |
    |:-------------------------------------|--------:|
    | Wardrobes                            |  1950   |
    | Sofas & armchairs                    |  1157   |
    | Beds                                 |  1093.5 |
    | Sideboards, buffets & console tables |   927.5 |
    | TV & media furniture                 |   860   |
    | Cabinets & cupboards                 |   845   |
    | Room dividers                        |   595   |
    | Trolleys                             |   537   |
    | Chests of drawers & drawer units     |   499   |
    | Chairs                               |   497   |
    | Tables & desks                       |   475   |
    | Bar furniture                        |   445   |
    | Café furniture                       |   370   |
    | Outdoor furniture                    |   345   |
    | Nursery furniture                    |   322.5 |
    | Bookcases & shelving units           |   310   |
    | Children's furniture                 |   202.5 |
    


```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_base, x='price', y='category')
plt.title('Розкид цін за категоріями')
plt.xlabel('Ціна')
plt.ylabel('Категорія')
plt.grid(axis='x') 
plt.show()
```


    
![png](output_26_0.png)
    


Провівши кількісний та ціновий аналіз товарів за категорією бачимо, що кількісно переважають категорії Bookcases & Shelving units,
Chairs, Sofas & Armchairs та Table & Desks найдорожчою категорією є 'Wardrobes', 'Sofas & armchairs' та 'Beds'. 


```python
plt.figure(figsize=(9,4))
sns.histplot(df_base['price'], kde=True, bins=20)
plt.title('Розподіл цін на товари IKEA')
plt.xlabel('Ціна')
plt.ylabel('Кількість')
plt.show()
```


    
![png](output_28_0.png)
    


Ми бачимо, що розподіл ціни сильно зміщений вправо (positive skew). Більшість меблів IKEA коштують до 2000 одиниць, але є поодинокі екземпляри дорожче 8000. Це означає, що середня ціна буде вищою за медіану.
Оскільки високі ціни є реальними ринковими даними для певних категорій (наприклад, гардеробів), я вирішила не видаляти їх, щоб модель могла розпізнавати дорогі сегменти. 


```python
plt.figure(figsize=(9, 4))
df_base['price_log'] = np.log(df_base['price']) 
sns.histplot(df_base['price_log'], kde=True, binwidth=0.2, color='olive')
plt.title('Розподіл цін (логарифмічна шкала)')
plt.show()
```


    
![png](output_30_0.png)
    


Для стабілізації дисперсії та приведення розподілу цільової змінної до нормального вигляду було застосовано логарифмічне перетворення (log-transformation).
Це дозволяє зменшити вплив цінових викидів та покращити якість майбутньої прогнозної моделі.


```python
df_base['sellable_online'].value_counts(normalize=True)                                         
```




    sellable_online
    True     0.993585
    False    0.006415
    Name: proportion, dtype: float64




```python
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df_base, x='sellable_online', palette='viridis')
for i in ax.containers:
    ax.bar_label(i)
plt.title('Розподіл онлайн-доступності товарів')
plt.xlabel('Доступно онлайн (True/False)')
plt.ylabel('Кількість товарів')
plt.show()
```

    C:\Users\Anna\AppData\Local\Temp\ipykernel_17484\1036427866.py:2: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      ax = sns.countplot(data=df_base, x='sellable_online', palette='viridis')
    


    
![png](output_33_1.png)
    


Аналіз стовпця sellable_online показав, що понад 99% товарів доступні для онлайн-замовлення. Через відсутність варіативності, ця ознака не несе інформативного навантаження для побудови прогнозних моделей і може бути вилучена з подальшого розгляду.


```python
df_base.drop(columns=['sellable_online'], inplace=True)
```


```python
df_base['other_colors'].value_counts(normalize=True) 
```




    other_colors
    No     0.552667
    Yes    0.447333
    Name: proportion, dtype: float64




```python
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=df_base, x='other_colors', palette='Blues', edgecolor='black')
for i in ax.containers:
    ax.bar_label(i)
plt.title('Загальна кількість продуктів з додатковими кольорами та без них', fontsize=14)
plt.xlabel('Наявність інших кольорів')
plt.ylabel('Кількість товарів')
plt.show()
```

    C:\Users\Anna\AppData\Local\Temp\ipykernel_17484\649557332.py:2: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      ax = sns.countplot(data=df_base, x='other_colors', palette='Blues', edgecolor='black')
    


    
![png](output_37_1.png)
    



```python
plt.figure(figsize=(8, 6))
price_median = df_base.groupby('other_colors')['price'].median().reset_index()
ax_price = sns.barplot(data=price_median, x='other_colors', y='price', palette='magma', edgecolor='black')
for i in ax_price.containers:
    ax_price.bar_label(i, fmt='%.0f', padding=3)
plt.title('Середня ціна продуктів залежно від варіантів кольорів', fontsize=14)
plt.xlabel('Наявність інших кольорів')
plt.ylabel('Середня ціна')
plt.show()
```

    C:\Users\Anna\AppData\Local\Temp\ipykernel_17484\2634378903.py:3: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      ax_price = sns.barplot(data=price_median, x='other_colors', y='price', palette='magma', edgecolor='black')
    


    
![png](output_38_1.png)
    


Ціновий розрив: Середня ціна товарів з додатковими кольорами (745) на 50% вища, ніж у товарів без них (495). Це статистичний факт у межах цього датасету.
Розподіл у каталозі: Товари без додаткових кольорів складають більшість асортименту (55%), але група з кольорами також є значною (45%).


```python
df_base['designer'].unique()
```




    array(['Nicholai Wiig Hansen', 'Francis Cayouette', 'Henrik Preutz',
           'Marcus Arvonen', 'Carina Bengs', 'K Hagberg/M Hagberg',
           'Sarah Fager', 'Ehlén Johansson', 'Nike Karlsson',
           'Maja Ganszyniec', 'Karl Malmvall',
           'John/Jonas/Petrus/Paul/Caroline', 'Nike Karlsson/Maja Ganszyniec',
           'J Karlsson/N Karlsson', 'IKEA of Sweden/Karl Malmvall',
           'IKEA of Sweden', 'Nike Karlsson/J Karlsson/N Karlsson',
           'Ola Wihlborg', 'IKEA of Sweden/Tina Christensen',
           'IKEA of Sweden/K Hagberg/M Hagberg',
           'Ola Wihlborg/IKEA of Sweden',
           '504.689.53 Small and easy-to-place chair-bed which can easily be converted into a single bed.The storage space under the seat has room for bedlinen or other things.Just as nice to look at from all sides – perfect to place in the middle of the room or use as a room divider.The cushion cover is easy to keep clean and fresh, as you can take it off and machine-wash it.Easy to assemble.1 cushion included.',
           'IKEA of Sweden/Ebba Strandmark',
           'K Hagberg/M Hagberg/IKEA of Sweden', 'Jon Karlsson',
           'IKEA of Sweden/Carina Bengs', 'David Wahl',
           'Jon Karlsson/IKEA of Sweden', 'IKEA of Sweden/Paulin Machado',
           'IKEA of Sweden/Eva Lilja Löwenhielm',
           'IKEA of Sweden/Synnöve Mork/Ola Wihlborg',
           'IKEA of Sweden/David Wahl', 'Ebba Strandmark/IKEA of Sweden',
           'Eva Lilja Löwenhielm',
           '903.310.91 The door can be hung with the opening to the right or the left.To be completed with 3 HJÄLPA hinges, sold separately.Fits with PLATSA wardrobe system.Knobs and handles are sold separately.',
           'IKEA of Sweden/Anna Efverlund', 'Paulin Machado',
           'Jonas Hultqvist', 'Gustav Carlberg', 'Carl Öjerstam',
           'Virgil Abloh', 'Monika Mulder/IKEA of Sweden',
           '443.610.10 Easy to keep clean since you can remove the fabric and wash it by machine.NEVER use for infants and young children.WARNING! Entanglement and strangulation hazard.The net is not intended to be used as protection against mosquitos or other insects.',
           'T Christensen/K Legaard', 'Tord Björklund',
           'Tom Dixon/IKEA of Sweden',
           'Synnöve Mork/Ola Wihlborg/IKEA of Sweden',
           'Ola Wihlborg/IKEA of Sweden/Synnöve Mork',
           'IKEA of Sweden/L Hilland/Lisa Hilland',
           'IKEA of Sweden/L Hilland',
           '702.842.03 A sofa-bed with small, neat dimensions which is easy to furnish with, even when space is limited.Readily converts into a bed.Fixed cover.',
           'Carina Bengs/IKEA of Sweden', 'Tom Dixon',
           '902.994.49 A good solution where space is limited.It’s easier to get in and out of the bed with a centered ladder. You can use the space under the bed for storage, a workspace or seating.Recommended for ages from 6 years.High beds and the upper bed of bunk or loft beds are not suitable for children under 6 years due to the risk of injury from falls.Bed base included.Mattress and bedlinen are sold separately.Max load indicates static weight, in other words the load which the bed withstands if you lie or sit on it.',
           'Fredriksson/L Löwenhielm/Hilland',
           'IKEA of Sweden/Tord Björklund',
           'Fredriksson/L Löwenhielm/Hilland/IKEA of Sweden',
           'IKEA of Sweden/Henrik Preutz', 'IKEA of Sweden/Tom Dixon',
           'Malin Unnborn', 'A Huldén/S Dahlman',
           'IKEA of Sweden/Virgil Abloh', 'S Edholm/L Ullenius',
           'IKEA of Sweden/Francis Cayouette/Ehlén Johansson',
           'Francis Cayouette/Ehlén Johansson/IKEA of Sweden',
           'Ebba Strandmark', 'J Asshoff/H Brogård',
           'Lisa Hilland/David Wahl/IKEA of Sweden',
           'Henrik Preutz/IKEA of Sweden', 'C Halskov/H Dalsgaard',
           'Ehlén Johansson/IKEA of Sweden',
           'Ola Wihlborg/Synnöve Mork/IKEA of Sweden',
           'David Wahl/Lisa Hilland/IKEA of Sweden',
           'IKEA of Sweden/Ehlén Johansson/Francis Cayouette',
           'IKEA of Sweden/Fredriksson/L Löwenhielm/Hilland',
           'IKEA of Sweden/Ehlén Johansson',
           'Ehlén Johansson/Francis Cayouette/IKEA of Sweden', 'Chenyi Ke',
           'Johan Kroon', 'Gillis Lundgren',
           '502.638.38 Shallow shelves help you to use small wall spaces effectively by accommodating small items in a minimum of space.Adjustable shelves; adapt space between shelves according to your needs.A simple unit can be enough storage for a limited space or the foundation for a larger storage solution if your needs change.This furniture must be fixed to the wall with the enclosed wall fastener.Different wall materials require different types of fixing devices. Use fixing devices suitable for the walls in your home, sold separately.Min. ceiling height required: 205 cm.1 fixed shelf and 4 adjustable shelves included.May be completed with BILLY corner fitting to form a stable corner unit.May be completed with BILLY height extension unit in the same width for added storage vertically.May be completed with doors; available in different colours and designs.',
           'IKEA of Sweden/Jon Karlsson', 'Gillis Lundgren/IKEA of Sweden',
           'IKEA of Sweden/Carl Öjerstam',
           '392.873.98 Glass doors keep your favourite items free from dust but still visible.Adjustable shelves; adapt space between shelves according to your needs.Adjustable hinges allow you to adjust the door horizontally and vertically.This furniture must be fixed to the wall with the enclosed wall fastener.Handle with care! A damaged edge or scratched surface can cause the glass to suddenly crack and/or break. Avoid collisions from the side - this is where the glass is most vulnerable.Different wall materials require different types of fixing devices. Use fixing devices suitable for the walls in your home, sold separately.Min. ceiling height required: 205 cm.1 fixed shelf and 4 adjustable shelves included.May be completed with BILLY height extension unit in the same width for added storage vertically.',
           '204.099.36 Choose whether you want to place it vertically or horizontally and use it as a shelf or sideboard.This furniture must be fixed to the wall with the enclosed wall fastener.Different wall materials require different types of fixing devices. Use fixing devices suitable for the walls in your home, sold separately.Two persons are needed for the assembly of this furniture.May be completed with KALLAX insert, sold separately.',
           '602.957.06 With a media shelf you can make the most of the wall area, while freeing up space on the floor.Different wall materials require different types of fixing devices. Use fixing devices suitable for the walls in your home, sold separately.',
           'Sarah Fager/IKEA of Sweden',
           'K Hagberg/M Hagberg/Gillis Lundgren',
           'IKEA of Sweden/Marcus Arvonen', 'Marcus Arvonen/IKEA of Sweden',
           '104.283.32 Suitable for both indoor and outdoor use.The cover is easy to put on and remove.Protects your things from dust.At the top of the cover there is an opening that allows air to circulate.You can also use the shelving unit with cover as a small greenhouse.',
           'Gillis Lundgren/K Hagberg/M Hagberg', 'Niels Gammelgaard',
           'H Preutz/A Fredriksson', 'IKEA of Sweden/Gillis Lundgren',
           '903.233.93 Drill template for marking the hole location of knobs or handles makes it easier for you to place them correctly.The strip on the drill template is put against the edge of the door or drawer front; helps you to place the handle perfectly straight vertically or horizontally.',
           '804.302.04 Suitable for both indoor and outdoor use.The cover is easy to put on and remove.Protects your things from dust.At the top of the cover there is an opening that allows air to circulate.You can also use the shelving unit with cover as a small greenhouse.',
           '204.237.20 Easy to assemble.The insert looks nice in a room divider as the back has also been finished.You can use the inserts to customise KALLAX shelving unit so that it suits your storage needs.Dimensioned for KALLAX shelving unit.To be completed with KALLAX shelving unit.',
           'Carl Öjerstam/IKEA of Sweden', 'Tord Björklund/IKEA of Sweden',
           'Carl Öjerstam/Marcus Arvonen/IKEA of Sweden', 'Mikael Warnhammar',
           '802.945.03 It’s easy to keep the cables from your TV and other devices out of sight but close at hand, as there are several cable outlets at the back of the TV bench.You can choose to stand the TV bench on the floor or mount it on the wall to free up floor space.If you want to organise inside you can complement with BESTÅ interior fittings.Steady also on uneven floors, thanks to the adjustable feet.This furniture must be fixed to the wall with the enclosed wall fastener.This TV bench can take a max load of 50 kg on the top.Different wall materials require different types of fixing devices. Use fixing devices suitable for the walls in your home, sold separately.May be completed with STALLARP, STUBBARP or NANNARP legs. This TV bench requires 4 legs and 1 BESTÅ supporting leg.May be completed with SULARP legs. This TV bench requires 2 legs and 1 BESTÅ supporting leg.',
           'K Malmvall/E Lilja Löwenhielm',
           '702.945.08 It’s easy to keep the cables from your TV and other devices out of sight but close at hand, as there are several cable outlets at the back of the TV bench.You can choose to stand the TV bench on the floor or mount it on the wall to free up floor space.If you want to organise inside you can complement with BESTÅ interior fittings.Steady also on uneven floors, thanks to the adjustable feet.This furniture must be fixed to the wall with the enclosed wall fastener.This TV bench can take a max load of 50 kg on the top.Different wall materials require different types of fixing devices. Use fixing devices suitable for the walls in your home, sold separately.May be completed with STALLARP, STUBBARP or NANNARP legs. This TV bench requires 4 legs and 2 BESTÅ supporting legs.May be completed with SULARP legs. This TV bench requires 2 legs and 2 BESTÅ supporting legs.',
           '704.436.26 The front may be used as a door or as a drawer front.You can choose to mount the door on the right or left side.The doors keep your belongings hidden and free from dust.To be completed with BESTÅ hinges if used as door. Sold separately.To be completed with BESTÅ drawer frame 60x25x40 cm and BESTÅ drawer runner if used as drawer front. Sold separately.May be completed with knobs or handles. Sold separately.Drill template for marking of hole positions for handles or knobs is sold separately.',
           '402.998.85 It’s easy to keep the cables from your TV and other devices out of sight but close at hand, as there are several cable outlets at the back of the TV bench.If you want to organise inside you can complement with BESTÅ interior fittings.Steady also on uneven floors, thanks to the adjustable feet.For safety reasons this TV bench shall not be hung on the wall.The TV bench must be fixed to the wall with the included wall fastener.This TV bench can take a max load of 50 kg on the top.Different wall materials require different types of fixing devices. Use fixing devices suitable for the walls in your home, sold separately.May be completed with STALLARP, STUBBARP or NANNARP legs. This TV bench requires 4 legs and 1 BESTÅ supporting leg.May be completed with SULARP legs. This TV bench requires 2 legs and 1 BESTÅ supporting leg.',
           'Gillis Lundgren/K Hagberg/M Hagberg/IKEA of Sweden',
           '604.415.81 Drawers make it easy to keep your things organised.To be completed with BESTÅ drawer frame 60X15X40 cm and BESTÅ drawer runner, sold separately.May be completed with knobs or handles. Sold separately.Drill template for marking of hole positions for handles or knobs is sold separately.',
           'J Löfgren/J Pettersson',
           'Gillis Lundgren/IKEA of Sweden/K Hagberg/M Hagberg',
           '702.998.79 It’s easy to keep the cables from your TV and other devices out of sight but close at hand, as there are several cable outlets at the back of the TV bench.If you want to organise inside you can complement with BESTÅ interior fittings.Steady also on uneven floors, thanks to the adjustable feet.For safety reasons this TV bench shall not be hung on the wall.This furniture must be fixed to the wall with the enclosed wall fastener.This TV bench can take a max load of 50 kg on the top.Different wall materials require different types of fixing devices. Use fixing devices suitable for the walls in your home, sold separately.May be completed with STALLARP, STUBBARP or NANNARP legs. This TV bench requires 4 legs and 2 BESTÅ supporting legs.May be completed with SULARP legs. This TV bench requires 2 legs and 2 BESTÅ supporting legs.',
           '802.612.58 You can choose to use either the soft-closing or push-open function.With the push-opener you don’t need knobs or handles and can open the door with a light push.With the soft-closing function your doors close silently and softly. Knobs and handles are sold separately.If you choose the soft closing function for your BESTÅ, we recommend complementing the fronts with knobs/handles to make the drawers/cabinets more convenient to open.',
           'Charlie Styrbjörn',
           '502.755.96 Adjustable hinges allow you to adjust the door horizontally and vertically.Behind the panel doors you can keep your belongings hidden and free from dust.Hinges included.Knobs included.1 door will fit BILLY bookcase 40 cm and 2 doors will fit BILLY bookcase 80 cm.Can be used on the corner unit only if the shelf next to it has no doors.The door does not fit bookcases purchased in the spring of 2014 or earlier.',
           '002.756.74 Adjustable hinges allow you to adjust the door horizontally and vertically.Panel/glass doors provide dust-free storage and let you hide or display things according to your needs.Hinges included.Knobs included.Handle with care! A damaged edge or scratched surface can cause the glass to suddenly crack and/or break. Avoid collisions from the side - this is where the glass is most vulnerable.1 door will fit BILLY bookcase 40 cm and 2 doors will fit BILLY bookcase 80 cm.Can be used on the corner unit only if the shelf next to it has no doors.The door does not fit bookcases purchased in the spring of 2014 or earlier.',
           '203.305.99 The TV-bracket can be angled for change of viewing position, and easy access to cables and connections.The TV bracket has integrated cable management so you can easily gather and conceal wires for a neater TV solution.UPPLEVA TV-brackets are VESA-compatibleFits most 37-55" flat screen TVs.Fits MOSTORP, MALSJÖ and BESTÅ TV benches.',
           '704.289.04 Metal legs raise your EKET combination from the floor, giving a light airy look and making it easy to clean the floor underneath.Adjustable feet; stands steady also on an uneven floor.',
           '502.756.19 Adjustable hinges allow you to adjust the door horizontally and vertically.Glass doors keep your favourite items free from dust but still visible.Hinges included.Knobs included.Handle with care! A damaged edge or scratched surface can cause the glass to suddenly crack and/or break. Avoid collisions from the side - this is where the glass is most vulnerable.Can be used on the corner unit only if the shelf next to it has no doors.',
           'Tina Christensen',
           '604.415.62 Drawers make it easy to keep your things organised.To be completed with BESTÅ drawer frame 60X15X40 cm and BESTÅ drawer runner, sold separately.May be completed with knobs or handles. Sold separately.Drill template for marking of hole positions for handles or knobs is sold separately.',
           '104.415.69 You can choose to mount the door on the right or left side.To be completed with BESTÅ hinges.1 door requires 1 pack of hinges. Sold separately.May be completed with knobs or handles. Sold separately.Drill template for marking of hole positions for handles or knobs is sold separately.',
           'Francis Cayouette/Eva Lilja Löwenhielm',
           'IKEA of Sweden/Sarah Fager',
           '904.436.25 The doors keep your belongings hidden and free from dust.You can choose to mount the door on the right or left side.To be completed with BESTÅ hinges.1 door requires 1 pack of hinges. Sold separately.May be completed with knobs or handles. Sold separately.Drill template for marking of hole positions for handles or knobs is sold separately.',
           '704.415.66 You can choose to mount the door on the right or left side.To be completed with BESTÅ hinges.1 door requires 1 pack of hinges. Sold separately.May be completed with knobs or handles. Sold separately.Drill template for marking of hole positions for handles or knobs is sold separately.',
           '304.415.54 The door keeps your belongings hidden and free from dust.You can choose to mount the door on the right or left side.To be completed with BESTÅ hinges.1 door requires 1 pack of hinges. Sold separately.May be completed with knobs or handles. Sold separately.Drill template for marking of hole positions for handles or knobs is sold separately.',
           'Francis Cayouette/IKEA of Sweden',
           '804.415.56 The front may be used as a door or as a drawer front.You can choose to mount the door on the right or left side.The door keeps your belongings hidden and free from dust.To be completed with BESTÅ hinges if used as door. Sold separately.To be completed with BESTÅ drawer frame 60x25x40 cm and BESTÅ drawer runner if used as drawer front. Sold separately.May be completed with knobs or handles. Sold separately.Drill template for marking of hole positions for handles or knobs is sold separately.',
           'Eva Lilja Löwenhielm/IKEA of Sweden',
           '304.289.15 A simple unit can be enough storage for a limited space or the foundation for a larger storage solution if your needs change.You can choose to place the cabinet on the floor or mount it on the wall to free up floor space.You can create your own unique solution by freely combining cabinets of different sizes, with or without doors and drawers.The drawers have integrated push-openers, so you don’t need knobs or handles and can open the drawer with just a light push.Must be completed with EKET suspension rail if you choose to mount the frame on the wall. This frame requires 1 suspension rail, 35 cm long, sold separately.To be completed with feet, legs or plinth if you choose to place the cabinet on the floor. Sold separately.The max load for a wall-hung cabinet depends on the wall material.',
           'Marcus Arvonen/Carl Öjerstam/IKEA of Sweden',
           '504.436.27 Drawers make it easy to keep your things organised.To be completed with BESTÅ drawer frame 60X15X40 cm and BESTÅ drawer runner, sold separately.May be completed with knobs or handles. Sold separately.Drill template for marking of hole positions for handles or knobs is sold separately.',
           '102.998.77 It’s easy to keep the cables from your TV and other devices out of sight but close at hand, as there are several cable outlets at the back of the TV bench.If you want to organise inside you can complement with BESTÅ interior fittings.Steady also on uneven floors, thanks to the adjustable feet.For safety reasons this TV bench shall not be hung on the wall.This furniture must be fixed to the wall with the enclosed wall fastener.This TV bench can take a max load of 50 kg on the top.Different wall materials require different types of fixing devices. Use fixing devices suitable for the walls in your home, sold separately.May be completed with STALLARP, STUBBARP or NANNARP legs. This TV bench requires 4 legs and 2 BESTÅ supporting legs.May be completed with SULARP legs. This TV bench requires 2 legs and 2 BESTÅ supporting legs.',
           'Eva Lilja Löwenhielm/K Malmvall/E Lilja Löwenhielm',
           '304.289.01 Transforms a single open cube in the EKET series into a display case.Protects the things you love and minimises the need to dust.Tempered glass is more impact resistant than ordinary glass and the surface is easy to clean.Handle with care! A damaged edge or scratched surface can cause the glass to suddenly crack or break. Avoid bumps from the side - this is where the glass is most vulnerable.',
           '204.415.78 The doors keep your belongings hidden and free from dust.You can choose to mount the door on the right or left side.To be completed with BESTÅ hinges.1 door requires 1 pack of hinges. Sold separately.May be completed with knobs or handles. Sold separately.Drill template for marking of hole positions for handles or knobs is sold separately.',
           'Mia Lagerman', 'Lisa Norinder/Marcus Arvonen',
           'Elizabet Gutierrez',
           '404.728.61 Height adjustable armchair which you can swivel to the desired height.Slim lines, easy to place.You sit comfortably since the chair is adjustable in height.The safety castors have a pressure-sensitive brake mechanism that keeps the chair in place when you stand up, and releases automatically when you sit down.This product has been developed and tested for domestic use.',
           '092.210.16 You sit comfortably thanks to the shaped back and armrests.Suitable for both indoor and outdoor use.',
           'Henrik Preutz/Olle Lundberg', 'Anna Palleschitz',
           'A Fredriksson/J Hultqvist/W Chong',
           '500.583.76 Rattan is a natural material which ages beautifully and develops its own unique character over time.Handwoven; each piece of furniture is unique.Stackable chair; saves space when not in use.Plastic feet; protect the furniture if in contact with a moist surface.May be completed with NORNA chair pad for enhanced seating comfort.',
           'Lycke von Schantz', 'S Holmbäck/U Nordentoft', 'Jooyeon Lee',
           'Lisa Hilland', 'Maria Vinka', 'Marcus Arvonen/Lisa Norinder',
           'C Styrbjörn/M Axelsson',
           'IKEA of Sweden/A Fredriksson/J Hultqvist/W Chong',
           "492.284.74 Suitable for both indoor and outdoor use.You sit comfortably thanks to the chair's shaped back and seat.",
           'Monika Mulder',
           "593.376.70 Height adjustable children's desk chair which you can swivel to the desired height.You sit comfortably since the chair is adjustable in height.May be completed with KOLON floor protector.This product has been tested for domestic use.Recommended for ages 7 – 12 years.",
           'Noboru Nakamura',
           '404.199.82 The leather ages beautifully and acquires a nice patina over time.10 year guarantee. Read about the terms in the guarantee brochure.The safety castors have a pressure-sensitive brake mechanism that keeps the chair in place when you stand up, and releases automatically when you sit down.Seat and backrest are adjustable in height and give you maximum support regardless of your body height.You get good support for your thighs and back since the seat depth is adjustable.You can lean back with perfect balance, as the tilt tension mechanism automatically adjusts the resistance to suit your weight and movements. This chair has been tested for office use and meets the requirements for durability and stability set forth in the following standards: EN 1335 and ANSI/BIFMA x5.1.',
           "301.150.66 You sit comfortably thanks to the chair's shaped back and seat.You can hang the chair on a hook on the wall to save space.You can fold the chair, so it takes less space when you're not using it.The chair is available in different colours – choose your favourite or mix.This chair has been tested for home use and meets the requirements for durability and safety, set forth in the following standards: EN 12520 and EN 1022.",
           '103.310.14 Hook and loop fasteners keep the chair cushion in place.The chair cushion has two identical sides so it can be turned over for even wear.Easy to keep clean since it is machine washable.',
           '402.409.89 This foot-rest helps you sit in a good working position at your desk and reduces strain on your legs, back and neck.It’s easy to tilt and adjust the platform to a comfortable angle just by applying pressure with your foot.Your feet stay in place when the platform is tilted because it has a non-slip textured surface.Rubber feet underneath keep the foot-rest firmly in place on the floor and protect sensitive surfaces.',
           'Andreas Fredriksson', 'IKEA of Sweden/E Strandmark',
           'M Kjelstrup/A Östgaard', 'K Hagberg/M Hagberg/Marcus Arvonen',
           '403.193.98 The high back gives good support for your neck and head.',
           'Lisa Norinder',
           '204.262.76 Solid oak is a hardwearing natural material which gives a warm, natural feeling.Wood is a natural living material, and variations in the grain, colour and texture makes each piece of wood furniture unique.Each piece of furniture is unique as it is handmade.For maximum quality, re-tighten the screws when necessary.',
           'H Preutz/N Karlsson', 'Lars Norinder', 'E Thomasson/P Süssmann',
           'David Wahl/Lisa Norinder', 'IKEA of Sweden/Noboru Nakamura',
           'Anna Efverlund',
           'Lisa Norinder/A Fredriksson/J Hultqvist/W Chong',
           '704.516.35 Easy to keep clean since the cushion can be machine washed.Recommended for ages from 3 years.Children’s armchair frame is sold separately.',
           '003.850.69 For increased stability, re-tighten the screws about two weeks after assembly and when necessary.Complete with FANBYN seat shell.',
           'Ola Wihlborg/Lisa Norinder',
           'J Löfgren/J Pettersson/S Lanneskog/J Marnell',
           'L Hilland/J Karlsson', 'Noboru Nakamura/IKEA of Sweden',
           'Carl Öjerstam/S Lanneskog/J Marnell',
           '003.786.67 You sit comfortably thanks to the shaped back and armrests.Complete with FANBYN chair frame.',
           'Lisa Norinder/David Wahl', 'Mikael Axelsson/IKEA of Sweden',
           'K Hagberg/M Hagberg/Mia Lagerman', 'IKEA of Sweden/Nike Karlsson',
           'Jon Karlsson/John/Jonas/Petrus/Paul/Caroline',
           '104.610.10 The lumbar cushion helps you to sit up straight, which relieves both the spine and lower back.You easily attach the lumbar cushion to your favourite chair with the sewn-on touch-and-close fastening.Fixed cover.',
           '200.130.73 The cushion can be turned over and therefore has two sides for even wear.Suitable for AGEN chair.',
           'Nike Karlsson/Mikael Axelsson', 'Jon Karlsson/David Wahl',
           'Olle Lundberg', 'HAY/A Fredriksson/J Hultqvist/W Chong',
           'K Hagberg/M Hagberg/J Löfgren/J Pettersson',
           'J Fritzdorf/J Feldman/J Hedberg', 'Ehlén Johansson/Ola Wihlborg',
           'Ebba Strandmark/Carina Bengs',
           'David Wahl/John/Jonas/Petrus/Paul/Caroline', 'S Fager/J Karlsson',
           'Chris Martin', 'Ehlén Johansson/K Hagberg/M Hagberg',
           'Marcus Arvonen/Nike Karlsson', 'P Süssmann/J Karlsson',
           'IKEA of Sweden/S Lanneskog/J Marnell',
           'Mia Lagerman/K Hagberg/M Hagberg',
           'John/Jonas/Petrus/Paul/Caroline/Lisa Norinder',
           'Ola Wihlborg/S Lanneskog/J Marnell',
           'Carl Öjerstam/Ola Wihlborg/IKEA of Sweden',
           'IKEA of Sweden/John/Jonas/Petrus/Paul/Caroline/David Wahl',
           'Francis Cayouette/Maja Ganszyniec',
           '804.190.70 The shape of the chair backs are adapted to fit the corners of the table, so you save space when the chairs are pushed up against the table.Table (length 84 cm, width 84 cm, height 75 cm). Chair (width 61 cm, depth 53 cm, height 76 cm, seat height 46 cm, seat width 55 cm, seat depth 39 cm).',
           'Mia Lagerman/Mikael Warnhammar',
           'Maja Ganszyniec/K Hagberg/M Hagberg',
           'IKEA of Sweden/Carl Öjerstam/Mia Lagerman',
           'S Lanneskog/J Marnell/IKEA of Sweden',
           'Carl Öjerstam/Ehlén Johansson',
           'Francis Cayouette/K Hagberg/M Hagberg',
           'Carina Bengs/IKEA of Sweden/E Strandmark',
           '104.710.85 The chair legs are made of solid wood, which is a durable natural material.You sit comfortably thanks to the high back and seat with polyester wadding.For increased stability, re-tighten the screws about two weeks after assembly and when necessary.This chair has been tested for home use and meets the requirements for durability and safety, set forth in the following standards: EN 12520 and EN 1022.',
           'S Lanneskog/J Marnell/Mia Lagerman',
           'Mia Lagerman/S Lanneskog/J Marnell',
           'S Lanneskog/J Marnell/Ola Wihlborg',
           'J Karlsson/N Karlsson/Maja Ganszyniec',
           'Mikael Warnhammar/Mia Lagerman',
           'Maja Ganszyniec/J Karlsson/N Karlsson',
           '704.655.38 You sit comfortably thanks to the restful flexibility of the seat.You sit comfortably thanks to the padded seat.Velvet.The velvet reflects light in a characteristic way which may make the colour appear as if it changes.',
           'Chris Martin/Ola Wihlborg/IKEA of Sweden',
           '904.710.86 The chair legs are made of solid wood, which is a durable natural material.You sit comfortably thanks to the high back and seat with polyester wadding.For increased stability, re-tighten the screws about two weeks after assembly and when necessary.This chair has been tested for home use and meets the requirements for durability and safety, set forth in the following standards: EN 12520 and EN 1022.',
           'K Hagberg/M Hagberg/Francis Cayouette',
           'David Wahl/IKEA of Sweden',
           'IKEA of Sweden/E Strandmark/Carina Bengs',
           '804.046.72 You sit comfortably thanks to the restful flexibility of the seat.You sit comfortably thanks to the padded seat.This chair has been tested for home use and meets the requirements for durability and safety, set forth in the following standards: EN 12520 and EN 1022.',
           'J Karlsson/N Karlsson/Nike Karlsson',
           'Lisa Norinder/John/Jonas/Petrus/Paul/Caroline', 'HAY',
           'Ehlén Johansson/J Löfgren/J Pettersson',
           '803.086.75 This product has been developed and tested for domestic use.To be completed with LEIFARNE or SVENBERTIL seat shell.',
           'Carina Bengs/Ebba Strandmark',
           'Mia Lagerman/IKEA of Sweden/Wiebke Braasch',
           'S Lanneskog/J Marnell', 'K Hagberg/M Hagberg/Maja Ganszyniec',
           'Francis Cayouette/Mia Lagerman',
           'K Hagberg/M Hagberg/John/Jonas/Petrus/Paul/Caroline',
           'Mia Lagerman/Ehlén Johansson',
           'Mikael Warnhammar/A Fredriksson/J Hultqvist/W Chong',
           'S Lanneskog/J Marnell/J Löfgren/J Pettersson',
           'Mia Lagerman/IKEA of Sweden/Chris Martin',
           'David Wahl/IKEA of Sweden/John/Jonas/Petrus/Paul/Caroline',
           'Karl Malmvall/Ehlén Johansson', 'Francis Cayouette/Nike Karlsson',
           'J Löfgren/J Pettersson/Marcus Arvonen',
           'Ola Wihlborg/Ehlén Johansson', 'Mia Lagerman/Francis Cayouette',
           "803.891.48 You sit comfortably thanks to the chair's shaped back and seat.Complete with FANBYN chair frame.",
           'Nike Karlsson/Francis Cayouette', 'IKEA of Sweden/Ola Wihlborg',
           'Johanna Asshoff', 'T Winkel/T Jacobsen',
           '804.334.86 Perfect height for small children. They can easily reach and find things on their own.With the included colourful stickers, you can quickly label the drawers in your own personal way.You can also write with chalk on the sticker to keep track of where you have your things.WARNING! TIPPING HAZARD – Unanchored furniture can tip over. This furniture shall be anchored to the wall with the enclosed safety fitting to prevent it from tipping over.Different wall materials require different types of fixing devices. Use fixing devices suitable for the walls in your home, sold separately.',
           'Studio Copenhagen', 'Eva Schildt', 'S Fager/J Jelinek',
           'Studio Copenhagen/Mia Lagerman',
           '504.224.94 You can position the shelf and clothes rail in two different ways – clothes rail at the top and shelf down below, or both together at the top of the wardrobe.Deep enough to hold standard-sized adult hangers.With the included colourful stickers, you can quickly label the doors in your own personal way.You can also write with chalk on the sticker to keep track of where you have your things.WARNING! TIPPING HAZARD – Unanchored furniture can tip over. This furniture shall be anchored to the wall with the enclosed safety fitting to prevent it from tipping over.Different wall materials require different types of fixing devices. Use fixing devices suitable for the walls in your home, sold separately.',
           'Thomas Sandell',
           '404.662.85 Fits in TROFAST frames.Can be stacked when completed with a lid.May be completed with TROFAST lid.',
           '304.662.81 Fits in TROFAST frames.Can be stacked when completed with a lid.May be completed with TROFAST lid.',
           '504.662.75 Fits in TROFAST frames.Can be stacked when completed with a lid.May be completed with TROFAST lid.',
           'Annie Huldén', 'IKEA of Sweden/Andreas Fredriksson',
           '104.114.40 This waterproof storage box protects your outdoor pads and cushions from rain, sun, dirt, dust and pollen and helps you keep them organised when they’re not being used.Protecting your outdoor pads and cushions in a waterproof storage box is a simple and effective way to make them look new and fresh longer.The colour stays fresh for longer as the fabric is fade resistant. Handles make it easy to move.Make sure the cushions are completely dry before storing them away in the storage box.During the off-season, store the box filled with cushions in a cool dry place indoors.The storage box is light in weight, so you may need to fill it to keep it from moving around in the wind.For example, the storage box can be filled with 3 seat cushions and 3 back cushions for outdoor sofas and 4 decorative cushions.For example, the storage box can be filled with 6 seat/back cushions and 6 decorative cushions.For example, the storage box can be filled with 2 seat cushions and 2 back cushions for outdoor sofas, 2 chair cushions for armchairs and 2-4 decorative cushions.',
           'Eva Lilja Löwenhielm/K Hagberg/M Hagberg/IKEA of Sweden',
           'Eva Lilja Löwenhielm/IKEA of Sweden/K Hagberg/M Hagberg',
           'Jonas Hultqvist/IKEA of Sweden/Eva Lilja Löwenhielm',
           'Marcus Arvonen/Andreas Fredriksson',
           'IKEA of Sweden/Eva Lilja Löwenhielm/K Hagberg/M Hagberg',
           'Magnus Elebäck',
           'Eva Lilja Löwenhielm/IKEA of Sweden/Nike Karlsson',
           'IKEA of Sweden/Jonas Hultqvist',
           'Eva Lilja Löwenhielm/Jonas Hultqvist/IKEA of Sweden',
           'Mikael Axelsson',
           'Eva Lilja Löwenhielm/IKEA of Sweden/Jonas Hultqvist',
           'Eva Lilja Löwenhielm/IKEA of Sweden/David Wahl',
           'Lisel Garsveden',
           'Jon Karlsson/IKEA of Sweden/Eva Lilja Löwenhielm',
           'IKEA of Sweden/David Wahl/Eva Lilja Löwenhielm',
           'Nike Karlsson/IKEA of Sweden',
           'Nike Karlsson/IKEA of Sweden/Eva Lilja Löwenhielm',
           'IKEA of Sweden/Nike Karlsson/Eva Lilja Löwenhielm',
           'Jonas Hultqvist/IKEA of Sweden',
           '104.246.21 KNOPPARP sofa is very durable thanks to the metal construction and strong supporting fabric.Thanks to the innovative construction, we can use less materials and foam when we make KNOPPARP sofa, while the padded cover ensures that the comfort is maintained.A sofa with small, neat dimensions which is easy to furnish with, even when space is limited. This cover is made from KNISA fabric in polyester, which is dope-dyed. It’s a durable material which has a soft feel.The dope-dyeing process reduces consumption of water and dyestuff compared to traditional dyeing techniques.The cover is easy to keep clean as it is removable and can be machine washed.Easy to bring home if you choose to carry it on your own. The packaging is just over one metre in height and weighs 17 kg.10 year guarantee. Read about the terms in the guarantee brochure.This cover’s ability to resist abrasion has been tested to handle 40,000 cycles. 15,000 cycles or more is suitable for furniture used every day at home. Over 30,000 cycles means a good ability to resist abrasion.The cover has a lightfastness level of 5-6 (the ability to resist colour fading) on a scale of 1 to 8. According to industry standards, a lightfastness level of 4 or higher is suitable for home use.',
           '104.691.86 The sofa is packaged in a space-efficient way, making it easy to transport and carry into your home.You can store remote controls and other smaller items in the practical pockets on the sides of the armrests.',
           'IKEA of Sweden/David Wahl/Lisa Hilland', 'Synnöve Mork',
           'Lisa Hilland/IKEA of Sweden/David Wahl',
           'Francis Cayouette/Ehlén Johansson',
           '193.254.57 The cover is easy to keep clean as it is removable and can be machine washed.',
           '304.510.67', 'Ehlén Johansson/Francis Cayouette',
           '992.396.20 The cover is easy to keep clean as it is removable and can be machine washed.This product is an extra cover. Sofa is sold separately.',
           '704.510.65',
           '904.134.78 SEGERSTA cover is made of cotton and polyester and is a very durable fabric, with a woven checkered pattern and a soft, smooth surface.This product is an extra cover. Armchair is sold separately.This cover’s ability to resist abrasion has been tested to handle 50,000 cycles. 15,000 cycles or more is suitable for furniture used every day at home. Over 30,000 cycles means a good ability to resist abrasion.The cover has a lightfastness level of 5 (the ability to resist colour fading) on a scale of 1 to 8. According to industry standards, a lightfastness level of 4 or higher is suitable for home use.',
           "803.826.08 This cover's ability to resist abrasion has been tested to handle 15,000 cycles, which is suitable for furniture that should withstand everyday use in the home.The cover has a lightfastness level of 5 (the ability to resist colour fading) on a scale of 1 to 8. According to industry standards, a lightfastness level of 4 or higher is suitable for home use.",
           "804.289.27 The cover is easy to keep clean as it is removable and can be machine washed.This cover's ability to resist abrasion has been tested to handle 20,000 cycles. A cover that withstands 15,000 cycles or more is suitable for furniture that should withstand everyday use in the home.The cover has a lightfastness level of 5 (the ability to resist colour fading) on a scale of 1 to 8. According to industry standards, a lightfastness level of 4 or higher is suitable for home use.This product is an extra cover. Sofa is sold separately.",
           'Ehlén Johansson/Fredriksson/L Löwenhielm/Hilland',
           '004.135.38 SEGERSTA cover is made of cotton and polyester and is a very durable fabric, with a woven checkered pattern and a soft, smooth surface.This cover’s ability to resist abrasion has been tested to handle 50,000 cycles. 15,000 cycles or more is suitable for furniture used every day at home. Over 30,000 cycles means a good ability to resist abrasion.The cover has a lightfastness level of 5 (the ability to resist colour fading) on a scale of 1 to 8. According to industry standards, a lightfastness level of 4 or higher is suitable for home use.This product is an extra cover. Frame is sold separately.',
           '603.826.28 These legs in nickel-plated steel give NORSBORG sofa a modern and stylish look.',
           '104.709.72 LJUNGEN is a hardwearing cover made of a polyester fabric with a soft, velvety surface and a slightly reflective lustre.This product is an extra cover. Sofa is sold separately.This cover’s ability to resist abrasion has been tested to handle 45,000 cycles. 15,000 cycles or more is suitable for furniture used every day at home. Over 30,000 cycles means a good ability to resist abrasion.The cover has a lightfastness level of 5 (the ability to resist colour fading) on a scale of 1 to 8. According to industry standards, a lightfastness level of 4 or higher is suitable for home use.',
           '992.396.15 The cover is easy to keep clean as it is removable and can be machine washed.This product is an extra cover. Sofa is sold separately.',
           '502.555.36 Easy to fold up and easy to move.Fits on flat sofa armrest width from 10 cm to 30 cm.',
           "003.825.94 This cover's ability to resist abrasion has been tested to handle 15,000 cycles, which is suitable for furniture that should withstand everyday use in the home.The cover has a lightfastness level of 5 (the ability to resist colour fading) on a scale of 1 to 8. According to industry standards, a lightfastness level of 4 or higher is suitable for home use.",
           '792.396.35 The cover is easy to keep clean as it is removable and can be machine washed.This product is an extra cover. Sofa is sold separately.',
           '104.417.34 The cover is easy to keep clean as it is removable and can be machine washed.',
           'IKEA of Sweden/Fredriksson/L Löwenhielm/Hilland/Ehlén Johansson',
           '304.337.90 This cover is made of dope-dyed GUNNARED fabric in polyester. It is a durable fabric with a wool-like feel, a warm look and a two-toned melange effect.The cover is easy to keep clean since it is removable and machine washable.This product is an extra cover. Sofa is sold separately.This cover’s ability to resist abrasion has been tested to handle 50,000 cycles. 15,000 cycles or more is suitable for furniture used every day at home. Over 30,000 cycles means a good ability to resist abrasion.The cover has a lightfastness level of 5 (the ability to resist colour fading) on a scale of 1 to 8. According to industry standards, a lightfastness level of 4 or higher is suitable for home use.',
           'IKEA of Sweden/Francis Cayouette',
           'Francis Cayouette/IKEA of Sweden/Ehlén Johansson',
           '904.762.58 The cover is easy to keep clean since it is removable and machine washable.This product is an extra cover. Sofa is sold separately.',
           '104.417.29 The cover is easy to keep clean as it is removable and can be machine washed.',
           'Nada Debs',
           '602.141.59 You can collect cables and extension leads on the shelf under the table top, so they’re hidden but still close at hand.Can be placed in the middle of a room because the back is finished.You can mount the storage unit to the right or left, according to your space or preference. Combines with other furniture in the MALM series.May be completed with SUMMERA drawer insert with 6 compartments to create order in the drawer.',
           'Johanna Asshoff/IKEA of Sweden', 'IKEA of Sweden/Johanna Asshoff',
           '104.321.93 The tabletop in tempered glass is stain resistant and easy to clean.Handle with care! A damaged edge or scratched surface can cause the glass to suddenly crack or break. Avoid bumps from the side - this is where the glass is most vulnerable.Safety fittings included.',
           '904.238.87 Small, neat dimensions make the table easy to furnish with, even when space is limited.Table with drop-leaves seats 2-4; makes it possible to adjust the table size according to need.You can store for example cutlery, table napkins and candles in the 6 drawers under the table top.This table has been tested against our strictest standards for stability, durability and safety to withstand everyday use in your home for years.Seats 2-4.Only recommended for indoor use.Combines with other furniture in the NORDEN series.For increased stability, re-tighten the screws about two weeks after assembly and when necessary.',
           '602.141.83 The pull-out panel gives you an extra work surface.You can collect cables and extension leads on the shelf under the table top, so they’re hidden but still close at hand.You can mount the pull-out panel to the left or right according to your needs.Can be placed in the middle of a room because the back is finished.Combines with other furniture in the MALM series.',
           '401.483.68 Sizes: 44,5x29,5x40 cm, 40x29,5x43 cm and 33,5x29,5x40 cm.',
           'IKEA of Sweden/Chenyi Ke/Johanna Asshoff',
           '304.156.25 Can be used individually or be pushed together to save space.Low weight; easy to move.Sizes: (length x width x height) 27x22x35 cm, 36x26x38 cm, 45x30x40 cm.',
           '902.179.72 Adjustable feet make the table stand steady also on uneven floors.Screws for fixing the legs to the table top are included.Suitable for table tops with a minimum thickness of 25 mm.',
           'Johanna Jelinek',
           '702.453.39 Can be placed in the middle of a room because the back is finished.The high-gloss surfaces reflect light and give a vibrant look.',
           'Johanna Asshoff/Jomi Evers',
           '104.158.53 Can be used individually or be pushed together to save space.Low weight; easy to move.Sizes: (length x width x height) 27x22x35 cm, 36x26x38 cm, 45x30x40 cm.',
           '202.959.25 Separate shelf for magazines, etc. helps you keep your things organised and the table top clear.The castors make it easy to move the table if needed.',
           '104.238.86 Small, neat dimensions make the table easy to furnish with, even when space is limited.Table with drop-leaves seats 2-4; makes it possible to adjust the table size according to need.You can store for example cutlery, table napkins and candles in the 6 drawers under the table top.This table has been tested against our strictest standards for stability, durability and safety to withstand everyday use in your home for years.Seats 2-4.Only recommended for indoor use.For increased stability, re-tighten the screws about two weeks after assembly and when necessary.Combines with other furniture in the NORDEN series.',
           '003.494.44 Separate shelf for magazines, etc. helps you keep your things organised and the table top clear.The castors make it easy to move the table if needed.',
           '204.002.76 One half of the table top can be raised to a more comfortable height for eating or maybe surfing on your laptop.Practical storage space underneath the table top.You can eat comfortably at your coffee table by raising the table top.',
           'Gustav Carlberg/Johanna Asshoff',
           'IKEA of Sweden/Gustav Carlberg', 'Jomi Evers',
           'Chenyi Ke/IKEA of Sweden', 'IKEA of Sweden/Jomi Evers',
           'Francis Cayouette/Jomi Evers', 'Gustav Carlberg/IKEA of Sweden',
           '404.262.75 Solid wood is a hardwearing natural material.Handmade by skilled craftspeople, which makes every product unique in design and size.For increased stability, re-tighten the screws about two weeks after assembly and when necessary.Seats 4-6.',
           'Jomi Evers/IKEA of Sweden',
           'Johanna Asshoff/IKEA of Sweden/Gustav Carlberg',
           'Johanna Asshoff/Gustav Carlberg', 'Chris Martin/IKEA of Sweden',
           'IKEA of Sweden/Chenyi Ke', 'Wiebke Braasch',
           'Francis Cayouette/Gustav Carlberg', 'Tina Christensen/Jomi Evers',
           '902.952.53 Cable outlets make it easy to lead cables out the back, so they’re hidden from view but close at hand when you need them.The large drawer makes it easy to keep remote controls, game controllers and other TV accessories organised.Behind the doors, there’s plenty of extra storage space to help keep your living room organised.A floor-standing TV bench must be fixed to the wall with the included wall fastener.Different wall materials require different types of fixing devices. Use fixing devices suitable for the walls in your home, sold separately.Push openers included.1 fixed shelf and 3 adjustable shelves included.',
           '102.945.11 It’s easy to keep the cables from your TV and other devices out of sight but close at hand, as there are several cable outlets at the back of the TV bench.You can choose to stand the TV bench on the floor or mount it on the wall to free up floor space.If you want to organise inside you can complement with BESTÅ interior fittings.Steady also on uneven floors, thanks to the adjustable feet.This furniture must be fixed to the wall with the enclosed wall fastener.This TV bench can take a max load of 50 kg on the top.Different wall materials require different types of fixing devices. Use fixing devices suitable for the walls in your home, sold separately.May be completed with STALLARP, STUBBARP or NANNARP legs. This TV bench requires 4 legs and 1 BESTÅ supporting leg.May be completed with SULARP legs. This TV bench requires 2 legs and 1 BESTÅ supporting leg.',
           'IKEA of Sweden/Marcus Arvonen/Carl Öjerstam',
           '402.998.90 It’s easy to keep the cables from your TV and other devices out of sight but close at hand, as there are several cable outlets at the back of the TV bench.If you want to organise inside you can complement with BESTÅ interior fittings.Steady also on uneven floors, thanks to the adjustable feet.For safety reasons this TV bench shall not be hung on the wall.The TV bench must be fixed to the wall with the included wall fastener.This TV bench can take a max load of 50 kg on the top.Different wall materials require different types of fixing devices. Use fixing devices suitable for the walls in your home, sold separately.May be completed with STALLARP, STUBBARP or NANNARP legs. This TV bench requires 4 legs and 1 BESTÅ supporting leg.May be completed with SULARP legs. This TV bench requires 2 legs and 1 BESTÅ supporting leg.',
           'J Karlsson/W Chong',
           'IKEA of Sweden/K Hagberg/M Hagberg/Ehlén Johansson',
           'K Hagberg/M Hagberg/Ehlén Johansson/IKEA of Sweden',
           'IKEA of Sweden/Ehlén Johansson/K Hagberg/M Hagberg',
           'Ehlén Johansson/K Hagberg/M Hagberg/IKEA of Sweden',
           'IKEA of Sweden/Ebba Strandmark/Ehlén Johansson',
           'Ehlén Johansson/IKEA of Sweden/K Hagberg/M Hagberg',
           'IKEA of Sweden/Ehlén Johansson/Ola Wihlborg',
           'Ebba Strandmark/Ehlén Johansson/IKEA of Sweden',
           'Ebba Strandmark/IKEA of Sweden/Ola Wihlborg/Ehlén Johansson',
           'IKEA of Sweden/Ola Wihlborg/Ehlén Johansson',
           'Ehlén Johansson/Ebba Strandmark/IKEA of Sweden',
           'IKEA of Sweden/Ehlén Johansson/Ebba Strandmark',
           'IKEA of Sweden/Ola Wihlborg/Ehlén Johansson/Ebba Strandmark',
           'Ehlén Johansson/Ebba Strandmark/Ola Wihlborg/IKEA of Sweden',
           'IKEA of Sweden/Andreas Fredriksson/Ehlén Johansson',
           'Ehlén Johansson/Ola Wihlborg/IKEA of Sweden',
           'Ola Wihlborg/Ehlén Johansson/IKEA of Sweden',
           'Ehlén Johansson/Andreas Fredriksson/IKEA of Sweden',
           'IKEA of Sweden/Ehlén Johansson/Andreas Fredriksson'], dtype=object)




```python
mask = ~df_base["designer"].str.contains(r'\d', na=False)
df_base = df_base.loc[mask].copy()
```


```python
df_base.loc[:, 'designer'] = df_base['designer'].str.replace(r'K Hagberg\s*/\s*M Hagberg', 'Hagberg family',regex=True)
```


```python
df_base['designer'].unique()
```




    array(['Nicholai Wiig Hansen', 'Francis Cayouette', 'Henrik Preutz',
           'Marcus Arvonen', 'Carina Bengs', 'Hagberg family', 'Sarah Fager',
           'Ehlén Johansson', 'Nike Karlsson', 'Maja Ganszyniec',
           'Karl Malmvall', 'John/Jonas/Petrus/Paul/Caroline',
           'Nike Karlsson/Maja Ganszyniec', 'J Karlsson/N Karlsson',
           'IKEA of Sweden/Karl Malmvall', 'IKEA of Sweden',
           'Nike Karlsson/J Karlsson/N Karlsson', 'Ola Wihlborg',
           'IKEA of Sweden/Tina Christensen', 'IKEA of Sweden/Hagberg family',
           'Ola Wihlborg/IKEA of Sweden', 'IKEA of Sweden/Ebba Strandmark',
           'Hagberg family/IKEA of Sweden', 'Jon Karlsson',
           'IKEA of Sweden/Carina Bengs', 'David Wahl',
           'Jon Karlsson/IKEA of Sweden', 'IKEA of Sweden/Paulin Machado',
           'IKEA of Sweden/Eva Lilja Löwenhielm',
           'IKEA of Sweden/Synnöve Mork/Ola Wihlborg',
           'IKEA of Sweden/David Wahl', 'Ebba Strandmark/IKEA of Sweden',
           'Eva Lilja Löwenhielm', 'IKEA of Sweden/Anna Efverlund',
           'Paulin Machado', 'Jonas Hultqvist', 'Gustav Carlberg',
           'Carl Öjerstam', 'Virgil Abloh', 'Monika Mulder/IKEA of Sweden',
           'T Christensen/K Legaard', 'Tord Björklund',
           'Tom Dixon/IKEA of Sweden',
           'Synnöve Mork/Ola Wihlborg/IKEA of Sweden',
           'Ola Wihlborg/IKEA of Sweden/Synnöve Mork',
           'IKEA of Sweden/L Hilland/Lisa Hilland',
           'IKEA of Sweden/L Hilland', 'Carina Bengs/IKEA of Sweden',
           'Tom Dixon', 'Fredriksson/L Löwenhielm/Hilland',
           'IKEA of Sweden/Tord Björklund',
           'Fredriksson/L Löwenhielm/Hilland/IKEA of Sweden',
           'IKEA of Sweden/Henrik Preutz', 'IKEA of Sweden/Tom Dixon',
           'Malin Unnborn', 'A Huldén/S Dahlman',
           'IKEA of Sweden/Virgil Abloh', 'S Edholm/L Ullenius',
           'IKEA of Sweden/Francis Cayouette/Ehlén Johansson',
           'Francis Cayouette/Ehlén Johansson/IKEA of Sweden',
           'Ebba Strandmark', 'J Asshoff/H Brogård',
           'Lisa Hilland/David Wahl/IKEA of Sweden',
           'Henrik Preutz/IKEA of Sweden', 'C Halskov/H Dalsgaard',
           'Ehlén Johansson/IKEA of Sweden',
           'Ola Wihlborg/Synnöve Mork/IKEA of Sweden',
           'David Wahl/Lisa Hilland/IKEA of Sweden',
           'IKEA of Sweden/Ehlén Johansson/Francis Cayouette',
           'IKEA of Sweden/Fredriksson/L Löwenhielm/Hilland',
           'IKEA of Sweden/Ehlén Johansson',
           'Ehlén Johansson/Francis Cayouette/IKEA of Sweden', 'Chenyi Ke',
           'Johan Kroon', 'Gillis Lundgren', 'IKEA of Sweden/Jon Karlsson',
           'Gillis Lundgren/IKEA of Sweden', 'IKEA of Sweden/Carl Öjerstam',
           'Sarah Fager/IKEA of Sweden', 'Hagberg family/Gillis Lundgren',
           'IKEA of Sweden/Marcus Arvonen', 'Marcus Arvonen/IKEA of Sweden',
           'Gillis Lundgren/Hagberg family', 'Niels Gammelgaard',
           'H Preutz/A Fredriksson', 'IKEA of Sweden/Gillis Lundgren',
           'Carl Öjerstam/IKEA of Sweden', 'Tord Björklund/IKEA of Sweden',
           'Carl Öjerstam/Marcus Arvonen/IKEA of Sweden', 'Mikael Warnhammar',
           'K Malmvall/E Lilja Löwenhielm',
           'Gillis Lundgren/Hagberg family/IKEA of Sweden',
           'J Löfgren/J Pettersson',
           'Gillis Lundgren/IKEA of Sweden/Hagberg family',
           'Charlie Styrbjörn', 'Tina Christensen',
           'Francis Cayouette/Eva Lilja Löwenhielm',
           'IKEA of Sweden/Sarah Fager', 'Francis Cayouette/IKEA of Sweden',
           'Eva Lilja Löwenhielm/IKEA of Sweden',
           'Marcus Arvonen/Carl Öjerstam/IKEA of Sweden',
           'Eva Lilja Löwenhielm/K Malmvall/E Lilja Löwenhielm',
           'Mia Lagerman', 'Lisa Norinder/Marcus Arvonen',
           'Elizabet Gutierrez', 'Henrik Preutz/Olle Lundberg',
           'Anna Palleschitz', 'A Fredriksson/J Hultqvist/W Chong',
           'Lycke von Schantz', 'S Holmbäck/U Nordentoft', 'Jooyeon Lee',
           'Lisa Hilland', 'Maria Vinka', 'Marcus Arvonen/Lisa Norinder',
           'C Styrbjörn/M Axelsson',
           'IKEA of Sweden/A Fredriksson/J Hultqvist/W Chong',
           'Monika Mulder', 'Noboru Nakamura', 'Andreas Fredriksson',
           'IKEA of Sweden/E Strandmark', 'M Kjelstrup/A Östgaard',
           'Hagberg family/Marcus Arvonen', 'Lisa Norinder',
           'H Preutz/N Karlsson', 'Lars Norinder', 'E Thomasson/P Süssmann',
           'David Wahl/Lisa Norinder', 'IKEA of Sweden/Noboru Nakamura',
           'Anna Efverlund',
           'Lisa Norinder/A Fredriksson/J Hultqvist/W Chong',
           'Ola Wihlborg/Lisa Norinder',
           'J Löfgren/J Pettersson/S Lanneskog/J Marnell',
           'L Hilland/J Karlsson', 'Noboru Nakamura/IKEA of Sweden',
           'Carl Öjerstam/S Lanneskog/J Marnell', 'Lisa Norinder/David Wahl',
           'Mikael Axelsson/IKEA of Sweden', 'Hagberg family/Mia Lagerman',
           'IKEA of Sweden/Nike Karlsson',
           'Jon Karlsson/John/Jonas/Petrus/Paul/Caroline',
           'Nike Karlsson/Mikael Axelsson', 'Jon Karlsson/David Wahl',
           'Olle Lundberg', 'HAY/A Fredriksson/J Hultqvist/W Chong',
           'Hagberg family/J Löfgren/J Pettersson',
           'J Fritzdorf/J Feldman/J Hedberg', 'Ehlén Johansson/Ola Wihlborg',
           'Ebba Strandmark/Carina Bengs',
           'David Wahl/John/Jonas/Petrus/Paul/Caroline', 'S Fager/J Karlsson',
           'Chris Martin', 'Ehlén Johansson/Hagberg family',
           'Marcus Arvonen/Nike Karlsson', 'P Süssmann/J Karlsson',
           'IKEA of Sweden/S Lanneskog/J Marnell',
           'Mia Lagerman/Hagberg family',
           'John/Jonas/Petrus/Paul/Caroline/Lisa Norinder',
           'Ola Wihlborg/S Lanneskog/J Marnell',
           'Carl Öjerstam/Ola Wihlborg/IKEA of Sweden',
           'IKEA of Sweden/John/Jonas/Petrus/Paul/Caroline/David Wahl',
           'Francis Cayouette/Maja Ganszyniec',
           'Mia Lagerman/Mikael Warnhammar', 'Maja Ganszyniec/Hagberg family',
           'IKEA of Sweden/Carl Öjerstam/Mia Lagerman',
           'S Lanneskog/J Marnell/IKEA of Sweden',
           'Carl Öjerstam/Ehlén Johansson',
           'Francis Cayouette/Hagberg family',
           'Carina Bengs/IKEA of Sweden/E Strandmark',
           'S Lanneskog/J Marnell/Mia Lagerman',
           'Mia Lagerman/S Lanneskog/J Marnell',
           'S Lanneskog/J Marnell/Ola Wihlborg',
           'J Karlsson/N Karlsson/Maja Ganszyniec',
           'Mikael Warnhammar/Mia Lagerman',
           'Maja Ganszyniec/J Karlsson/N Karlsson',
           'Chris Martin/Ola Wihlborg/IKEA of Sweden',
           'Hagberg family/Francis Cayouette', 'David Wahl/IKEA of Sweden',
           'IKEA of Sweden/E Strandmark/Carina Bengs',
           'J Karlsson/N Karlsson/Nike Karlsson',
           'Lisa Norinder/John/Jonas/Petrus/Paul/Caroline', 'HAY',
           'Ehlén Johansson/J Löfgren/J Pettersson',
           'Carina Bengs/Ebba Strandmark',
           'Mia Lagerman/IKEA of Sweden/Wiebke Braasch',
           'S Lanneskog/J Marnell', 'Hagberg family/Maja Ganszyniec',
           'Francis Cayouette/Mia Lagerman',
           'Hagberg family/John/Jonas/Petrus/Paul/Caroline',
           'Mia Lagerman/Ehlén Johansson',
           'Mikael Warnhammar/A Fredriksson/J Hultqvist/W Chong',
           'S Lanneskog/J Marnell/J Löfgren/J Pettersson',
           'Mia Lagerman/IKEA of Sweden/Chris Martin',
           'David Wahl/IKEA of Sweden/John/Jonas/Petrus/Paul/Caroline',
           'Karl Malmvall/Ehlén Johansson', 'Francis Cayouette/Nike Karlsson',
           'J Löfgren/J Pettersson/Marcus Arvonen',
           'Ola Wihlborg/Ehlén Johansson', 'Mia Lagerman/Francis Cayouette',
           'Nike Karlsson/Francis Cayouette', 'IKEA of Sweden/Ola Wihlborg',
           'Johanna Asshoff', 'T Winkel/T Jacobsen', 'Studio Copenhagen',
           'Eva Schildt', 'S Fager/J Jelinek',
           'Studio Copenhagen/Mia Lagerman', 'Thomas Sandell', 'Annie Huldén',
           'IKEA of Sweden/Andreas Fredriksson',
           'Eva Lilja Löwenhielm/Hagberg family/IKEA of Sweden',
           'Eva Lilja Löwenhielm/IKEA of Sweden/Hagberg family',
           'Jonas Hultqvist/IKEA of Sweden/Eva Lilja Löwenhielm',
           'Marcus Arvonen/Andreas Fredriksson',
           'IKEA of Sweden/Eva Lilja Löwenhielm/Hagberg family',
           'Magnus Elebäck',
           'Eva Lilja Löwenhielm/IKEA of Sweden/Nike Karlsson',
           'IKEA of Sweden/Jonas Hultqvist',
           'Eva Lilja Löwenhielm/Jonas Hultqvist/IKEA of Sweden',
           'Mikael Axelsson',
           'Eva Lilja Löwenhielm/IKEA of Sweden/Jonas Hultqvist',
           'Eva Lilja Löwenhielm/IKEA of Sweden/David Wahl',
           'Lisel Garsveden',
           'Jon Karlsson/IKEA of Sweden/Eva Lilja Löwenhielm',
           'IKEA of Sweden/David Wahl/Eva Lilja Löwenhielm',
           'Nike Karlsson/IKEA of Sweden',
           'Nike Karlsson/IKEA of Sweden/Eva Lilja Löwenhielm',
           'IKEA of Sweden/Nike Karlsson/Eva Lilja Löwenhielm',
           'Jonas Hultqvist/IKEA of Sweden',
           'IKEA of Sweden/David Wahl/Lisa Hilland', 'Synnöve Mork',
           'Lisa Hilland/IKEA of Sweden/David Wahl',
           'Francis Cayouette/Ehlén Johansson',
           'Ehlén Johansson/Francis Cayouette',
           'Ehlén Johansson/Fredriksson/L Löwenhielm/Hilland',
           'IKEA of Sweden/Fredriksson/L Löwenhielm/Hilland/Ehlén Johansson',
           'IKEA of Sweden/Francis Cayouette',
           'Francis Cayouette/IKEA of Sweden/Ehlén Johansson', 'Nada Debs',
           'Johanna Asshoff/IKEA of Sweden', 'IKEA of Sweden/Johanna Asshoff',
           'IKEA of Sweden/Chenyi Ke/Johanna Asshoff', 'Johanna Jelinek',
           'Johanna Asshoff/Jomi Evers', 'Gustav Carlberg/Johanna Asshoff',
           'IKEA of Sweden/Gustav Carlberg', 'Jomi Evers',
           'Chenyi Ke/IKEA of Sweden', 'IKEA of Sweden/Jomi Evers',
           'Francis Cayouette/Jomi Evers', 'Gustav Carlberg/IKEA of Sweden',
           'Jomi Evers/IKEA of Sweden',
           'Johanna Asshoff/IKEA of Sweden/Gustav Carlberg',
           'Johanna Asshoff/Gustav Carlberg', 'Chris Martin/IKEA of Sweden',
           'IKEA of Sweden/Chenyi Ke', 'Wiebke Braasch',
           'Francis Cayouette/Gustav Carlberg', 'Tina Christensen/Jomi Evers',
           'IKEA of Sweden/Marcus Arvonen/Carl Öjerstam',
           'J Karlsson/W Chong',
           'IKEA of Sweden/Hagberg family/Ehlén Johansson',
           'Hagberg family/Ehlén Johansson/IKEA of Sweden',
           'IKEA of Sweden/Ehlén Johansson/Hagberg family',
           'Ehlén Johansson/Hagberg family/IKEA of Sweden',
           'IKEA of Sweden/Ebba Strandmark/Ehlén Johansson',
           'Ehlén Johansson/IKEA of Sweden/Hagberg family',
           'IKEA of Sweden/Ehlén Johansson/Ola Wihlborg',
           'Ebba Strandmark/Ehlén Johansson/IKEA of Sweden',
           'Ebba Strandmark/IKEA of Sweden/Ola Wihlborg/Ehlén Johansson',
           'IKEA of Sweden/Ola Wihlborg/Ehlén Johansson',
           'Ehlén Johansson/Ebba Strandmark/IKEA of Sweden',
           'IKEA of Sweden/Ehlén Johansson/Ebba Strandmark',
           'IKEA of Sweden/Ola Wihlborg/Ehlén Johansson/Ebba Strandmark',
           'Ehlén Johansson/Ebba Strandmark/Ola Wihlborg/IKEA of Sweden',
           'IKEA of Sweden/Andreas Fredriksson/Ehlén Johansson',
           'Ehlén Johansson/Ola Wihlborg/IKEA of Sweden',
           'Ola Wihlborg/Ehlén Johansson/IKEA of Sweden',
           'Ehlén Johansson/Andreas Fredriksson/IKEA of Sweden',
           'IKEA of Sweden/Ehlén Johansson/Andreas Fredriksson'], dtype=object)




```python
def clean_designer_final(text):
    if pd.isna(text):
        return "Unknown"
    parts = re.split(r'[/,]| \band\b ', text)
    
    parts = [p.strip() for p in parts if p.strip()]
    
    if len(parts) > 1 and "IKEA of Sweden" in parts:
        parts.remove("IKEA of Sweden")

    if not parts:
        return "IKEA of Sweden"
    
    if parts[0] == "John":
        return "IKEA Team"
        
    return parts[0]

df_base['designer_cleaned'] = df_base['designer'].apply(clean_designer_final)

```


```python
print(df_base['designer_cleaned'].value_counts().head(15).to_markdown())  
```

    | designer_cleaned     |   count |
    |:---------------------|--------:|
    | IKEA of Sweden       |     683 |
    | Ehlén Johansson      |     310 |
    | Ola Wihlborg         |     174 |
    | Francis Cayouette    |     168 |
    | Jon Karlsson         |     155 |
    | Hagberg family       |     134 |
    | Ebba Strandmark      |      83 |
    | Henrik Preutz        |      78 |
    | Eva Lilja Löwenhielm |      75 |
    | Carina Bengs         |      71 |
    | K Malmvall           |      55 |
    | Nike Karlsson        |      55 |
    | Marcus Arvonen       |      52 |
    | Tord Björklund       |      46 |
    | David Wahl           |      46 |
    


```python
sorted_designers= sorted(df_base['designer_cleaned'].unique().astype(str))
for name in sorted_designers:
    print(name)  
```

    A Fredriksson
    A Huldén
    Andreas Fredriksson
    Anna Efverlund
    Anna Palleschitz
    Annie Huldén
    C Halskov
    C Styrbjörn
    Carina Bengs
    Carl Öjerstam
    Charlie Styrbjörn
    Chenyi Ke
    Chris Martin
    David Wahl
    E Strandmark
    E Thomasson
    Ebba Strandmark
    Ehlén Johansson
    Elizabet Gutierrez
    Eva Lilja Löwenhielm
    Eva Schildt
    Francis Cayouette
    Fredriksson
    Gillis Lundgren
    Gustav Carlberg
    H Preutz
    HAY
    Hagberg family
    Henrik Preutz
    IKEA Team
    IKEA of Sweden
    J Asshoff
    J Fritzdorf
    J Karlsson
    J Löfgren
    Johan Kroon
    Johanna Asshoff
    Johanna Jelinek
    Jomi Evers
    Jon Karlsson
    Jonas Hultqvist
    Jooyeon Lee
    K Malmvall
    Karl Malmvall
    L Hilland
    Lars Norinder
    Lisa Hilland
    Lisa Norinder
    Lisel Garsveden
    Lycke von Schantz
    M Kjelstrup
    Magnus Elebäck
    Maja Ganszyniec
    Malin Unnborn
    Marcus Arvonen
    Maria Vinka
    Mia Lagerman
    Mikael Axelsson
    Mikael Warnhammar
    Monika Mulder
    Nada Debs
    Nicholai Wiig Hansen
    Niels Gammelgaard
    Nike Karlsson
    Noboru Nakamura
    Ola Wihlborg
    Olle Lundberg
    P Süssmann
    Paulin Machado
    S Edholm
    S Fager
    S Holmbäck
    S Lanneskog
    Sarah Fager
    Studio Copenhagen
    Synnöve Mork
    T Christensen
    T Winkel
    Thomas Sandell
    Tina Christensen
    Tom Dixon
    Tord Björklund
    Virgil Abloh
    Wiebke Braasch
    


```python
final_name_fixer = {
    'A Fredriksson': 'Andreas Fredriksson',
    'A Huldén': 'Annie Huldén',
    'C Styrbjörn': 'Charlie Styrbjörn',
    'E Strandmark': 'Ebba Strandmark',
    'E Lilja Löwenhielm': 'Eva Lilja Löwenhielm',                                                   
    'Fredriksson': 'Andreas Fredriksson',
    'Hilland': 'Lisa Hilland',
    'H Preutz': 'Henrik Preutz',
    'J Asshoff': 'Johanna Asshoff',
    'J Hultqvist': 'Jonas Hultqvist',
    'J Jelinek': 'Johanna Jelinek',
    'J Karlsson': 'Jon Karlsson',
    'K Malmvall': 'Karl Malmvall',
    'L Löwenhielm': 'Eva Lilja Löwenhielm',
    'L Hilland': 'Lisa Hilland',
    'M Axelsson': 'Mikael Axelsson',
    'S Fager': 'Sarah Fager',
    'T Christensen': 'Tina Christensen',
    'N Karlsson': 'Nike Karlsson'
}


df_base['designer_final'] = df_base['designer_cleaned'].map(lambda x: final_name_fixer.get(x, x))

```


```python
sort_des_rep = sorted(df_base['designer_final'].unique().astype(str))
for name in sort_des_rep:
    print(name)
```

    Andreas Fredriksson
    Anna Efverlund
    Anna Palleschitz
    Annie Huldén
    C Halskov
    Carina Bengs
    Carl Öjerstam
    Charlie Styrbjörn
    Chenyi Ke
    Chris Martin
    David Wahl
    E Thomasson
    Ebba Strandmark
    Ehlén Johansson
    Elizabet Gutierrez
    Eva Lilja Löwenhielm
    Eva Schildt
    Francis Cayouette
    Gillis Lundgren
    Gustav Carlberg
    HAY
    Hagberg family
    Henrik Preutz
    IKEA Team
    IKEA of Sweden
    J Fritzdorf
    J Löfgren
    Johan Kroon
    Johanna Asshoff
    Johanna Jelinek
    Jomi Evers
    Jon Karlsson
    Jonas Hultqvist
    Jooyeon Lee
    Karl Malmvall
    Lars Norinder
    Lisa Hilland
    Lisa Norinder
    Lisel Garsveden
    Lycke von Schantz
    M Kjelstrup
    Magnus Elebäck
    Maja Ganszyniec
    Malin Unnborn
    Marcus Arvonen
    Maria Vinka
    Mia Lagerman
    Mikael Axelsson
    Mikael Warnhammar
    Monika Mulder
    Nada Debs
    Nicholai Wiig Hansen
    Niels Gammelgaard
    Nike Karlsson
    Noboru Nakamura
    Ola Wihlborg
    Olle Lundberg
    P Süssmann
    Paulin Machado
    S Edholm
    S Holmbäck
    S Lanneskog
    Sarah Fager
    Studio Copenhagen
    Synnöve Mork
    T Winkel
    Thomas Sandell
    Tina Christensen
    Tom Dixon
    Tord Björklund
    Virgil Abloh
    Wiebke Braasch
    


```python
print(df_base['designer_final'].value_counts().head(15).to_markdown())
```

    | designer_final       |   count |
    |:---------------------|--------:|
    | IKEA of Sweden       |     683 |
    | Ehlén Johansson      |     310 |
    | Ola Wihlborg         |     174 |
    | Jon Karlsson         |     169 |
    | Francis Cayouette    |     168 |
    | Hagberg family       |     134 |
    | Henrik Preutz        |     110 |
    | Ebba Strandmark      |      87 |
    | Eva Lilja Löwenhielm |      75 |
    | Carina Bengs         |      71 |
    | Karl Malmvall        |      60 |
    | Nike Karlsson        |      55 |
    | Marcus Arvonen       |      52 |
    | Andreas Fredriksson  |      51 |
    | Tord Björklund       |      46 |
    


```python
plt.figure(figsize=(14, 6))
top_designer = df_base['designer_final'].value_counts().nlargest(10).reset_index()
top_designer.columns = ['designer_final', 'product_count']
barplot = sns.barplot(data=top_designer, x='designer_final', y='product_count', edgecolor='black')
for i in barplot.containers:
    barplot.bar_label(i)
plt.xticks(rotation=45, ha='right')
plt.title("Топ-10 дизайнерів за кількістю продуктів")
plt.xlabel("Ім'я дизайнера")
plt.ylabel("Кількість товарів")
plt.show()
```


    
![png](output_50_0.png)
    


Найбільший внесок в асортимент робить команда IKEA of Sweden (683 унікальні позиції). Серед індивідуальних авторів найпродуктивнішими є Ehlén Johansson та Ola Wihlborg.


```python
plt.figure(figsize=(15, 6))
designer_price = df_base.groupby('designer_final')['price'].median().nlargest(20).reset_index()
designer_price_plot = sns.barplot(data=designer_price, x='designer_final', y='price', edgecolor='black')
for i in designer_price_plot.containers:
    designer_price_plot.bar_label(i)
plt.title('Середня ціна за дизайнером')
plt.xlabel('Дизайнер')
plt.ylabel('Середня ціна')
plt.xticks(rotation=45, ha='right')
plt.show()
```


    
![png](output_52_0.png)
    


Візуалізація підтвердила значну варіативність цін залежно від авторства. Виявлено "преміальний" сегмент дизайнерів (наприклад, S Lanneskog із середньою ціною > 4000 та Niels Gammelgaard > 3000), що робить колонку дизайнера критично важливою для прогнозування вартості.


```python
df_base['is_ikea_design'] = df_base['designer'].apply(
    lambda x: 'IKEA of Sweden' if 'IKEA of Sweden' in str(x) else 'External Designer'
)
plt.figure(figsize=(16, 6)) 

plt.subplot(1, 2, 1) 
top_category_1 = 'Wardrobes'
category_df_1 = df_base[df_base['category'] == top_category_1]
sns.violinplot(x='is_ikea_design', y='price_log', data=category_df_1, inner="quart")
plt.title(f'Розподіл цін у категорії "{top_category_1}"')


plt.subplot(1, 2, 2) 
top_category_2 = 'Sofas & armchairs'
category_df_2 = df_base[df_base['category'] == top_category_2]
sns.violinplot(x='is_ikea_design', y='price_log', data=category_df_2, inner="quart")
plt.title(f'Розподіл цін у категорії "{top_category_2}"')

plt.show() 
```


    
![png](output_54_0.png)
    



```python
print(df_base[['width', 'height', 'depth']].isnull().sum().to_markdown())   
```

    |        |    0 |
    |:-------|-----:|
    | width  |  414 |
    | height |  695 |
    | depth  | 1062 |
    


```python
for col in ['depth', 'height', 'width']:
    df_base[col] = df_base[col].fillna(df_base.groupby('category')[col].transform('median'))
```


```python
print(df_base[['width', 'height', 'depth']].isnull().sum().to_markdown()) 
```

    |        |   0 |
    |:-------|----:|
    | width  |   0 |
    | height |   0 |
    | depth  |   0 |
    


```python
df_base['volume'] = df_base['width'] * df_base['height'] * df_base['depth']
```


```python
plt.figure(figsize=(10, 6))
correlation_spearman = df_base[['price', 'depth', 'height', 'width', 'volume']].corr(method='spearman')
mask = np.triu(np.ones_like(correlation_spearman, dtype=bool))
sns.heatmap(correlation_spearman, annot=True, mask=mask, cmap='Blues', vmin=-1, vmax=1)
plt.title('Кореляція Спірмена: ціна та розмір')
```




    Text(0.5, 1.0, 'Кореляція Спірмена: ціна та розмір')




    
![png](output_59_1.png)
    


Кореляційний аналіз підтвердив, що габарити виробу є значущими предикторами ціни. Найвищий рівень кореляції спостерігається між шириною та вартістю 
(r = 0.63). Низька взаємозалежність між самими габаритами (від -0.046 до 0.37) 
свідчить про відсутність мультиколінеарності, що дозволяє використовувати всі три виміри як незалежні ознаки в моделі машинного навчання.

Гіпотеза №1: Вплив дизайнера на ціну
Формулювання: Дизайнер суттєво впливає на ціноутворення продукту.
Нульова гіпотеза: Медіанні ціни на товари різних дизайнерів однакові (дизайнер не впливає на ціну).
Альтернативна гіпотеза: Існує статистично значуща різниця в цінах між різними дизайнерами.

Гіпотеза №2: Зв'язок ширини та ціни
Формулювання: Ціна продукту прямо залежить від його загального об'єму (V = W * H * D).
Нульова гіпотеза: Між об'ємом товару та його ціною немає статистично значущого зв'язку (кореляція дорівнює 0).
Альтернативна гіпотеза: Існує сильна позитивна кореляція між об'ємом та ціною.

Перед перевіркою гіпотез перевіряємо розподіл даних в дата сеті.


```python
stats.probplot(df_base['price'], dist="norm", plot=plt)
plt.title("Q-Q plot Price")
plt.show()
```


    
![png](output_62_0.png)
    



```python
stat, p = stats.shapiro(df_base['price'])

print(f'Статистика тесту: {stat:.4f}')
print(f'p-value: {p:.4e}')

alpha = 0.05
if p > alpha:
    print("Дані розподілені нормально.")
else:
    print("Дані НЕ розподілені нормально (це очікувано для цін).")
```

    Статистика тесту: 0.7385
    p-value: 2.8120e-55
    Дані НЕ розподілені нормально (це очікувано для цін).
    

Я провела тест Шапіро-Вілка та подивилася на QQ-plot, щоб перевірити ціни на нормальність. Отримала p < 0.05, отже, дані розподілені ненормально. Оскільки припущення про нормальність порушено, я не можу використовувати звичайний ANOVA. 


```python
designer_counts = df_base['designer_final'].value_counts()
cumulative_share = designer_counts.cumsum() / designer_counts.sum()
top_designers = cumulative_share[cumulative_share <= 0.50].index

df_base['designer_hypothesis'] = df_base['designer_final'].apply(   
    lambda x: x if x in top_designers else 'Other'                                                                 ########"Поверни x (саме ім'я), ЯКЩО цей x є у списку топів, ІНАКШЕ поверни слово 'Other'
)
```


```python
plt.figure(figsize=(8, 5))

sns.kdeplot(data=df_base[df_base['designer_hypothesis'] != 'Other'], x='price_log', 
            fill=True, label='Top-50% Designers', color='gold', alpha=0.6)
sns.kdeplot(data=df_base[df_base['designer_hypothesis'] == 'Other'], x='price_log', 
            fill=True, label='Other Designers', color='skyblue', alpha=0.4)

plt.title('Чи дорожчі Топ-дизайнери? (Розподіл логарифмованих цін)')
plt.xlabel('Log(Price)')
plt.ylabel('Щільність')
plt.legend()
plt.show()
```


    
![png](output_66_0.png)
    


Візуальний аналіз щільності розподілу (KDE) підтверджує, що цінові стратегії топ-дизайнерів та інших авторів відрізняються. Перевіромо це ще раз за допомогою тесту Мана -Вітні.


```python
top_group_prices = df_base[df_base['designer_hypothesis'] != 'Other']['price']
other_group_prices = df_base[df_base['designer_hypothesis'] == 'Other']['price']

stat, p_val = stats.mannwhitneyu(top_group_prices, other_group_prices)

print(f"P-value: {p_val:.4e}")
if p_val < 0.05:
    print("Висновок: Різниця значуща. Топ-дизайнери мають іншу цінову політику.")
else:
    print("Висновок: Різниця випадкова. Ім'я дизайнера не впливає на ціну так сильно.")
```

    P-value: 3.6446e-01
    Висновок: Різниця випадкова. Ім'я дизайнера не впливає на ціну так сильно.
    

Статистичний тест (Mann-Whitney U) показав p-value = 0.364, що значно > за поріг 0.05. Це означає, що ми не можемо відкинути нульову гіпотезу. Статистично значущої різниці між цінами "Топ-дизайнерів" та іншими авторами не виявлено. Оскільки імена дизайнерів не мають прямого впливу на ціноутворення, цей фактор не буде пріоритетним для нашої моделі ML. Переходимо до перевірки Гіпотези №2.


```python
corr_spearman, p_val_spearman = stats.spearmanr(df_base['volume'], df_base['price'])
print(f"Correlation coefficient: {corr_spearman:.4f}")
print(f"P-value: {p_val_spearman:.4e}")
if p_val_spearman < 0.05:
   
    strength = "strong" if corr_spearman > 0.5 else "moderate"
    print(f"Conclusion: Relationship confirmed. A coefficient of {corr_spearman:.2f} indicates a {strength} positive correlation.")
else:
    print("Conclusion: No significant relationship between volume and price was found.")
```

    Correlation coefficient: 0.6669
    P-value: 0.0000e+00
    Conclusion: Relationship confirmed. A coefficient of 0.67 indicates a strong positive correlation.
    

Коефіцієнт кореляції Спірмена 0.67 та p-value < 0.05 підтверджують наявність сильного позитивного зв'язку між об'ємом товару та його ціною.


```python
plt.figure(figsize=(8, 4))

sns.regplot(data=df_base, x='volume', y='price', 
            scatter_kws={'alpha':0.3, 'color':'teal'}, 
            line_kws={'color':'red'})

plt.xscale('log')
plt.yscale('log')

plt.title('Візуалізація Гіпотези №2: Зв\'язок Об\'єму та Ціни')
plt.xlabel('Об\'єм у логарифмічній шкалі')
plt.ylabel('Ціна у логарифмічній шкалі')

plt.show()
```


    
![png](output_72_0.png)
    


Візуалізація (Scatter Plot) продемонструвала чітку тенденцію: зі збільшенням фізичних габаритів товару його вартість зростає. Хоча існують винятки (дороговартісні малогабаритні товари), загальний об'єм є ключовим ціноутворюючим фактором для асортименту IKEA.

# Data Preparation


```python
ml_data = df_base.copy()
```


```python
ml_data.columns
```




    Index(['name', 'category', 'price', 'old_price', 'other_colors',
           'short_description', 'designer', 'depth', 'height', 'width',
           'price_log', 'designer_cleaned', 'designer_final', 'is_ikea_design',
           'volume', 'designer_hypothesis'],
          dtype='object')




```python
ml_data.drop(columns=['name','old_price', 'short_description', 'price_log', 'designer', 'designer_cleaned', 
                      'designer_final', 'designer_hypothesis'], inplace=True)
```


```python
ml_data.columns
```




    Index(['category', 'price', 'other_colors', 'depth', 'height', 'width',
           'is_ikea_design', 'volume'],
          dtype='object')




```python
features = ['category', 'depth', 'height', 'width', 'volume', 'other_colors']
X = ml_data[features]
y = ml_data['price']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Тренувальні дані: {X_train.shape}")
print(f"Тестові дані: {X_test.shape}")
```

    Тренувальні дані: (2288, 6)
    Тестові дані: (572, 6)
    


```python
numeric_features = ['width', 'height', 'depth', 'volume']
categorical_features = ['category', 'other_colors']

## Створюємо "рецепт" обробки для чисел
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), ## Якщо в даних пропуски програма автоматично підставить туди медіану по цьому стовпцю.
    ('scaler', StandardScaler())       ##  Це масштабування. Моделі важко працювати, коли одне число — 5000 (ціна), а інше — 0.5 (глибина). Скалер приводить їх до одного масштабу (зазвичай від -3 до 3), щоб великі числа не "тиснули" на модель сильніше за малі.
          
])

 ## Pipeline для категорій: заповнюємо пропуски модою + TargetEncoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), ## Пропуски в категоріях (наприклад, не вказано колір) заповнюються модою (найбільш популярним значенням).
    ('encoder', TargetEncoder(target_type='continuous')) ## Замість того, щоб просто присвоїти категорії числа (1, 2, 3), цей енкодер замінює назву категорії на середнє значення ціни для цієї категорії. Наприклад, якщо категорія "Дивани" зазвичай дорога, вона отримає високе числове значення. Це допомагає моделі краще зрозуміти зв'язок між категорією та ціною.
])


col_prepr = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

col_prepr
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.estimator-table summary {
    padding: .5rem;
    font-family: monospace;
    cursor: pointer;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover {
    background-color: #e0e0e0;
}

.estimator-table table td {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left;
}

.user-set td.value pre {
    color:rgb(255, 94, 0) !important;
    background-color: transparent !important;
}

.default td {
    color: black;
    text-align: left;
}

.user-set td i,
.default td i {
    color: black;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}
</style><body><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;median&#x27;)),
                                                 (&#x27;scaler&#x27;, StandardScaler())]),
                                 [&#x27;width&#x27;, &#x27;height&#x27;, &#x27;depth&#x27;, &#x27;volume&#x27;]),
                                (&#x27;cat&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;encoder&#x27;,
                                                  TargetEncoder(target_type=&#x27;continuous&#x27;))]),
                                 [&#x27;category&#x27;, &#x27;other_colors&#x27;])])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>ColumnTransformer</div></div><div><a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for ColumnTransformer</span></a><span class="sk-estimator-doc-link ">i<span>Not fitted</span></span></div></label><div class="sk-toggleable__content " data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('transformers',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">transformers&nbsp;</td>
            <td class="value">[(&#x27;num&#x27;, ...), (&#x27;cat&#x27;, ...)]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('remainder',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">remainder&nbsp;</td>
            <td class="value">&#x27;drop&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('sparse_threshold',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">sparse_threshold&nbsp;</td>
            <td class="value">0.3</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_jobs',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">n_jobs&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('transformer_weights',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">transformer_weights&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">verbose&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose_feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">verbose_feature_names_out&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('force_int_remainder_cols',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">force_int_remainder_cols&nbsp;</td>
            <td class="value">&#x27;deprecated&#x27;</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>num</div></div></label><div class="sk-toggleable__content " data-param-prefix="num__"><pre>[&#x27;width&#x27;, &#x27;height&#x27;, &#x27;depth&#x27;, &#x27;volume&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content " data-param-prefix="num__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">missing_values&nbsp;</td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">strategy&nbsp;</td>
            <td class="value">&#x27;median&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">fill_value&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">copy&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">add_indicator&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">keep_empty_features&nbsp;</td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content " data-param-prefix="num__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">copy&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">with_mean&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">with_std&nbsp;</td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>cat</div></div></label><div class="sk-toggleable__content " data-param-prefix="cat__"><pre>[&#x27;category&#x27;, &#x27;other_colors&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content " data-param-prefix="cat__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">missing_values&nbsp;</td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">strategy&nbsp;</td>
            <td class="value">&#x27;most_frequent&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">fill_value&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">copy&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">add_indicator&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">keep_empty_features&nbsp;</td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>TargetEncoder</div></div><div><a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.TargetEncoder.html">?<span>Documentation for TargetEncoder</span></a></div></label><div class="sk-toggleable__content " data-param-prefix="cat__encoder__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">categories&nbsp;</td>
            <td class="value">&#x27;auto&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('target_type',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">target_type&nbsp;</td>
            <td class="value">&#x27;continuous&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('smooth',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">smooth&nbsp;</td>
            <td class="value">&#x27;auto&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('cv',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">cv&nbsp;</td>
            <td class="value">5</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('shuffle',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">shuffle&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('random_state',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">random_state&nbsp;</td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.fa-regular.fa-copy').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling.textContent.trim();
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});
</script></body>




```python
def getBestRegressor(X_train, X_test, Y_train, Y_test, col_prepr):
   
    models = [
        LinearRegression(),
        LassoCV(),
        RidgeCV(),
        SVR(kernel="linear"), 
        KNeighborsRegressor(n_neighbors=16),
        DecisionTreeRegressor(max_depth=10, random_state=42),
        RandomForestRegressor(random_state=42),
        GradientBoostingRegressor(),
        BaggingRegressor(random_state=42),
        HistGradientBoostingRegressor(),
        XGBRegressor(objective="reg:squarederror", random_state=42)
    ]

    results_list = [] 

    for model in models:
        
        model_pipeline = Pipeline(steps=[
            ('col_prepr', col_prepr),
            ('to_dense', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x)),
            ('model', model)
        ])

        model_pipeline.fit(X_train, Y_train)
        
        # Прогноз
        y_pred = model_pipeline.predict(X_test)


        results_list.append({
            "Model": model.__class__.__name__,
            "R^2": r2_score(Y_test, y_pred),
            "MAE": mean_absolute_error(Y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(Y_test, y_pred))
        })

    test_models_df = pd.DataFrame(results_list)
    test_models_df = test_models_df.sort_values(by="R^2", ascending=False).set_index("Model")

    return {
        "report": test_models_df,
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test
    }

results = getBestRegressor(X_train, X_test, y_train, y_test, col_prepr)
print(results["report"])

```

                                        R^2         MAE         RMSE
    Model                                                           
    XGBRegressor                   0.793820  404.800976   673.464649
    RandomForestRegressor          0.780689  398.806456   694.578108
    BaggingRegressor               0.774861  421.244279   703.747769
    HistGradientBoostingRegressor  0.774503  402.842947   704.306142
    GradientBoostingRegressor      0.708211  472.292599   801.172256
    DecisionTreeRegressor          0.689683  480.757665   826.216737
    RidgeCV                        0.473105  651.997611  1076.597405
    LinearRegression               0.473098  651.770824  1076.604103
    SVR                            0.383477  656.615751  1164.570448
    LassoCV                        0.366386  747.103313  1180.602370
    KNeighborsRegressor            0.246652  769.935227  1287.327512
    


```python
df_report = results['report']

plt.figure(figsize=(8, 4))

df_report['R^2'].sort_values().plot(kind='barh', color='skyblue')

plt.title('Яка модель найкраще прогнозує ціну? (Порівняння R²)')
plt.xlabel('Точність (R²)')
plt.ylabel('Назва моделі')
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```


    
![png](output_83_0.png)
    



```python
final_pipe = Pipeline(steps=[
    ('col_prepr', col_prepr),
    ('model', RandomForestRegressor(random_state=42))
])

param_grid = {
    'model__n_estimators': [100, 300],                                               # Кількість дерев
    'model__max_depth': [10, 20, None],                                              # Обмежуємо глибину (як просив ментор)
    'model__min_samples_leaf': [1, 2, 4],                                            # Мінімальна кількість об'єктів у листку
    'model__max_features': ['sqrt', 'log2']                                          # Використовуємо всі ознаки або корінь з них
}

grid_search = GridSearchCV(final_pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Найкращі параметри: {grid_search.best_params_}")
print(f"Найкращий R^2 після тюнінгу: {grid_search.best_score_:.4f}")

best_model_upg = grid_search.best_estimator_
```

    Найкращі параметри: {'model__max_depth': None, 'model__max_features': 'log2', 'model__min_samples_leaf': 1, 'model__n_estimators': 300}
    Найкращий R^2 після тюнінгу: 0.7619
    


```python
final_scores = cross_val_score(best_model_upg, X_train, y_train, cv=5, scoring='r2')
print(f"Результати крос-валідації: {final_scores}")
print(f"Середній R^2: {final_scores.mean():.4f}")
print(f"Відхилення (стабільність): {final_scores.std():.4f}")
```

    Результати крос-валідації: [0.77986204 0.76056837 0.74011319 0.77056404 0.74196062]
    Середній R^2: 0.7586
    Відхилення (стабільність): 0.0156
    


```python
best_model = Pipeline(steps=[
    ('col_prepr', col_prepr),
    ('model', RandomForestRegressor(random_state=42))
])

final_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')

print(f"Результати крос-валідації (дефолт): {final_scores}")
print(f"Середній R^2: {final_scores.mean():.4f}")
print(f"Відхилення (стабільність): {final_scores.std():.4f}")
```

    Результати крос-валідації (дефолт): [0.72686245 0.74835522 0.74773186 0.74156855 0.75289926]
    Середній R^2: 0.7435
    Відхилення (стабільність): 0.0091
    


```python
plt.figure(figsize=(8, 5))
bars = plt.bar(['Фолд 1', 'Фолд 2', 'Фолд 3', 'Фолд 4', 'Фолд 5'], final_scores, color='skyblue')
plt.ylim(0.5, 0.8) 
plt.axhline(y=final_scores.mean(), color='red', linestyle='--', label=f'Середнє: {final_scores.mean():.2f}')
plt.title('Результати крос-валідації (R²)')
plt.ylabel('Значення R²')
plt.legend()

plt.show()
```


    
![png](output_87_0.png)
    


# Machine Learning


```python
final_model = best_model_upg
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) 

print("=== ФІНАЛЬНИЙ ЗВІТ МОДЕЛІ ===")
print(f"R^2 (Точність):        {r2:.4f}")
print(f"MAE (Сер. помилка):     {mae:.2f} грн")
print(f"MSE (Квадр. помилка):   {mse:.2f}")
print(f"RMSE (Корінь з MSE):    {rmse:.2f} грн")
```

    === ФІНАЛЬНИЙ ЗВІТ МОДЕЛІ ===
    R^2 (Точність):        0.7966
    MAE (Сер. помилка):     387.46 грн
    MSE (Квадр. помилка):   447360.30
    RMSE (Корінь з MSE):    668.85 грн
    


```python

importances = final_model.named_steps['model'].feature_importances_

feature_names = final_model.named_steps['col_prepr'].get_feature_names_out()

feature_names = [name.split('__')[1] for name in feature_names]

importance_df = pd.DataFrame({'Фактор': feature_names, 'Важливість': importances})

final_report = importance_df.groupby('Фактор').sum().sort_values(by='Важливість', ascending=False)
final_report['Вплив (%)'] = final_report['Важливість'] * 100

print(final_report[['Вплив (%)']])
```

                  Вплив (%)
    Фактор                 
    volume        35.160973
    width         26.666864
    depth         14.085388
    category      10.300102
    height         9.128127
    other_colors   4.658546
    


```python
plt.figure(figsize=(8, 4))
final_report['Вплив (%)'].plot(kind='bar', color='skyblue')
plt.title('Що найбільше впливає на ціну?')
plt.ylabel('Відсотки (%)')
plt.xticks(rotation=45)
plt.show()
```


    
![png](output_91_0.png)
    


Для прогнозування цін IKEA було обрано модель RandomForestRegressor. Після оптимізації параметрів за допомогою GridSearchCV (глибина 15, 200 дерев) модель досягла точності R^2 = 0.75. Крос-валідація підтвердила стабільність моделі (відхилення лише 2%).Найважливішими факторами при формуванні ціни виявилися volume та width, які сумарно визначають майже 76% вартості товару. Це підтверджує гіпотезу про те, що фізичні габарити та призначення меблів є ключовими ціноутворюючими показниками.


```python
y_pred_final = final_model.predict(X_test)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_final, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Реальні ціни vs Прогнозовані ціни')
plt.xlabel('Реальна ціна')
plt.ylabel('Прогноз моделі')
plt.show()
```


    
![png](output_93_0.png)
    


Графік залишків вказує на наявність систематичної недооцінки дорогих товарів. Це свідчить про те, що для преміум-сегмента існують значущі предиктори, які не представлені у поточному наборі даних. З огляду на специфіку ринку меблів, можна висунути гіпотезу, що такими факторами є тип матеріалів (наприклад, дерево vs ДСП) або належність до лімітованих колекцій. Це визначає напрямок для майбутнього покращення моделі — додавання ознак про склад продукту.

 (Коефіцієнт детермінації) — «Наскільки модель розумна?»
Це показник від 0 до 1 (або від 0% до 100%).
Що він каже: Яку частку розкиду цін модель змогла пояснити.
На прикладі IKEA: Якщо 
, це означає, що ваша модель зрозуміла 78% причин, чому товари мають різну ціну (врахувала розміри, категорії, дизайнерів). Решта 22% — це випадкові фактори або дані, яких у нас немає.
Чим вище, тим краще. (
 — це ідеал, але в реальному житті це ознака перенавчання).
 
MAE (Mean Absolute Error) — «Середня помилка в грошах»
Це середня арифметична помилка в тих самих одиницях, що й ціна (наприклад, у кронах).
Що він каже: В середньому, на скільки одиниць модель промахується "повз ціль".
На прикладі IKEA: Якщо MAE = 383, це означає, що в середньому модель помиляється на 383 крони (в ту чи іншу сторону).
Це найбільш зрозуміла метрика для бізнесу: «Ми помиляємося в середньому на 383 крони на кожному товарі».
Чим нижче, тим краще.

RMSE (Root Mean Squared Error) — «Покарання за грубі помилки»
Це корінь із середнього квадрата помилок.
Що він каже: Вона схожа на MAE, але сильніше штрафує модель за великі промахи.
На прикладі IKEA:
Якщо модель помилилася на 10 крон — це дрібниця.
Якщо модель помилилася на 5000 крон — RMSE різко зросте, значно сильніше, ніж MAE.
Навіщо вона потрібна: Якщо RMSE набагато більша за MAE (наприклад, MAE 380, а RMSE 700), це сигнал, що у вашому датасеті є викиди (дуже дорогі товари, на яких модель катастрофічно помиляється).
Чим нижче, тим краще.

 


```python

```
