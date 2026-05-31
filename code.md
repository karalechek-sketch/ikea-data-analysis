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


```python
df = df.drop(columns=['Unnamed: 0'])
```


```python
df.describe()
```


```python
df.shape
```


```python
df.info()
```


```python
df.isnull().sum()
```


```python
df.isnull().sum() * 100 / len(df)
```


```python
df.columns
```


```python
df.duplicated().sum()
```


```python
print(df.nunique().to_markdown())
```


```python
df.duplicated(subset=['item_id']).sum()
```


```python
df[df["item_id"].duplicated(keep=False)].sort_values(by="item_id")
```

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


```python
df_base.drop(columns=['item_id'], inplace=True)
```


```python
df_base.head()
```


```python
print(df_base["category"].value_counts().to_markdown())        
```


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


```python
print((df_base.groupby('category')['price'].median().sort_values(ascending=False)).to_markdown())
```


```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_base, x='price', y='category')
plt.title('Розкид цін за категоріями')
plt.xlabel('Ціна')
plt.ylabel('Категорія')
plt.grid(axis='x') 
plt.show()
```

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

Ми бачимо, що розподіл ціни сильно зміщений вправо (positive skew). Більшість меблів IKEA коштують до 2000 одиниць, але є поодинокі екземпляри дорожче 8000. Це означає, що середня ціна буде вищою за медіану.
Оскільки високі ціни є реальними ринковими даними для певних категорій (наприклад, гардеробів), я вирішила не видаляти їх, щоб модель могла розпізнавати дорогі сегменти. 


```python
plt.figure(figsize=(9, 4))
df_base['price_log'] = np.log(df_base['price']) 
sns.histplot(df_base['price_log'], kde=True, binwidth=0.2, color='olive')
plt.title('Розподіл цін (логарифмічна шкала)')
plt.show()
```

Для стабілізації дисперсії та приведення розподілу цільової змінної до нормального вигляду було застосовано логарифмічне перетворення (log-transformation).
Це дозволяє зменшити вплив цінових викидів та покращити якість майбутньої прогнозної моделі.


```python
df_base['sellable_online'].value_counts(normalize=True)                                         
```


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

Аналіз стовпця sellable_online показав, що понад 99% товарів доступні для онлайн-замовлення. Через відсутність варіативності, ця ознака не несе інформативного навантаження для побудови прогнозних моделей і може бути вилучена з подальшого розгляду.


```python
df_base.drop(columns=['sellable_online'], inplace=True)
```


```python
df_base['other_colors'].value_counts(normalize=True) 
```


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

Ціновий розрив: Середня ціна товарів з додатковими кольорами (745) на 50% вища, ніж у товарів без них (495). Це статистичний факт у межах цього датасету.
Розподіл у каталозі: Товари без додаткових кольорів складають більшість асортименту (55%), але група з кольорами також є значною (45%).


```python
df_base['designer'].unique()
```


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


```python
sorted_designers= sorted(df_base['designer_cleaned'].unique().astype(str))
for name in sorted_designers:
    print(name)  
```


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


```python
print(df_base['designer_final'].value_counts().head(15).to_markdown())
```


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


```python
print(df_base[['width', 'height', 'depth']].isnull().sum().to_markdown())   
```


```python
for col in ['depth', 'height', 'width']:
    df_base[col] = df_base[col].fillna(df_base.groupby('category')[col].transform('median'))
```


```python
print(df_base[['width', 'height', 'depth']].isnull().sum().to_markdown()) 
```


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

Візуалізація (Scatter Plot) продемонструвала чітку тенденцію: зі збільшенням фізичних габаритів товару його вартість зростає. Хоча існують винятки (дороговартісні малогабаритні товари), загальний об'єм є ключовим ціноутворюючим фактором для асортименту IKEA.

# Data Preparation


```python
ml_data = df_base.copy()
```


```python
ml_data.columns
```


```python
ml_data.drop(columns=['name','old_price', 'short_description', 'price_log', 'designer', 'designer_cleaned', 
                      'designer_final', 'designer_hypothesis'], inplace=True)
```


```python
ml_data.columns
```


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


```python
final_scores = cross_val_score(best_model_upg, X_train, y_train, cv=5, scoring='r2')
print(f"Результати крос-валідації: {final_scores}")
print(f"Середній R^2: {final_scores.mean():.4f}")
print(f"Відхилення (стабільність): {final_scores.std():.4f}")
```


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


```python

importances = final_model.named_steps['model'].feature_importances_

feature_names = final_model.named_steps['col_prepr'].get_feature_names_out()

feature_names = [name.split('__')[1] for name in feature_names]

importance_df = pd.DataFrame({'Фактор': feature_names, 'Важливість': importances})

final_report = importance_df.groupby('Фактор').sum().sort_values(by='Важливість', ascending=False)
final_report['Вплив (%)'] = final_report['Важливість'] * 100

print(final_report[['Вплив (%)']])
```


```python
plt.figure(figsize=(8, 4))
final_report['Вплив (%)'].plot(kind='bar', color='skyblue')
plt.title('Що найбільше впливає на ціну?')
plt.ylabel('Відсотки (%)')
plt.xticks(rotation=45)
plt.show()
```

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
