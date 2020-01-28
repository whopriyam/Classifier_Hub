import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB
countries = pd.read_csv('countries of the world.csv')
countries.head()
countries.info()
countries.isnull().sum()
len(countries[countries.isna().any(axis=1)])
countries_clean = countries.drop(columns = ['Climate', 'Phones (per 1000)', 'Literacy (%)'])
countries_clean.isnull().sum()
len(countries_clean[countries_clean.isna().any(axis=1)])
countries_clean = countries_clean.dropna()
countries_clean.shape
countries_clean.isnull().sum()
countries_clean['Region'].unique()countries_clean['Region'].count()
countries_clean.head()
countries_clean.Region.value_counts()
countries_clean = countries_clean.replace({'Region':{'ASIA (EX. NEAR EAST)         ': 'Asia',
                                   'C.W. OF IND. STATES': 'Asia',
                                   'C.W. OF IND. STATES ': 'Asia',
                                   'ASIA (EX. NEAR EAST)         ': 'Asia',
                                   'NEAR EAST                          ': 'Asia',
                                   'EASTERN EUROPE                     ': 'Europe',
                                   'WESTERN EUROPE                     ': 'Europe',
                                   'BALTICS                            ':'Europe',
                                   'LATIN AMER. & CARIB    ': 'S. America',
                                   'NORTHERN AFRICA                    ': 'Africa',
                                   'SUB-SAHARAN AFRICA                 ': 'Africa',
                                   'OCEANIA                            ': 'Oceania',
                                   'NORTHERN AMERICA                   ': 'N. America'
                                   
    
}})
countries_clean.head()
countries_clean['Pop. Density (per sq. mi.)'] = countries_clean['Pop. Density (per sq. mi.)'].str.replace(',', '.').astype(float)
countries_clean['Coastline (coast/area ratio)'] = countries_clean['Coastline (coast/area ratio)'].str.replace(',', '.').astype(float)
countries_clean['Net migration'] = countries_clean['Net migration'].str.replace(',', '.').astype(float)
countries_clean['Infant mortality (per 1000 births)'] = countries_clean['Infant mortality (per 1000 births)'].str.replace(',', '.').astype(float)
countries_clean['Arable (%)'] = countries_clean['Arable (%)'].str.replace(',', '.').astype(float)
countries_clean['Crops (%)'] = countries_clean['Crops (%)'].str.replace(',', '.').astype(float)
countries_clean['Other (%)'] = countries_clean['Other (%)'].str.replace(',', '.').astype(float)
countries_clean['Birthrate'] = countries_clean['Birthrate'].str.replace(',', '.').astype(float)
countries_clean['Deathrate'] = countries_clean['Deathrate'].str.replace(',', '.').astype(float)
countries_clean['Agriculture'] = countries_clean['Agriculture'].str.replace(',', '.').astype(float)
countries_clean['Industry'] = countries_clean['Industry'].str.replace(',', '.').astype(float)
countries_clean['Service'] = countries_clean['Service'].str.replace(',', '.').astype(float)
countries_clean.head()
countries_clean.groupby('Region').hist(figsize=(12, 12))
countries_clean.columns
countries_classify = countries_clean[['Country', 'Region', 'Population',
       'Pop. Density (per sq. mi.)',
       'Net migration', 'Infant mortality (per 1000 births)',
       'GDP ($ per capita)', 'Arable (%)', 'Crops (%)', 'Other (%)',
       'Birthrate', 'Agriculture', 'Industry']]
countries_classify.Region.value_counts()
X = countries_classify.iloc[:, 3:-1].values
Y = countries_classify.iloc[:, 1:2].values
print(countries_classify.groupby('Region').size())
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
classifier = GaussianNB()
classifier.fit(X_train_scaled, Y_train)
acc= np.mean(Y_test==Y_pred)
print( acc)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_pred)