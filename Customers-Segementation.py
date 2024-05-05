# ML-Project-9-Customers-Segmentation

Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import SilhouetteVisualizer
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from matplotlib import colors
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
palette = ["#a7e149","#e149a7", "#c393f4", "#d683f2", "#dd9e9e", "#fa91aa", "#fc9e9e", "#49e183", "#aa91aa", "#ff93f4"]
sns.set()
sns.set_style('whitegrid')
palette = ['#17bece', '#9467bd', '#2178b4', '#e177c2']
cmap = ListedColormap(palette)
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
import matplotlib
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from sklearn import metrics
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,f1_score,confusion_matrix,classification_report
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
%matplotlib inline
pd.set_option('display.max_rows', 50)

Loading the Data

df = pd.read_csv('/content/drive/MyDrive/marketing_campaign.csv', delimiter = '\t')
df.head()

Understanding the Data

df.columns

Data Cleaning

Missing Values

df.info()

df= df.dropna()
print("The total number of data-points after removing the rows with missing values are:", len(df))
The total number of data-points after removing the rows with missing values are: 2216

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def summary(df):
    # Create a DataFrame with missing value information
    missing_info = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])

    # Create a DataFrame with data types
    data_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
    # Combine all the information into a single DataFrame
    summary_df = pd.concat([missing_info, data_types], axis=1)

    # Iterate through each column in your DataFrame
    for column in df.columns:
          # Check if the column is categorical
        if pd.api.types.is_object_dtype(df[column]):
            num_unique_choices = df[column].nunique()
            # Add the number of unique choices to the summary DataFrame
            summary_df.loc[column, 'Unique Choices'] = num_unique_choices

        if pd.api.types.is_numeric_dtype(df[column]):
            # Get the lower and upper thresholds for outliers
            low_limit, up_limit = outlier_thresholds(df, column)

            # Calculate the range (max - min) for numeric columns
            summary_df.loc[column, 'min'] = df[column].min()
            # Calculate the range (max - min) for numeric columns
            summary_df.loc[column, 'max'] = df[column].max()
            summary_df.loc[column, 'Mean'] = df[column].mean()
            # Calculate the median (50% percentile)
            summary_df.loc[column, 'Median'] = df[column].median()
            # Calculate the variance
            summary_df.loc[column, 'Variance'] = df[column].var()
            # Calculate the standard deviation
            summary_df.loc[column, 'deviation'] = df[column].std()


            # Count the number of outliers in the column
            num_outliers = len(df[(df[column] < low_limit) | (df[column] > up_limit)])
            # Add the outlier count to the summary DataFrame
            summary_df.loc[column, 'Num Outliers'] = num_outliers


    return summary_df

summary(df)

print("Values of column 'Marital_Status' are:", df["Marital_Status"].unique())
print("Values of column 'Education' are:",df["Education"].unique())

display(df[0:5].T)

Checking correlation between the attributes

df.head()

df1=df.drop(['Education','Marital_Status','Dt_Customer'],axis=1)

# Check correlation

df_corr = df1.corr()
f, ax = plt.subplots(figsize=(16, 16))
# sns.heatmap(df_corr, vmax=.8, square=True)
# plt.show()

sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='RdYlGn',annot_kws={'size': 10}, ax=ax)
plt.show()

# Check the Correlation Report
corr_data = df1.corr()
corr_data.abs().unstack().sort_values(ascending=False)[24:50:2]

df.duplicated().sum()

print(df.isnull().sum())

# Downloading library to change null data in mean' for 'Income column''

from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

df.Income = mean_imputer.fit_transform(df[["Income"]])

# Check the number of null numbers (NaN)

df.isnull().sum()

** Outliers**

# Identify number of columns by type

numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
discrete_features = [feature for feature in numerical_features if len(df[feature].unique())<25]
continuous_features = [feature for feature in numerical_features if feature not in discrete_features]
categorical_features = [feature for feature in df.columns if feature not in numerical_features]
binary_categorical_features = [feature for feature in categorical_features if len(df[feature].unique()) <=3]
print("Numerical Features Count {}".format(len(numerical_features)))
print("Discrete features Count {}".format(len(discrete_features)))
print("Continuous features Count {}".format(len(continuous_features)))
print("Categorical features Count {}".format(len(categorical_features)))
print("Binary Categorical features Count {}".format(len(binary_categorical_features)))

outliers_features = [feature for feature in continuous_features if feature not in ['ID']]
print(outliers_features)

# Delete outliers

def plot_boxplot(df, continuous_features):
    # create copy of dataframe
    data = df[continuous_features].copy()
    # Create subplots
    fig, axes = plt.subplots(nrows=len(data.columns)//2, ncols=2,figsize=(15,10))
    fig.subplots_adjust(hspace=0.7)

    # set fontdict
    font = {'family': 'serif',
        'color':  'darkblue',
        'weight': 'normal',
        'size': 16,
        }

    # Generate distplot
    for ax, feature in zip(axes.flatten(), data.columns):
        sns.boxplot(data[feature],ax=ax)
        ax.set_title(f'Analysis of {feature}', fontdict=font)
    plt.show()

plot_boxplot(df, continuous_features)

# Remove outliers

def remove_outliers(df,outliers_features):

    data = df.copy()

    for feature in data[outliers_features].columns:
        Q3 = data[feature].quantile(0.75)
        Q1 = data[feature].quantile(0.25)
        IQR = Q3 - Q1
        lower_limit = round(Q1 - 1.5 * IQR)
        upper_limit = round(Q3 + 1.5 * IQR)
        data.loc[data[feature]>= upper_limit,feature] = upper_limit
        data.loc[data[feature]<=lower_limit,feature] = lower_limit
        data = data[(data[feature] < upper_limit) & (data[feature] > lower_limit)]
    return data

df = remove_outliers(df,outliers_features)

plot_boxplot(df, outliers_features)

df.describe()

# Checking non-numeric df columns

df.describe(include=['O'])

# check the unique values for 'Education'
df['Education'].value_counts()

df['Marital_Status'].value_counts()

df.shape

df.describe().T

def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage"""

    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of DataFrame is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

df= reduce_mem_usage(df)

df.rename(columns = {'MntGoldProds':'MntGoldProducts'}, inplace = True)

df['Income'].skew()

#📌 If the skewness is between -0.5 and 0.5, the data is fairly symmetrical. If the skewness is between -1 and – 0.5 or between 0.5 and 1, the data are moderately skewed. If the skewness is less than -1 or greater than 1, the data are highly skewed.

#drop un-needed columns
df_fin= df.drop(["ID","Z_CostContact", "Z_Revenue"], axis=1)
#drop rows with missing values
df_fin= df.dropna()

# Percentage of customers with different education levels
x = df_fin['Education'].value_counts().sort_values()
colors = plt.cm.Set2(range(len(x)))
plt.figure(figsize=(9,9))
plt.pie(x=x, colors=colors, labels=['2n Cycle', 'Basic', 'Master', 'PhD', 'Graduation'],
        wedgeprops={ 'width': 0.5},autopct = '%1.1f%%')
plt.title('Propotion of Education')

Univariant Analysis

#CHECKING NUMBER OF UNIQUE CATEGORIES PRESENT IN THE "Year_Birth"
print("Unique categories present in the Year_Birth:",df_fin["Year_Birth"].value_counts())

#CHECKING NUMBER OF UNIQUE CATEGORIES PRESENT IN THE "Education"
print("Unique categories present in the Education:",df_fin["Education"].value_counts())
print('\n')

#VISUALIZING THE "Education"
df['Education'].value_counts().plot(kind='bar',color = 'turquoise',edgecolor = "black",linewidth = 3)
plt.title("Frequency Of Each Category in the Education Variable \n",fontsize=24)
plt.figure(figsize=(8,8))

#CHECKING NUMBER OF UNIQUE CATEGORIES PRESENT IN THE "Marital_Status"
print("Unique categories present in the Marital_Status:",df['Marital_Status'].value_counts())
print("\n")

#VISUALIZING THE "Marital_Status"
df['Marital_Status'].value_counts().plot(kind='bar',color = 'turquoise',edgecolor = "black",linewidth = 3)
plt.title("Frequency Of Each Category in the Marital_Status Variable \n",fontsize=24)
plt.figure(figsize=(8,8))

#Maximum Income
df_fin['Income'].max()

#Minimum Income
df_fin['Income'].min()

#AverageIncome
df_fin['Income'].mean()

plt.figure(figsize=(8,8))
sns.distplot(df_fin["Income"],color = 'turquoise')
plt.show()
df_fin["Income"].plot.box(figsize=(8,8),color = 'turquoise')
plt.show()

df_fin['Kidhome'].unique()

df_fin['Teenhome'].unique()

df_fin['MntWines'].unique()

df_fin['MntFruits'].unique()

df_fin['MntMeatProducts'].unique()

df_fin['MntFishProducts'].unique()

df_fin['MntSweetProducts'].unique()

df_fin['MntGoldProducts'].unique()

df_fin['AcceptedCmp1'].unique()

df_fin['AcceptedCmp2'].unique()

df_fin['AcceptedCmp3'].unique()

df_fin['AcceptedCmp4'].unique()

df_fin['AcceptedCmp5'].unique()

df_fin['NumWebPurchases'].unique()

df_fin['NumCatalogPurchases'].unique()

df_fin['NumStorePurchases'].unique()

df_fin['NumDealsPurchases'].unique()

# People
df_fin[["Response", "Marital_Status"]].groupby(["Marital_Status"], as_index = False).mean().sort_values(by = "Response", ascending = False)

# People
df_fin[["Response", "Education"]].groupby(["Education"], as_index = False).mean().sort_values(by = "Response", ascending = False)

# People
df_fin[["Response", "Kidhome"]].groupby(["Kidhome"], as_index = False).mean().sort_values(by = "Response", ascending = False)

# People
df_fin[["Response", "Teenhome"]].groupby(["Teenhome"], as_index = False).mean().sort_values(by = "Response", ascending = False)

# People
df_fin[["Response", "Complain"]].groupby("Complain", as_index = False).mean().sort_values("Response", ascending = False)

# Promotion
df_fin[["Response", "NumDealsPurchases"]].groupby("NumDealsPurchases", as_index = False).mean().sort_values("Response", ascending = False)

# Promotion
df_fin[["Response", "AcceptedCmp1"]].groupby("AcceptedCmp1", as_index = False).mean().sort_values("Response", ascending = False)

# Promotion
df_fin[["Response", "AcceptedCmp2"]].groupby("AcceptedCmp2", as_index = False).mean().sort_values("Response", ascending = False)

# Promotion
df_fin[["Response", "AcceptedCmp3"]].groupby("AcceptedCmp3", as_index = False).mean().sort_values("Response", ascending = False)

# Promotion
df_fin[["Response", "AcceptedCmp4"]].groupby("AcceptedCmp4", as_index = False).mean().sort_values("Response", ascending = False)

# Promotion
df_fin[["Response", "AcceptedCmp5"]].groupby("AcceptedCmp5", as_index = False).mean().sort_values("Response", ascending = False)

df_fin[["Response", "NumWebPurchases"]].groupby("NumWebPurchases", as_index = False).mean().sort_values("Response", ascending = False)

# Place
df_fin[["Response", "NumCatalogPurchases"]].groupby("NumCatalogPurchases", as_index = False).mean().sort_values("Response", ascending = False)

# Place
df_fin[["Response", "NumStorePurchases"]].groupby("NumStorePurchases", as_index = False).mean().sort_values("Response", ascending = False)

# Place
df_fin[["Response", "NumWebVisitsMonth"]].groupby("NumWebVisitsMonth", as_index = False).mean().sort_values("Response", ascending = False)

income_null = df_fin[df_fin["Income"].isnull()]

df_fin[df_fin["Income"].isnull()]

promotion = df_fin[["NumDealsPurchases", "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]]
product = df_fin[["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProducts", "Response"]]
place =df_fin[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Response"]]

sns.heatmap(promotion.corr(), fmt = ".2f",annot = True)
plt.show

sns.heatmap(product.corr(), fmt = ".2f",annot = True)
plt.show

sns.heatmap(place.corr(), fmt = ".2f",annot = True)
plt.show

#View feature correlations with the 'Response' column
#Note: 'Response' will be the target for predictive modeling
response_corr_abs = np.abs(df1_fin.corr()['Response']).sort_values(ascending=False)[1:]
response_corr = df1_fin.corr()['Response'].sort_values(ascending=False)[1:]
print("Correlation Coefficients for 'Response'")
print('--------------------------------------------------------')
print(response_corr)

sns.countplot(x = "Teenhome", data = df_fin)
plt.xticks(rotation = 60)
plt.show()

sns.countplot(x = "Kidhome", data = df_fin)
plt.xticks(rotation = 60)
plt.show()

sns.countplot(x = "Education", data = df_fin)
plt.show()

sns.countplot(x = "Marital_Status", data = df_fin)
plt.show()

sns.countplot(x = "Marital_Status", data = df_fin)
plt.show()

#I don't know what these labels are :) YOLO, absurd as marital status?

#Married earn more money, Together follow them as a 2nd in the list; IF we see only the total amount they earn. However we can ignore Alone, absurd and YOLO ones. It's better to see what the other ones bought.

#If we think about the number differences of each group, getting sum of Wine, Meat etc would not be a good idea. I'd prefer means here. First, see table by grouby function.

var = 'Response'
sns.countplot(x=var, data=df_fin)
print(df[var].value_counts())

# ahh this plot looks good but, Instead of year of birth if we have age there, It will be great, So base on the age We can comment what
# is there expenditure capacity, What type of product they prefer
import seaborn as sns
from matplotlib import pyplot as plt
plt.figure(figsize=(18,6))
sns.set_theme(style = 'darkgrid')
sns.countplot(x = 'Year_Birth', data = df_fin)
plt.xticks(rotation=40)
plt.show()

Yea , I knew it, this plot make more sense Comments on Graph Most of our customers are in range of 40-57, Hurray... Surprise Surprise In this dataset i got customer with age 121, 122,128. I think he should go for Guiness World record, Yeaaa, You are wrong..... Not for his Age, I am thinking about his fitness level, That he came to buy, Enough talking... I think it is outlier

plt.figure(figsize=(18,6))
sns.set_theme(style = 'darkgrid')
sns.countplot(x = [2024]*len(df_fin) - df_fin['Year_Birth'].to_numpy())
plt.xlabel('Age by 2024')
plt.xticks(rotation=40)
plt.show()

#1.2) Education

#Possibilities a) It may be possible that the one with more education having more chances to have good salary than others. b) If he is having good salary then There will be high money flow going in and out, So chances to buy high quality products, high value products are higher. c) Usually high educated people want to invest there time more productvely compare to others, mean less time to spend on the groceries. There are chances that number of products purchased per visit are more than others.

#Comments

plt.figure(figsize=(18,8))
sns.set_theme(style = 'darkgrid')
sns.countplot(x = df_fin['Marital_Status'].to_numpy())
plt.xticks(rotation=40)
plt.show()

plt.figure(figsize=(25,15))
plt.subplot(2,3,1)
sns.barplot(y=df_fin.Marital_Status,x=df_fin.MntWines)
plt.title('Amount spent on Wines vs Marital Status')
plt.xlabel('Amount spent on Wines')
plt.ylabel('Marital Status')
plt.subplot(2,3,2)
sns.barplot(y=df_fin.Marital_Status,x=df_fin.MntFruits)
plt.title('Amount spent on Fruits vs Marital Status')
plt.xlabel('Amount spent on Fruits')
plt.ylabel('Marital Status')
plt.subplot(2,3,3)
sns.barplot(y=df_fin.Marital_Status,x=df_fin.MntMeatProducts)
plt.title('Amount spent on Meat Products vs Marital Status')
plt.xlabel('Amount spent on Meat Products')
plt.ylabel('Marital Status')
plt.subplot(2,3,4)
sns.barplot(y=df_fin.Marital_Status,x=df_fin.MntFishProducts)
plt.title('Amount spent on Fish Products vs Marital Status')
plt.xlabel('Amount spent on Fish Products')
plt.ylabel('Marital Status')
plt.subplot(2,3,5)
sns.barplot(y=df_fin.Marital_Status,x=df_fin.MntSweetProducts)
plt.title('Amount spent on Sweet Products vs Marital Status')
plt.xlabel('Amount spent on Sweet Products')
plt.ylabel('Marital Status')
plt.subplot(2,3,6)
sns.barplot(y=df_fin.Marital_Status,x=df_fin.MntGoldProducts)
plt.title('Amount spent on Gold Products vs Marital Status')
plt.xlabel('Amount spent on Gold Products')
plt.ylabel('Marital Status')
plt.show()

plt.figure(figsize=(20,15))
plt.subplot(2,3,1)
sns.barplot(x=df_fin.Response,y=df_fin.MntWines)
plt.title('Amount spent on Wines vs Response')
plt.xlabel('Amount spent on Wines')
plt.ylabel('Response')
plt.subplot(2,3,2)
sns.barplot(x=df_fin.Response,y=df_fin.MntFruits)
plt.title('Amount spent on Fruits vs Response')
plt.xlabel('Amount spent on Fruits')
plt.ylabel('Response')
plt.subplot(2,3,3)
sns.barplot(x=df_fin.Response,y=df_fin.MntMeatProducts)
plt.title('Amount spent on Meat Products vs Response')
plt.xlabel('Amount spent on Meat Products')
plt.ylabel('Response')
plt.subplot(2,3,4)
sns.barplot(x=df_fin.Response,y=df_fin.MntFishProducts)
plt.title('Amount spent on Fish Products vs Response')
plt.xlabel('Amount spent on Fish Products')
plt.ylabel('Response')
plt.subplot(2,3,5)
sns.barplot(x=df_fin.Response,y=df_fin.MntSweetProducts)
plt.title('Amount spent on Sweet Products vs Response')
plt.xlabel('Amount spent on Sweet Products')
plt.ylabel('Response')
plt.subplot(2,3,6)
sns.barplot(x=df_fin.Response,y=df_fin.MntGoldProducts)
plt.title('Amount spent on Gold Products vs Response')
plt.xlabel('Amount spent on Gold Products')
plt.ylabel('Response')
plt.show()

plt.figure(figsize=(20,15))
plt.subplot(2,3,1)
sns.barplot(x=df_fin.Complain,y=df_fin.MntWines)
plt.title('Amount spent on Wines vs Customer Complaints')
plt.xlabel('Amount spent on Wines')
plt.ylabel('Customer Complaints')
plt.subplot(2,3,2)
sns.barplot(x=df_fin.Complain,y=df_fin.MntFruits)
plt.title('Amount spent on Fruits vs Customer Complaints')
plt.xlabel('Amount spent on Fruits')
plt.ylabel('Customer Complaints')
plt.subplot(2,3,3)
sns.barplot(x=df_fin.Complain,y=df_fin.MntMeatProducts)
plt.title('Amount spent on Meat Products vs Customer Complaints')
plt.xlabel('Amount spent on Meat Products')
plt.ylabel('Customer Complaints')
plt.subplot(2,3,4)
sns.barplot(x=df_fin.Complain,y=df_fin.MntFishProducts)
plt.title('Amount spent on Fish Products vs Customer Complaints')
plt.xlabel('Amount spent on Fish Products')
plt.ylabel('Customer Complaints')
plt.subplot(2,3,5)
sns.barplot(x=df_fin.Complain,y=df_fin.MntSweetProducts)
plt.title('Amount spent on Sweet Products vs Customer Complaints')
plt.xlabel('Amount spent on Sweet Products')
plt.ylabel('Customer Complaints')
plt.subplot(2,3,6)
sns.barplot(x=df_fin.Complain,y=df_fin.MntGoldProducts)
plt.title('Amount spent on Gold Products vs Customer Complaints')
plt.xlabel('Amount spent on Gold Products')
plt.ylabel('Customer Complaints')
plt.show()

Conclusions On Objective 1 Positive response to previous marketing campaigns was the most correlated with a response to the most recent ad campaign. This shows that possibly the customers are very happy with the marketing campaigns and decide to respond to the next campaign. Or this could be showing a certain group of customers that are more influenced by the campaigns. Total amount spent on products, especially wines and meats, are very highly correlated with whether the customer responded to the marketing campaign. However, amount spent on gold, fish, sweets and fish were not as correlated. This could be due to the nature of the most recent marketing campaign - perhaps the store was trying to sell meat and wine in the most recent campaign? Catalog purchases correlate with response to the current marketing campaign where as in store, online, and deal purchases have very little to no correlation. This may be due to the medium that the marketing campaign was using - maybe it was not displayed in store/online but was in all the catalogs? Another possibility is that those customers who perform catalog purchases are more influenced by the campaigns Customers with smaller family size responded better to the marketing campaign. Maybe the customers without family had more money to spend on the products in the campaign or the products in the campaign were for signle customers (like alcohol and party supplies). Without further inforamation on the details on the campaign it is hard to say. Customers who recently purchased something are likely to respond to the marketing campaign. This is pretty clear - more recent purchases = probable pattern of shopping at the store Income and Total Amount Spent are very correlated. Customers who earn more spend more. Finally, of note is Age and Complaining had virtually 0 correlation with response. This shows that the campaign did a good job of catering to all age groups and that customers who complained in the past continued bussiness at the store

var = 'Response'
sns.countplot(x=var, data=df_fin)
print(df_fin[var].value_counts())

Plot2 = ['MntWines',    'MntFruits',    'MntMeatProducts',  'MntFishProducts',  'MntSweetProducts','MntGoldProducts','Income']

sns.pairplot(df[Plot2], hue = 'Income')

def categories(d):
    df_wine2 = pd.DataFrame((d.loc[:,('MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProducts')]).melt())
    fig = px.pie(df_wine2, values='value', names='variable',  width=800, height=400)
    fig.show()
The main category that customer spends the most money (50%) is a wine category. The second important category is meat.

wine = df_fin.groupby('Education')['MntWines'].mean().reset_index()
fruits = df_fin.groupby('Education')['MntFruits'].mean().reset_index()
meat = df_fin.groupby('Education')['MntMeatProducts'].mean().reset_index()
fish = df_fin.groupby('Education')['MntFishProducts'].mean().reset_index()
sweet = df_fin.groupby('Education')['MntSweetProducts'].mean().reset_index()
gold = df_fin.groupby('Education')['MntGoldProducts'].mean().reset_index()

fig, ax = plt.subplots()
fig.set_figheight(10)
fig.set_figwidth(10)
ax.bar(wine['Education'], wine['MntWines'], label='MntWines')
ax.bar(meat['Education'], meat['MntMeatProducts'],bottom=wine['MntWines'],label='MntMeatProducts')
ax.bar(fruits['Education'], fruits['MntFruits'],bottom=meat['MntMeatProducts'] + wine['MntWines'],label='MntFruits')
ax.bar(fish['Education'], fish['MntFishProducts'],bottom=meat['MntMeatProducts'] + wine['MntWines'] + fruits['MntFruits'],label='MntFishProducts')
ax.bar(sweet['Education'], sweet['MntSweetProducts'],bottom=meat['MntMeatProducts'] + wine['MntWines'] + fruits['MntFruits']+ fish['MntFishProducts'],label='MntSweetProducts')
ax.bar(gold['Education'], gold['MntGoldProducts'],bottom=meat['MntMeatProducts'] + wine['MntWines'] + fruits['MntFruits']+ fish['MntFishProducts'] + sweet['MntSweetProducts'],label='MntGoldProds')

ax.legend()

plt.show()

People with basic education spent much less at the store, so product managers shouls focus their activity primarily on the people with high education. Intriguily, PhD candidates spent the most for wine which is the main business source.

to_plot = ['Income', 'Recency', 'Marital_Status']
sns.pairplot(df_fin[to_plot], hue='Marital_Status', palette='Set1')
plt.show()

cont_col=['Income','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProducts']

for i in cont_col:
    plt.figure(figsize=(18,4))
    plt.subplot(1,2,1)
    sns.boxplot(x=df_fin[i],data=df_fin)
    plt.subplot(1,2,2)
    sns.histplot(x=df_fin[i],data=df_fin)
    plt.show()

obj = ['Education','Marital_Status']

for i in range(len(obj)):
    x='Marital_Status'
    for j in range(1):
        if obj[i] != x:
            sns.barplot(x= x,y='Income',hue=obj[i],data=df_fin)
            sns.set(rc={'figure.figsize':(11,12)})
            plt.show()

for all who do basic income is less Income for single is almost equal for 2nCycle,master,PhD less for basic income for graduation,masters, PhD is averagely above 40k average income for basic is 10k to 20k¶

# Education & Response
plt.figure(figsize=(15,5))
plt.subplot(121)
sns.histplot(data=df_fin, x="Education", hue="Response", multiple="stack", stat="percent")

# Marital_Status & Response
plt.subplot(122)
sns.histplot(data=df_fin, x="Marital_Status", hue="Response",stat="percent", multiple="stack")
plt.show()

From the left figure, we can find that the compaign acceptance rate in high education groups(Master and PhD) are higher than that in low education groups.

From the right plot, we find that the single people tend to say yes to this compaign.

# Kid Home & Response
plt.figure(figsize=(15,5))
plt.subplot(121)
sns.histplot(data=df_fin, x="Kidhome", hue="Response", multiple="stack", stat="percent", discrete=True)
plt.xticks([0, 1, 2])
# Teen Home & Response
plt.subplot(122)
sns.histplot(data=df_fin, x="Teenhome", hue="Response", multiple="stack", stat="percent", discrete=True)
plt.xticks([0, 1, 2])
plt.show()

It seems that customers with no kids and no teens at home are more likely to accept the offer in this campaign than customers with 2 kids but less than customers with one kid.

# Income (by Response/Marital_Status/Education/Kidhome)
plt.figure(figsize=(15,10))
plt.subplot(221)
sns.kdeplot(
   data=df_fin, x="Income", hue="Response", log_scale= True,
   fill=True, common_norm=False,
   alpha=.5, linewidth=0,
)
plt.gca().axes.get_yaxis().set_visible(False) # Set y invisible
plt.xlabel('Income')

# segment by Marital_Status
plt.subplot(222)
sns.kdeplot(
   data=df_fin, x="Income", hue="Marital_Status", log_scale= True,
   fill=True, common_norm=False, palette="crest",
   alpha=.5, linewidth=0,
)
plt.gca().axes.get_yaxis().set_visible(False)

# segment by Education
plt.subplot(223)
sns.kdeplot(
   data=df_fin, x="Income", hue="Education", log_scale= True,
   fill=True, common_norm=False, palette="crest",
   alpha=.5, linewidth=0,
   )
plt.gca().axes.get_yaxis().set_visible(False)

# segment by Kidhome
plt.subplot(224)
sns.kdeplot(
   data=df_fin, x="Income", hue="Kidhome", log_scale= True,
   fill=True, common_norm=False, palette="crest",
   alpha=.5, linewidth=0,
)
plt.gca().axes.get_yaxis().set_visible(False)

The plots about Income VS 4 different Discrete Variables give us some interesting information.¶ 1) The high income groups have larger possibility to accept offer in the compaign, as we can see the income distributions of people who say 'yes' and 'no' have a slight difference. 2) There are no clear income difference between people with different maritial status. 3) Customers only with basic education have significantly lower income, while bachelors, masters, and PhDs do not have clear difference in income level. 4) It seems that customers who don't have any kids at home have higher income levels.

Feature Engineering and bivariant analysis

#give each feature a smaller set of values
edu= {"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"}
df_fin["Education"]= df_fin["Education"].replace(edu)

status= {"YOLO": "Single", "Absurd": "Single", "Alone": "Single", "Widow": "Single", "Divorced": "Single", "Together": "Taken", "Married": "Taken"}
df_fin["Marital_Status"]= df_fin["Marital_Status"].replace(status)

#new values
print("Education Values: ", df_fin["Education"].unique())
print("Marital_Status Values:", df_fin["Marital_Status"].unique())
Education Values:  ['Graduate', 'Postgraduate', 'Undergraduate']
Categories (3, object): ['Undergraduate', 'Graduate', 'Postgraduate']
Marital_Status Values: ['Single', 'Taken']
Categories (2, object): ['Taken', 'Single']

sns.countplot(df_fin.Marital_Status)
sns.set(rc={'figure.figsize':(4,4)})
plt.show()

# Combining different dataframe into a single column to reduce the number of dimension

df_fin['Kids'] = df_fin['Kidhome'] + df_fin['Teenhome']

# Combining different dataframe into a single column to reduce the number of dimension

df_fin['Expenses'] = df_fin['MntWines'] + df_fin['MntFruits'] + df_fin['MntMeatProducts'] + df_fin['MntFishProducts'] + df_fin['MntSweetProducts'] + df_fin['MntGoldProducts']
df_fin['Expenses'].head(10)
]
sns.barplot(x = df_fin['Expenses'],y = df_fin['Education']);
plt.title('Total Expense based on the Education Level');

sns.barplot(x = df_fin['Income'],y = df_fin['Education']);
plt.title('Total Income based on the Education Level');

df_fin.describe()

#Minimum Expenses
df_fin['Expenses'].min()

#Maximum Expenses
df_fin['Expenses'].max()

#Average Expenses
df_fin['Expenses'].mean()

plt.figure(figsize=(8,8))
sns.distplot(df_fin["Expenses"],color = 'turquoise')
plt.show()
df_fin["Expenses"].plot.box(figsize=(8,8),color='turquoise')
plt.show()

fig, ax = plt.subplots(1,2, figsize = (20,12))
sns.histplot(ax = ax[0], data = df_fin.Income, color = "steelblue")
sns.histplot(ax = ax[1], data = df_fin.Expenses, color = "steelblue")

ax[0].set_title("Income of Customers", fontsize = 22, pad = 50)
ax[0].set_xlabel("Income [USD $]", fontsize = 20, labelpad = 35)

ax[1].set_title("Spending of Customers", fontsize = 22, pad = 50)
ax[1].set_xlabel("Amount Spent [USD $]", fontsize = 20, labelpad = 35)

for num in [0,1]:
    ax[num].grid(axis = "x")
    ax[num].set(ylabel = None)

df_fin.Income.describe()

df_fin.Expenses.describe()

The income of our customers varies by quite a bit with a normal distribution. The average salary seems to be around 52,000 USD and average spending around 200 USD.

Seems to be a pretty even distribution of old and new customers. This suggests that the company seems to be growing at a steady rate.

pd.crosstab(df_fin['Education'],df_fin['Expenses'],margins=True).style.background_gradient(cmap='Greys')

sns.set_theme(style="white")
plt.figure(figsize=(8,8))
plt.title("How Expenses impacts on Education?",fontsize=24)
ax = sns.barplot(x="Education", y="Expenses", data=df_fin,palette="rainbow")

pd.crosstab(df_fin['Marital_Status'],df_fin['Expenses'],margins=True).style.background_gradient(cmap='Greys')

sns.set_theme(style="white")
plt.figure(figsize=(8,8))
plt.title("How Marital_Status impacts on Education?",fontsize=24)
ax = sns.barplot(x="Marital_Status", y="Expenses", data=df_fin,palette="rainbow")

pd.crosstab(df_fin['Kids'],df_fin['Expenses'],margins=True).style.background_gradient(cmap='Greys')

sns.set_theme(style="white")
plt.figure(figsize=(8,8))
plt.title("How Kids impacts on Education?",fontsize=24)
ax = sns.barplot(x="Kids", y="Expenses", data=df_fin,palette="rainbow")

df_fin['TotalAcceptedCmp'] = df_fin['AcceptedCmp1'] + df_fin['AcceptedCmp2'] + df_fin['AcceptedCmp3'] + df_fin['AcceptedCmp4'] + df_fin['AcceptedCmp5']++ df_fin['Response']

pd.crosstab(df_fin['TotalAcceptedCmp'],df_fin['Expenses'],margins=True).style.background_gradient(cmap='Greys')

sns.set_theme(style="white")
plt.figure(figsize=(8,8))
plt.title("How TotalAcceptedCmp impacts on Education?",fontsize=24)
ax = sns.barplot(x="TotalAcceptedCmp", y="Expenses", data=df_fin,palette="rainbow")

sns.set_theme(style="white")
plt.figure(figsize=(8,8))
plt.title("How TotalAcceptedCmp impacts on Education?",fontsize=24)
ax = sns.barplot(x="TotalAcceptedCmp", y="Expenses", data=df_fin,palette="rainbow")

df_fin['NumTotalPurchases'] = df_fin['NumWebPurchases'] + df_fin['NumCatalogPurchases'] + df_fin['NumStorePurchases'] + df_fin['NumDealsPurchases']
df_fin['NumTotalPurchases'].unique()

#Minimum NumTotalPurchases
df_fin['NumTotalPurchases'].min()

#Maximum NumTotalPurchases
df_fin['NumTotalPurchases'].max()

#Mean NumTotalPurchases
df_fin['NumTotalPurchases'].mean()

pd.crosstab(df_fin['NumTotalPurchases'],df_fin['Expenses'],margins=True).head().style.background_gradient(cmap='Greys')

sns.set_theme(style="white")
plt.figure(figsize=(8,8))
plt.title("How NumTotalPurchases impacts on Education?",fontsize=24)
ax = sns.barplot(x="NumTotalPurchases", y="Expenses", data=df_fin,palette="rainbow")

#CHECKING NUMBER OF UNIQUE CATEGORIES PRESENT IN THE "Kids"
print("Unique categories present in the Kids:",df_fin['Kids'].value_counts())
print("\n")

#VISUALIZING THE "Kids"
df_fin['Kids'].value_counts().plot(kind='bar',color = 'turquoise',edgecolor = "black",linewidth = 3)
plt.title("Frequency Of Each Category in the Kids Variable \n",fontsize=24)
plt.figure(figsize=(8,8))

#CHECKING NUMBER OF UNIQUE CATEGORIES PRESENT IN THE "TotalAcceptedCmp"
print("Unique categories present in the TotalAcceptedCmp:",df_fin['TotalAcceptedCmp'].value_counts())
print("\n")

#VISUALIZING THE "TotalAcceptedCmp"
df_fin['TotalAcceptedCmp'].value_counts().plot(kind='bar',color = 'turquoise',edgecolor = "black",linewidth = 3)
plt.title("Frequency Of Each Category in the TotalAcceptedCmp Variable \n",fontsize=24)
plt.figure(figsize=(8,8))

plt.figure(figsize=(8,8))
sns.distplot(df_fin["NumTotalPurchases"],color = 'turquoise')
plt.show()
df_fin["NumTotalPurchases"].plot.box(figsize=(8,8),color = 'turquoise')
plt.show()

df_fin.Dt_Customer = pd.to_datetime(df_fin.Dt_Customer, format = "%d-%m-%Y")

df_fin.Dt_Customer

#ADDING A COLUMN "Age" IN THE DATAFRAME....
df_fin['Age'] = (pd.Timestamp('now').year) - (pd.to_datetime(df_fin['Dt_Customer']).dt.year)

#CHECKING NUMBER OF UNIQUE CATEGORIES PRESENT IN THE "Age"
print("Unique categories present in the Age:",df_fin['Age'].value_counts())
print("\n")

#VISUALIZING THE "Age"
df_fin['Age'].value_counts().plot(kind='bar',color = 'turquoise',edgecolor = "black",linewidth = 3)
plt.title("Frequency Of Each Category in the Age Variable \n",fontsize=24)
plt.figure(figsize=(8,8))

pd.crosstab(df_fin['Age'],df_fin['Expenses'],margins=True).style.background_gradient(cmap='Greys')

df_fin.head(5).style.background_gradient(cmap='Greys')

sns.set_theme(style="white")
plt.figure(figsize=(8,8))
plt.title("How Age impacts on Education?",fontsize=24)
ax = sns.barplot(x="Age", y="Expenses", data=df_fin,palette="rainbow")

print("Total categories in the feature Marital_Status:\n", df_fin["Marital_Status"].value_counts(), "\n")
print("Total categories in the feature Education:\n", df_fin["Education"].value_counts())

Total categories in the feature Education:

#Feature Engineering

#Feature pertaining parenthood
df_fin["Is_Parent"] = np.where(df_fin.Kids> 0, 1, 0)

#For clarity
df_fin=df_fin.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProducts":"Gold"})

#Dropping some of the redundant features
to_drop = ["Marital_Status"]
data = df_fin.drop(to_drop, axis=1)

ax = sns.countplot(data = df_fin, x = "Is_Parent", palette = "flare")
ax.set(xlabel=None,
      title = "Child Status of Customers")

df_fin.describe()

The above stats show some discrepancies in mean Income and Age and max Income and age.

Do note that max-age is 84 years, As I calculated the age that would be today (i.e. 2024) and the data is old.

I must take a look at the broader view of the data. I will plot some of the selected features.

df_fin[df_fin['Income']>80000]

#Dropping the outliers by setting a cap on Age and income.
df_fin =df_fin[(df_fin["Age"]<90)]
df_fin = df_fin[(df_fin["Income"]<80000)]
print("The total number of data-points after removing the outliers are:", len(df_fin))
The total number of data-points after removing the outliers are: 1235

df_fin

df_fin1=df_fin.drop(['Education','Marital_Status'],axis=1)

#correlation matrix
corrmat= df_fin1.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corrmat,annot=True, cmap=cmap, center=0)

def hist_with_vline(data, column):
    """This function gets data and column name.
    Plots a histogram with 100 bins, draws a Vline of the column mean and median"""

    plt.figure(figsize=(12,6))
    _ = sns.histplot(df[column], bins= 100)
    plt.title('Histogram of ' + column)
    miny, y_lim = plt.ylim()
    plt.text(s = f"Mean  {column} : {df[column].mean():.2f}", x =df[column].mean() * 1.1,  y = y_lim * 0.95, color = 'r')
    _ =plt.axvline(df[column].mean(), color = 'r')
    _ = plt.axvline(df[column].median())
    plt.text(s = f"Median {column} : {df[column].median():.2f}", x=df[column].median() * 1.1, y= y_lim * 0.90, color = 'b')

hist_with_vline(df_fin, 'Income')

The customers earn more than 80k; these ones are outliers also. I'll remove these too.

PS: I can remove outliers by Z score method or Tukey's IQR method; however, I wanted to remove the outliers by certain columns (the ones seemed important to me such as age, income)

df_fin = df_fin[df_fin['Income']<80000]

hist_with_vline(df_fin, 'Income')

Observation Most of the customers earn between 20k to 60k.

All the sold products histograms are right skewed. Majority of the customers buys items lower than certain amounts.

On the other hand, Wines are the most sold items (15884) and Meat producs follow with 364k, while the Fruit and Sweet products are the least sold items (6860 and 5878 respectively).

plt.figure(figsize=(8,8))
sns.barplot(x=df_fin['Marital_Status'], y=df_fin['Expenses'], hue = df_fin["Education"])
plt.title("Analysis of the Correlation between Marital Status and Expenses with respect to Education")
plt.show()

Observation: Less number of single customers and very high expenses for single customers.

plt.figure(figsize=(8,8))
plt.hist("Expenses", data = df_fin[df_fin["Marital_Status"] == "Taken"], alpha = 0.5, label = "Taken")
plt.hist("Expenses", data = df_fin[df_fin["Marital_Status"] == "Single"], alpha = 0.5, label = "Single")
plt.title("Distribution of Expenses with respect to Marital Status")
plt.xlabel("Expenses")
plt.legend(title = "Marital Status")
plt.show()

#from numpy.core.fromnumeric import size
plt.figure(figsize=(8,8))
plt.hist("Expenses", data = df_fin[df_fin["Education"] == "Postgraduate"], alpha = 0.5, label = "Postgraduate")
plt.hist("Expenses", data = df_fin[df_fin["Education"] == "Undergraduate"], alpha = 0.5, label = "Undergraduate")
plt.title("Distribution of Expenses with respect to Education")
plt.xlabel("Expenses")
plt.legend(title = "Education")
plt.show()

plt.figure(figsize=(8,8))
plt.hist("NumTotalPurchases", data = df_fin[df_fin["Education"] == "Postgraduate"], alpha = 0.5, label = "Postgraduate")
plt.hist("NumTotalPurchases", data = df_fin[df_fin["Education"] == "Undergraduate"], alpha = 0.5, label = "Undergraduate")
plt.title("Distribution of Number of Total Purchases with respect to Education")
plt.xlabel("Number of Total Purchases")
plt.legend(title = "Education")
plt.show()

plt.figure(figsize=(8,8))
plt.hist("Age", data = df_fin[df_fin["Marital_Status"] == "relationship"], alpha = 0.5, label = "relationship")
plt.hist("Age", data = df_fin[df_fin["Marital_Status"] == "Single"], alpha = 0.5, label = "Single")
plt.title("Distribution of Age with respect to Marital Status")
plt.xlabel("Age")
plt.legend(title = "Marital Status")
plt.show()

plt.figure(figsize=(8,8))
plt.hist("Income", data = df_fin[df_fin["Marital_Status"] == "relationship"], alpha = 0.5, label = "relationship")
plt.hist("Income", data = df_fin[df_fin["Marital_Status"] == "Single"], alpha = 0.5, label = "Single")
plt.title("Distribution of Income with respect to Marital Status")
plt.xlabel("Income")
plt.legend(title = "Marital Status")
plt.show()

sns.barplot(x = df_fin['Expenses'],y = df_fin['Education']);
plt.title('Total Expense based on the Education Level');
plt.show()

sns.barplot(x = df_fin['Income'],y = df_fin['Education']);
plt.title('Total Income based on the Education Level');
plt.show()

plt.figure(figsize=(12,6))

_ = sns.scatterplot(x ='Income',y = 'Expenses', data = df_fin)
_ = plt.title('Income vs Expenses')
_ = plt.ylabel('Total Items Bought')

There is a linear relation with income and number of items bought.

df_fin.Education.value_counts()
Education
Graduate         571
Postgraduate     507
Undergraduate    157
Name: count, dtype: int64

fig, (ax0, ax1 )= plt.subplots(1,2 , figsize=(12,6))
_= sns.barplot(x = 'Education', y = 'Income', data = df_fin, ax = ax0)
ax0.set_title('Income According to Education')
_ = sns.barplot(x = 'Education', y = 'Expenses', data = df_fin, ax=ax1)
ax1.set_title('Expenses By Custormers by Their Educational Status')

_ = ax0.text(s = f"n :{df_fin.Education.value_counts()[0]}", x = -0.35, y = 10000)
_ = ax0.text(s = f"n :{df_fin.Education.value_counts()[1]}", x = 0.75, y = 10000)
_ = ax0.text(s = f"n :{df_fin.Education.value_counts()[2]}", x = 1.75, y = 10000)
_ = ax0.text(s = f"n :{df_fin.Education.value_counts()[3]}", x = 2.75, y = 10000)
_ = ax0.text(s = f"n :{df_fin.Education.value_counts()[4]}", x = 3.75, y = 10000)

Customers with a Postgraduate earn and spend more than any other customers with different educational background. And, not so surprisingly Undergraduate level educated customers earn and spend the least amount of money.

And when we investigate the number of customers in each group, it is wise to investigate what customers buy with different educational backgrounds.

Does Children effect market shopping?

fig, (ax0,ax1) = plt.subplots(1,2,figsize=(12,6), sharex=True)
_ = sns.barplot(x= df_fin.Kids, y= df_fin.Income, ax=ax0)
_ = sns.barplot(x= df_fin.Kids, y= df_fin.Expenses, ax=ax1)
ax0.text(s= f"n:{df_fin[df_fin['Kids']==0]['Kids'].count()}", x = -0.25, y = 20000)
ax0.text(s= f"n:{df_fin[df_fin['Kids']==1]['Kids'].count()}", x = 0.75, y = 20000)
ax0.text(s= f"n:{df_fin[df_fin['Kids']==2]['Kids'].count()}", x = 1.75, y = 20000)
ax0.text(s= f"n:{df_fin[df_fin['Kids']==3]['Kids'].count()}", x = 2.75, y = 20000)

ax1.text(s = f"Mean Sales: \n{df_fin[df_fin['Kids']==0]['Expenses'].mean():.2f}", x = -0.35, y = 50)
ax1.text(s = f"Mean Sales: \n{df_fin[df_fin['Kids']==1]['Expenses'].mean():.2f}", x = 0.65, y = 50)
ax1.text(s = f"Mean Sales: \n{df_fin[df_fin['Kids']==2]['Expenses'].mean():.2f}", x = 1.65, y = 50)
ax1.text(s = f"Mean Sales: \n{df_fin[df_fin['Kids']==3]['Expenses'].mean():.2f}", x = 2.65, y = 50)

cols_to_plot = ["Income", "Age", "Expenses", "Recency", "NumTotalPurchases", "Kids"]
sns.set_theme()

sns.pairplot(df_fin[cols_to_plot])

Insights:

Middle age adults spent much more than the other age groups

plt.figure(figsize=(20, 10))
ax = sns.histplot(data = df_fin.Income, color = "midnightblue")
ax.set(title = "Income Distribution of Customers");

plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('Income', fontsize=20, labelpad=20)
plt.ylabel('Counts', fontsize=20, labelpad=20);

Insights:

The salaries of the customers have normal distribution with most of the customers earning between 25000 and 85000

Products = df_fin[['Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold']]
product_means = Products.mean(axis=0).sort_values(ascending=False)
product_means_df = pd.DataFrame(list(product_means.items()), columns=['Products', 'Expenses'])

plt.figure(figsize=(20,10))
plt.title('Expenses on Products')
sns.barplot(data=product_means_df, x='Products', y='Expenses');
plt.xlabel('Products', fontsize=20, labelpad=20)
plt.ylabel('Expenses', fontsize=20, labelpad=20);

Insights:

Wine and Meats products are the most famous products among the customers

Sweets and Fruits are not being purchased often

childrenspending = df_fin.groupby('Kids')['Expenses'].mean().sort_values(ascending=False)
childrenspending_df = pd.DataFrame(list(childrenspending.items()), columns=['Kids', 'Expenses'])

plt.figure(figsize=(20,10))

sns.barplot(data=childrenspending_df,  x="Kids", y="Expenses");
plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('Kids', fontsize=20, labelpad=20)
plt.ylabel('Expenses', fontsize=20, labelpad=20);

The families with no children and with one children earn and spend more than families with children.

# Group the data based on whether there are children and draw a scatter plot matrix
pairplot = data.loc[:, ['Income', 'Age', 'Expenses', 'Recency', 'Is_Parent']]

sns.pairplot(pairplot, hue='Is_Parent', palette='Set1')

# The relationship between education level and shopping expenses
# Comparison of customer spending in different marital status

to_boxplot = ['Education', 'Marital_Status']
fig, axes = plt.subplots(2, 1, figsize=(12, 12))
axes = axes.flatten()

for col, ax in zip(to_boxplot, axes):
    ax = sns.boxplot(data=df_fin, x=col, y='Expenses', ax=ax, palette='Set2')
    ax.set_title(f'boxplot of spent by {col}')

# Group customers with different marital status based on their education level, and compare their differences in income and shopping expenses
fig, axes = plt.subplots(2, 1, figsize=(14,10))
sns.barplot(x= 'Marital_Status',y='Income',hue='Education',data=df_fin, ci=0,palette='Set2', ax=axes[0])
sns.barplot(x= 'Marital_Status',y='Expenses',hue='Education',data=df_fin, ci=0, palette='Set2', ax=axes[1])

# Income and expenditure show a linear growth relationship
fig, axes = plt.subplots(1, 2, figsize=(14,6))
sns.scatterplot(y=data['Expenses'], x=data['Income'], ax=axes[0])
sns.regplot(y='Expenses', x='Income', data=data, ax=axes[1])

# Histogram of overall website customer spending
plt.figure(figsize=(12,6))
sns.histplot(data['Expenses'], bins=50, kde=True)
plt.title('histogram of Expenses')

# Frequency distribution of the number of days since the user's last purchase
plt.figure(figsize=(12,6))
sns.histplot(df_fin['Recency'], bins=60)

# Statistics of monthly activity frequency of website users
plt.figure(figsize=(12,6))
sns.histplot(df_fin['NumWebVisitsMonth'], bins=50, kde=True)
plt.title('Monthly activities (5-7)/month')

# Website complaints
x = df_fin['Complain'].value_counts().sort_values()
colors = plt.cm.Set2(range(len(x)))
plt.figure(figsize=(8,8))
plt.pie(x=x, colors=colors, wedgeprops={ 'width': 0.5},
        labels=['YES', 'NO'], autopct = '%1.1f%%')
plt.title('Complain of customer')

# Frequency statistics of 'MntWines', 'MntFishProducts', 'MntFruits', 'MntGoldProds', 'MntMeatProducts', 'MntSweetProducts'  respective sales
to_histplot = ['Wines', 'Fish', 'Fruits', 'Gold',
               'Meat', 'Sweets']

fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(14, 10))
axes = axes.flatten()

for col, ax in zip(to_histplot, axes):
    ax = sns.histplot(data=data, x=col, ax=ax)
    ax.set_title(f'histogram of {col}')

# Statistics of the frequency of purchases through the three channels of catalogs, stores and websites
to_histplot = ['NumCatalogPurchases', 'NumStorePurchases', 'NumWebPurchases']
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(14, 9))
axes = axes.flatten()

for col, ax in zip(to_histplot, axes):
    ax = sns.histplot(data=df_fin, x=col, ax=ax)
    plt.subplots_adjust(hspace = 0.5, wspace = 0.3)

# Create an indicator of the total number of bids accepted by the activity
acceptedConcat = df_fin[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']]
acceptedConcat = acceptedConcat.sum()

print('acceptedConcat:\n', acceptedConcat)

# Create an indicator of the total number of bids accepted by the activity
plt.figure(figsize=(12,6))
plt.title('accepted the campaings in every attempt')
sns.barplot(x=acceptedConcat.index, y=acceptedConcat, palette='Set2')

# The relationship between the number of days since the user’s last purchase and whether the offer was accepted in the last activity
plt.figure(figsize=(12,6))
plt.title('Recency Vs Acceptance of an offer')
sns.lineplot(x='Recency', y='Response', data=df_fin)

# Deleting some column to reduce dimension and complexity of model

col_del = ["AcceptedCmp1" , "AcceptedCmp2", "AcceptedCmp3" , "AcceptedCmp4","AcceptedCmp5", "Response","NumWebVisitsMonth", "NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumDealsPurchases" , "Kidhome", "Teenhome","Marital_Status","Education","Marital_Status","Dt_Customer","Age_Group","Z_CostContact","Z_Revenue"]
df=df_fin.drop(col_del,axis=1)
df.head()

df.shape

# Heatmap
df= df.corr()
f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn',annot_kws={'size': 10}, ax=ax)
plt.show()

x = df.columns
for i in x:
     print(i)

df.head(5).style.background_gradient(cmap='Greys')

df

#REARRANGE THE ORDER OF COLUMNS:-
order = [0,1,3,4,6,7,8,2,5]
df = df[[df.columns[i] for i in order]]
df.head(5).style.background_gradient(cmap='Greys')

df.head(5).style.background_gradient(cmap='Greys')

df.describe(include = 'all').style.background_gradient(cmap='Greys')

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True,cmap = 'Greys',linewidths=1)

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

encoder= LabelEncoder()
df= df_fin.apply(encoder.fit_transform)
#hot encode marital_status
df = pd.get_dummies(df_fin)

#check data
df.info()

df1=df.drop('Dt_Customer',axis=1)

df1

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
scaled_features = StandardScaler().fit_transform(df1.values)
sf_df = pd.DataFrame(scaled_features, index=df.index, columns=df1.columns)
temp = df1
X_std = StandardScaler().fit_transform(temp)
X = normalize(X_std, norm = 'l2')

df1 = df.fillna(df.mean()) # There are some nan value in input column , so filling value with mean

df1.head().style.background_gradient(cmap='Greys')

X = df1.iloc[:, [7, 3]].values
print(X)

#Scaled data to be used for reducing the dimensionality
print("Dataframe to be used for further modelling:")
df1.head()

df1=df1.drop(['Dt_Customer','Education_Undergraduate','Education_Graduate','Education_Postgraduate','Marital_Status_Taken','Marital_Status_Single','Age_Group_Young adult','Age_Group_Adult','Age_Group_Middel Aged','Age_Group_Senior Citizen'],axis=1)
DIMENSIONALITY REDUCTION

#Initiating PCA to reduce dimentions aka features to 3
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(df1)
PCA_ds = pd.DataFrame(pca.transform(df1), columns=(["col1","col2", "col3"]))
PCA_ds.describe().T

#A 3D Projection Of Data In The Reduced Dimension
x =PCA_ds["col1"]
y =PCA_ds["col2"]
z =PCA_ds["col3"]
#To plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z, c="maroon", marker="o" )
ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
plt.show()

Elbow Method - Now Lets Find The Number Of Clusters :-

# Quick examination of elbow method to find numbers of clusters to make.
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()

Agglomerative

#Initiating the Agglomerative Clustering model
AC = AgglomerativeClustering(n_clusters=5)
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
#Adding the Clusters feature to the orignal dataframe.
PCA_ds["Clusters"]= yhat_AC

#Plotting the clusters
fig = plt.figure(figsize=(10,8))
plt.axes(projection='3d').scatter(PCA_ds["col1"], PCA_ds["col2"], PCA_ds["col3"], c=PCA_ds["Clusters"], marker='o', cmap = 'viridis')
plt.title("The Plot Of The Clusters by Agglomerative model")

fig = sns.countplot(x=PCA_ds["Clusters"], palette= "Accent")
fig.set_title("Distribution Of The Clusters")
plt.show()

From the above plot, it can be clearly seen that cluster 0 is our biggest set of customers closely followed by cluster 1. We can explore what each cluster is spending on for the targeted marketing strategies.

Let us next explore how did our campaigns do in the past.

# Visualize the clustering results according to income and expenditure
plt.figure(figsize=(12, 6))
pl = sns.scatterplot(data = sf_df,x=sf_df["Expenses"], y=sf_df["Income"],hue=PCA_ds["Clusters"], palette= 'Set2')
pl.set_title("Cluster's Profile Based On Income And Spent")
plt.legend()
plt.show()

Income and expenditure graph shows cluster mode

Group 0: High expenditure and average income Group 1: High consumption and high income Group 2: Low expenditure and low income Group 3: Low expenditure and High income

plt.figure(figsize=(12,6))
sns.swarmplot(x=PCA_ds["Clusters"], y=sf_df["Expenses"], alpha=0.9, palette= 'Set2' )

# Visualize the total acceptance of activities according to the clustering results
plt.figure(figsize=(12,6))
pl = sns.countplot(x=sf_df["TotalAcceptedCmp"],hue=PCA_ds["Clusters"], palette= 'Set2')
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()

There has not been an overwhelming response to the campaigns so far. Very few participants overall. Moreover, no one part take in all 5 of them. Perhaps better-targeted and well-planned campaigns are required to boost sales.

Personal = ["Kidhome","Teenhome", "Age", "Kids", "Is_Parent", "Education_Undergraduate","Education_Graduate","Education_Postgraduate","Marital_Status_Single","Marital_Status_Taken"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=sf_df[i], y=sf_df["Expenses"], hue =PCA_ds["Clusters"], kind="kde", palette='Set2')
    plt.show()

Silhouette Score

from sklearn.metrics import silhouette_score

silhouette_scores = []
for i in range(2,10):
    m1=KMeans(n_clusters=i, random_state=42)
    c = m1.fit_predict(sf_df)
    silhouette_scores.append(silhouette_score(sf_df, m1.fit_predict(sf_df)))
plt.bar(range(2,10), silhouette_scores)
plt.xlabel('Number of clusters', fontsize = 20)
plt.ylabel('S(i)', fontsize = 20)
plt.show()

# Getting the maximum value of silhouette score and adding 2 in index because index starts from 2.

sc=max(silhouette_scores)
number_of_clusters=silhouette_scores.index(sc)+2
print("Number of Cluster Required is : ", number_of_clusters)
Number of Cluster Required is :  2
We should chose an amount of clusters for which there are no wide fluctuations in the size of the clusters, represented by the width of each one. Therefore, 4 seems like the right amount of clusters.

from sklearn.metrics import silhouette_score
def visualize_silhouette_layer(data):
    clusters_range = range(2,10)
    results = []

    for i in clusters_range:
        km = KMeans(n_clusters=i, random_state=42)
        cluster_labels = km.fit_predict(sf_df)
        silhouette_avg = silhouette_score(sf_df, PCA_ds["Clusters"])
        results.append([i, silhouette_avg])

    result = pd.DataFrame(results, columns=["n_clusters", "silhouette_score"])
    pivot_km = pd.pivot_table(result, index="n_clusters", values="silhouette_score")

    plt.figure()
    sns.heatmap(pivot_km, annot=True, linewidths=1, fmt='.3f', cmap='RdYlGn')
    plt.tight_layout()
    plt.show()

visualize_silhouette_layer(PCA_ds)

# First let's use KMeans to cluster the data
agrupador = KMeans(n_clusters = 4)
agrupador.fit(sf_df)
labels = agrupador.labels_
labels

agrupador_kmeans = KMeans(n_clusters = 2)
labels_kmeans = agrupador_kmeans.fit_predict(sf_df)
print("Labels K-means: ", labels_kmeans)

# Coefficient Silhouette - Score
print("The K-means silhouette coefficient is:", silhouette_score(sf_df, labels))
The K-means silhouette coefficient is: 0.08799378332986799
Insight: The silhouette coefficient varies from -1 to 1. If it is positive, we consider it good, and the closer to 1, the better.

# Statistical summary of data by cluster

sf_df["cluster"] = labels
sf_df.groupby("cluster").describe()

range_n_clusters = [i for i in range(2,10)]
print(range_n_clusters)

# Categorical variable distribution among clusters
sns.countplot(data=sf_df, x='cluster', hue='Marital_Status_Taken')
plt.title('Partner Distribution Among Clusters')
plt.show()

# Categorical variable distribution among clusters
sns.countplot(data=sf_df, x='cluster', hue='Education_Undergraduate')
plt.title('Education_Level Distribution Among Clusters')
plt.show()

# Categorical variable distribution among clusters
sns.countplot(data=sf_df, x='cluster', hue='Education_Graduate')
plt.title('Education_Level Distribution Among Clusters')
plt.show()

# Categorical variable distribution among clusters
sns.countplot(data=sf_df, x='cluster', hue='Education_Postgraduate')
plt.title('Education_Level Distribution Among Clusters')
plt.show()

# Categorical variable distribution among clusters
sns.countplot(data=sf_df, x='cluster', hue='Kids')
plt.title('Kids Distribution Among Clusters')
plt.show()

Analyzing the data above, we can extract several attempts, such as:

Cluster with highest 'Income' has highest 'Education_Level', highest 'Expenses' and 'TotalAcceptedCmp'

The cluster with the lowest 'Income' has the lowest 'Kids', 'Age', 'Education_Level', 'highest 'Expenses', 'TotalAcceptedCmp' and 'n_clients

The cluster with the second highest 'Income' has high 'Age", high 'Education_Level' and high 'n_clients'

The cluster with the third largest 'Income' has high 'Children' and the highest 'n_customers'

The 'Taken' variable is very close for all clusters.

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(sf_df)

    ssd.append(kmeans.inertia_)

plt.plot(ssd)
plt.show()

Gaussian Mixture Model

from sklearn.mixture import GaussianMixture

log_like_lst = []
all_cluster = 15

for k in range(2, all_cluster):
    gmm = GaussianMixture(n_components = k, random_state = 100).fit(sf_df)
    log_like = gmm.bic(sf_df)
    log_like_lst.append(log_like)

elbow = 8
plt.plot(range(2, all_cluster), log_like_lst, alpha=0.5)
plt.scatter(elbow, log_like_lst[elbow-2], s=100, c='r', marker='*')
plt.ylabel('BIC')
plt.xlabel('K')
plt.annotate('Optimal Point' ,(elbow, log_like_lst[elbow-1]), xytext=(elbow - 0.5,log_like_lst[elbow-2] + 3000))
plt.show()

GMM is a soft version of K-means, calculating the sample probability to different clusters. It is also a good clustering algorithm. Here, we use BIC to evluate the effectiveness of clustering. When K=8, the BIC score comes to the balanced point (will not show much improvement when increasing K), so we choose 8 as the final clustering result.

# Building & Fitting GMM Models
gmm = GaussianMixture(n_components = 8, random_state = 100).fit(sf_df)
labels = gmm.predict(sf_df)

sf_df['Cluster_GMM'] = labels + 1

# Inspect the cluter nums
sf_df["Cluster_GMM"].value_counts()

gmm = GaussianMixture(n_components = 8, covariance_type = "spherical", random_state = 0, max_iter = 1000).fit(X)
labels = gmm.fit_predict(sf_df)
sf_df["Cluster"] = labels
sf_df.head()

#Initiating the GMM Clustering model
gmm = GaussianMixture(n_components = 8, covariance_type = 'spherical', max_iter = 3000, random_state = 228).fit(PCA_ds)
# fit model and predict clusters
labels = gmm.predict(PCA_ds)
PCA_ds["Clusters"] = labels
#Adding the Clusters feature to the orignal dataframe.
PCA_ds["Clusters"]= labels

#Plotting the clusters
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(PCA_ds["col1"], PCA_ds["col2"], PCA_ds["col3"], s=40, c=PCA_ds["Clusters"], marker='o', cmap = 'viridis' )
ax.set_title("The Plot Of The Clusters")
plt.show()

# Inspect the cluster difference (By Income & Age)
sns.scatterplot(x='Income',y='Age',hue='Cluster_GMM',data=sf_df)
plt.show()

# For every columns in dataset
for i in sf_df:
    if i == 'Cluster':
        continue
    g = sns.FacetGrid(sf_df, col = "Cluster_GMM", hue = "Cluster_GMM", palette = "coolwarm", sharey=False, sharex=False)
    g.map(sns.histplot,i)
    g.set_xticklabels(rotation=30)
    g.set_yticklabels()
    g.fig.set_figheight(5)
    g.fig.set_figwidth(20)

Observations There are 8 different clusters, which is difficult to describe, but we could see clear difference in their basic information, family condition and consumption power ...

#Plotting countplot of clusters
pl = sns.countplot(x=sf_df["Cluster"])
pl.set_title("Distribution Of The Clusters")
plt.show()

#Creating a feature to get a sum of accepted promotions
sf_df["TotalAcceptedCmp"] = sf_df["AcceptedCmp1"]+ sf_df["AcceptedCmp2"]+ sf_df["AcceptedCmp3"]+ sf_df["AcceptedCmp4"]+ sf_df["AcceptedCmp5"]
#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=sf_df["TotalAcceptedCmp"],hue=sf_df["Cluster"])
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()

#Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=df["NumDealsPurchases"],x=sf_df["Cluster"])
pl.set_title("Number of Deals Purchased")
plt.show()

fig = plt.figure(figsize = (13, 4))
palette = ['#dd4124', '#009473', '#b4b4b4', '#336b87']
plt.title('Which clients take part in the promotions the most?', size = 25, x = 0.47, y = 1.1)
a = sns.barplot(data = sf_df.groupby(['Cluster']).agg({'TotalAcceptedCmp': 'sum'}).reset_index(),
                x = 'TotalAcceptedCmp', y = 'Cluster', palette = palette, linestyle = "-", linewidth = 1, edgecolor = "black")
plt.xticks([])
plt.yticks(fontname = 'monospace', size = 16, color = 'black')
plt.xlabel('')
plt.ylabel('')

for p in a.patches:
    width = p.get_width()
    plt.text(23 + width, p.get_y() + 0.55*p.get_height(), f'{round((width / 1001) * 100, 1)}%',
             ha = 'center', va = 'center', fontname = 'monospace', fontsize = 16, color = 'black')

for j in ['right', 'top', 'bottom']:
    a.spines[j].set_visible(False)
a.spines['left'].set_linewidth(1.5)

plt.show()

pd.options.display.float_format = "{:.0f}".format
summary = sf_df[['Income','Expenses','Cluster']]
summary.set_index("Cluster", inplace = True)
summary=summary.groupby('Cluster').describe().transpose()
summary

ax = sns.scatterplot(x = sf_df.Income,
               y = sf_df.Expenses,
               hue = sf_df.Cluster,
               palette = "muted")

ax.set_xlabel("Income [USD $]", fontsize = 20, labelpad = 20)
ax.set_ylabel("Amount Spent [USD $]", fontsize = 20, labelpad = 20)

sf_df_Expenses= sf_df.groupby('Cluster')[['Wines', 'Fruits','Meat',
                                                  'Fish', 'Sweets', 'Gold']].sum()

plt.figure(figsize=(30,15))
sf_df_Expenses.plot(kind='bar', stacked=True)

plt.title('Spending Habits by Cluster')
plt.xlabel('Cluster', fontsize=20, labelpad=20)
plt.ylabel('Expenses', fontsize=20, labelpad=20);
plt.xticks(rotation=0, ha='center');

Customers from all the segments have spent most of their money on Wine and Meat products

sf_df_purchases = sf_df.groupby('Cluster')[['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
                                                  'NumStorePurchases', 'NumWebVisitsMonth']].sum()

plt.figure(figsize=(30,15))
sf_df_purchases.plot(kind='bar', color=['black', 'red', 'green', 'coral', 'cyan'])

plt.title('Purchasing Habits by Cluster')
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Purchases', fontsize=20, labelpad=20);
plt.xticks(rotation=0, ha='center');

Insights:

1 and 5 Customers mostly likely to do store purchasing Most of the web and catalog purchases are also done by the customers from 1 and 5 segments 1 and 5 categoriesnalso like to buy from the stores Deal purchases are common among the Gold and Silver customers 5 category customers made the most number of web visits while customers from 2 segment have least web visits

sf_df_campaign = sf_df.groupby('Cluster')[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4',
                                                  'AcceptedCmp5', 'Response']].sum()

plt.figure(figsize=(30,15))
sf_df_campaign.plot(kind='bar', color=['tomato', 'salmon', 'green', 'coral', 'cyan', 'orange'])

plt.title('Promotions Acceptance by Cluster')
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Promotion Counts', fontsize=20, labelpad=20);
plt.xticks(rotation=0, ha='center');

cluster 4 accepted the most of the offers from the comapany Compaign 1, 5 and 1 seems to be the most successful one's cluster 5 showed the least interest in the promotion campaings of the company

ax = sns.barplot(x = sf_df.Cluster, y = sf_df.NumWebVisitsMonth, palette = "muted")
ax.set_ylabel("Number of Website Visits", labelpad = 20)
ax.set_xlabel(None)
ax.set_title("Average Website Visits in the Last Month by Cluster")

Interesting enough, the groups that make up the most website visits are the groups that spend the least.

temp = sf_df.loc[:, ["Wines", "Fruits", "Meat", "Sweets", "Cluster"]]
temp = temp.groupby("Cluster").sum()
temp.head()

Seems like customers spend the most money on Wine. Lets investigate this product category further.

sns.set_theme(style="whitegrid")
pal = ["#2E003E", "#3D1E6D", "#8874A3", "#D5CEE1"]

plt.figure(figsize=(12, 8))
sns.barplot(x=sf_df["Cluster"], y=sf_df["Income"], palette=pal)
plt.title("Income vs Cluster", size=15)
plt.show()

Observations:

The cluster which has the highest income is Cluster 6 Income of Cluster 2 is relatively lower than incomes of other clusters

plt.figure(figsize=(12, 8))
sns.boxenplot(x=sf_df["Cluster"], y=sf_df["Expenses"], palette=pal)
plt.title("Money Spent vs Clusters", size=15)
plt.show()

Observations:

Cluster 0 and 5 are spending the least money Cluster 2 is the cluster that spends the most money among other clusters

plt.figure(figsize=(12, 8))
sns.boxplot(x=sf_df["Cluster"], y=sf_df["NumTotalPurchases"], palette=pal)
plt.title("Purchase Number vs Cluster", size=15)
plt.show()

Observations:

Cluster 2 has the highest purchase number Cluster 2 does the least shopping

plt.figure(figsize=(12, 8))
sns.barplot(x=sf_df["Cluster"], y=sf_df["Kids"], palette=pal)
plt.title("Kids Number vs Cluster", size=15)
plt.show()

Observations:

Cluster 2 has nearly no child Cluster 6 has the most children among other clusters

plt.figure(figsize=(12, 8))
sns.violinplot(x=sf_df["Cluster"], y=sf_df["TotalAcceptedCmp"], palette=pal)
plt.title("Number of Purchase with Discount vs Clusters", size=15)
plt.show()

Observations:

Cluster 1 benefits least from the discounts Cluster 7 has the highest number of purchase with discount

Conclusion¶

Cluster 2: Is least-earner

Cluster 0 Has a tendecy to spend less money

Cluster 1 Has least purchase number (shop-hater)

Cluster 6: Has the highest income

Cluster 2 Spends the most money

Cluster 7 Has the highest purchase number (shop-lover)

Cluster 2 Has the least number of children

Cluster 1 Is the one that benefits least from discounts

Cluster 6: Has most children

Cluster 7 Is the cluster that shops most when there is a discount

Marketing Suggestions¶ Cluster 0 spends the least money So, you should gather the information about the its location and know the reason behind this.

Cluster 1 has the least purchase number and benefits least from discounts. So, you should gather the information about the its location and increase the discount rates at shops located at those locations.

Cluster 2 is the least earner ,spends the most money and has the least number of children

Cluster 6 has the highest income and most children. It can also be observed that they shave the least purchase number . Meaning that you need to consider discounting. In addition to that, if you make the discounts with a slogan like "Make Your Child Happy" in shops at those locations where these people live, because it could remind them that they are parents, it would possibly increase the number of sales.

Cluster 7 has the highest purchase number(shop lover) and it is the highest shopping cluster when there is discount.Shops discount policy should be implemented on other shops in the remaining clusters

Affinity Clustering Model

from sklearn.cluster import AffinityPropagation
#Initiating the Affinity Clustering model
AP = AffinityPropagation(damping=0.9)
# fit model and predict clusters
AP_df = AP.fit_predict(PCA_ds)
PCA_ds["Cluster"] = AP_df
#Adding the Clusters feature to the orignal dataframe.
sf_df["Cluster"]= AP_df

#Plotting the clusters
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Cluster"], marker='o', cmap = 'viridis' )
ax.set_title("The Plot Of The Clusters")
plt.show()

#Plotting countplot of clusters
pl = sns.countplot(x=sf_df["Cluster"])
pl.set_title("Distribution Of The Clusters")
plt.show()

pl = sns.scatterplot(data = sf_df,x=sf_df["Expenses"], y=sf_df["Income"],hue=sf_df["Cluster"])
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()

plt.figure()
pl=sns.swarmplot(x=sf_df["Cluster"], y=sf_df["Expenses"], color= "#CBEDDD", alpha=0.5 )
pl=sns.boxenplot(x=sf_df["Cluster"], y=sf_df["Expenses"])
plt.show()

#Creating a feature to get a sum of accepted promotions
sf_df["TotalAcceptedCmp"] = sf_df["AcceptedCmp1"]+ sf_df["AcceptedCmp2"]+ sf_df["AcceptedCmp3"]+ sf_df["AcceptedCmp4"]+ sf_df["AcceptedCmp5"]
#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=df["TotalAcceptedCmp"],hue=sf_df["Cluster"])
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()

#Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=sf_df["NumDealsPurchases"],x=sf_df["Cluster"])
pl.set_title("Number of Deals Purchased")
plt.show()

BIRCH

from sklearn.cluster import Birch
#Initiating the Birch Clustering model
BP = Birch(threshold=0.01, n_clusters=4)
# fit model and predict clusters
BP_df = BP.fit_predict(PCA_ds)
PCA_ds["Clusters"] = BP_df
#Adding the Clusters feature to the orignal dataframe.
sf_df["Clusters"] = BP_df

#Plotting the clusters
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap = 'viridis' )
ax.set_title("The Plot Of The Clusters")
plt.show()

#Plotting countplot of clusters
pl = sns.countplot(x=sf_df["Clusters"])
pl.set_title("Distribution Of The Clusters")
plt.show()

pl = sns.scatterplot(data = sf_df,x=sf_df["Expenses"], y=df["Income"],hue=sf_df["Clusters"])
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()

#Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=sf_df["NumDealsPurchases"],x=sf_df["Clusters"])
pl.set_title("Number of Deals Purchased")
plt.show()

K - means Clustering :-

#Initiating the KMeans Clustering model
kmeans = KMeans(n_clusters =4 , init = 'k-means++', random_state = 50)
# fit model and predict clusters
labels = kmeans.fit_predict(PCA_ds)
PCA_ds["Clusters"] = labels
#Adding the Clusters feature to the orignal dataframe.
sf_df["Clusters"]= labels

#Plotting the clusters
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(PCA_ds["col1"], PCA_ds["col2"], PCA_ds["col3"], s=40, c=PCA_ds["Clusters"], marker='o', cmap = 'viridis' )
ax.set_title("The Plot Of The Clusters")
plt.show()

kmeans = KMeans(n_clusters=4,max_iter=100)
kmeans.fit(PCA_ds)

sf_df['Cluster'] = kmeans.labels_
sf_df.head()

from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan

def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

    rand_X = sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0

    return H

plt.figure(figsize=(10,5))
sns.scatterplot(x='Wines',y='Fruits',data=sf_df,hue='Cluster',palette='deep')
plt.title('Amount Spent on Fruits vs Amount Spent on Wines')
plt.xlabel('Amount Spent on Wines')
plt.ylabel('Amount Spent on Fruits')
plt.show()

plt.figure(figsize=(10,5))
sns.scatterplot(x='Meat',y='Fish',data=sf_df,hue='Cluster',palette='deep')
plt.title('Amount Spent on Meats vs Amount Spent on Fish')
plt.ylabel('Amount Spent on Fish')
plt.xlabel('Amount Spent on Meats')
plt.show()

plt.figure(figsize=(10,5))
sns.scatterplot(x='Sweets',y='Gold',data=sf_df,hue='Cluster',palette='deep')
plt.title('Amount Spent on Sweets vs Amount Spent on Gold')
plt.xlabel('Amount Spent on Sweets')
plt.ylabel('Amount Spent on Gold')
plt.show()

print('Cluster 1: ',list(sf_df[sf_df.Cluster==0].index))

print('Cluster 2: ',list(sf_df[sf_df.Cluster==1].index))

print('Cluster 3: ',list(sf_df[sf_df.Cluster==2].index))

print('Cluster 4: ',list(sf_df[sf_df.Cluster==3].index))

hopkins(sf_df)

k_range = range(2, 6)
fig, axes = plt.subplots(4, 1, figsize=(10, 18))

for i in k_range:
    model = KMeans(i, init='k-means++', n_init=100, random_state=42)
    visualizer = SilhouetteVisualizer(model, colors='yellowbrick', ax=axes[i-2])
    visualizer.fit(PCA_ds)
    visualizer.finalize()
    axes[i-2].set_xlim(-0.1, 1)

plt.tight_layout()

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(PCA_ds)

pred = kmeans.predict(PCA_ds)

PCA_ds= PCA_ds.copy()
sf_df['Cluster'] = pred + 1

# Libraries for clustering and evaluation
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# Instantiate the clustering model and visualizer
model = KMeans(init = 'k-means++')
k_lst = []

# perform K-means 4 times(different intial clusters)
plt.figure(figsize=(15,10))
plt.subplot(221)
visualizer = KElbowVisualizer(model, k=(2,15), metric='distortion')
visualizer.fit(PCA_ds)        # Fit the data to the visualizer
visualizer.finalize()
k_lst.append(visualizer.elbow_value_)

plt.subplot(222)
visualizer = KElbowVisualizer(model, k=(2,15), metric='distortion')
visualizer.fit(PCA_ds)        # Fit the data to the visualizer
visualizer.finalize()
k_lst.append(visualizer.elbow_value_)

plt.subplot(223)
visualizer = KElbowVisualizer(model, k=(2,15), metric='distortion')
visualizer.fit(PCA_ds)        # Fit the data to the visualizer
visualizer.finalize()
k_lst.append(visualizer.elbow_value_)

plt.subplot(224)
visualizer = KElbowVisualizer(model, k=(2,15), metric='distortion')
visualizer.fit(PCA_ds)        # Fit the data to the visualizer
visualizer.finalize()
k_lst.append(visualizer.elbow_value_)

print('Mean K: ', np.mean(k_lst))

Different Scoring Metrics for K-means The above scoring parameter metric is set to distortion, which computes the sum of squared distances from each point to its assigned center.

The silhouette score calculates the mean Silhouette Coefficient of all samples, while the calinski_harabasz score computes the ratio of dispersion between and within clusters. I also use these 2 metrics to get clusters.

# Calinski_harabasz Scoring Matrix
plt.figure(figsize=(18,5))

plt.subplot(121)
visualizer = KElbowVisualizer(model, k=(2,15), metric='calinski_harabasz')
visualizer.fit(PCA_ds)        # Fit the data to the visualizer
visualizer.finalize()

# Silhouette Scoring Matrix
plt.subplot(122)
visualizer = KElbowVisualizer(model, k=(2,15), metric='silhouette')
visualizer.fit(PCA_ds)        # Fit the data to the visualizer
visualizer.finalize()
plt.show()

Choosing K value When we use inertia, Calinski_harabasz scorer and Silhouette scorer, we all get elbow point at K=2;

While Distortion scorer says that K=7, we could see distortion scores come at eblow point at about 3.

Considering all the clutering results, I finally choose K=2 to cluster the customers.

# Building & Fitting K-Means Models
kmeans = KMeans(n_clusters=2, init = 'k-means++').fit(PCA_ds)
pred = kmeans.predict(PCA_ds)
sf_df['Cluster'] = pred + 1

sf_df.head()

# Inspect the cluter nums
sf_df["Cluster"].value_counts()

# We could see the the clear difference between the 2 cluster (A.T Income and MntTotal)
sns.scatterplot(x='Income',y='Expenses',hue='Cluster',data=sf_df)
plt.show()

# For every columns in dataset
for i in sf_df:
    g = sns.FacetGrid(sf_df, col = "Cluster", hue = "Cluster", palette = "coolwarm", sharey=False, sharex=False)
    g.map(sns.histplot,i)

    g.set_xticklabels(rotation=30)
    g.set_yticklabels()
    g.fig.set_figheight(5)
    g.fig.set_figwidth(20)

Observations The income level of this 2 groups shows clear difference. Cluster 2 have obvious higher income than Cluster 1. Cluster 2 have less kids(as well as less children) at home. Most of them have no kids and only a few(about 5%) have 1 kids; While most customers in cluster 2 have 1 kid, and some have 2 kids. Cluster 2 customers buy much more amount products than cluster 1 customers. Almost every products shows the same trend. This result indicates that people in cluster 2 have more consumption power, and they are more likely to purchase goods from the company. The total number of accepting offers in compaigns is also consistent with my conclusion. The group with more consumption power(cluster 2) accept more offers than the other. Also, people in cluster 1 have much more purchasing numbers in different place. Among all these places, they may prefer to buy products in real store. It is not surprising since most of our customers are in their middle age or old age. The last small obervation is that cluster 2 have some extreme situations in product purchasing amount. Some customer in cluster 2 purchasing unusual amount of products. One plausible assumption is that they might buy a lot of goods for special festivals or parties. Checking their purchasing date may verify this assumption.

from sklearn.cluster import *
kmeans = KMeans(n_clusters=4)
yhat_kmeans = kmeans.fit_predict(PCA_ds)
PCA_ds["KmeanCluster"] = yhat_kmeans
sf_df["KmeanClusters"]= yhat_kmeans

sf_df

pca = PCA(n_components=2)
pca_data = pca.fit_transform(sf_df)
pca_df = pd.DataFrame.from_records(data=pca_data, columns=["x1","x2"])
pca_df["Cluster"] = pred + 1

fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d')
ax.scatter(pca_df['x1'], pca_df['x2'], c=pca_df["Cluster"], marker='o',s=50,cmap = 'brg' )
plt.show()

sns.scatterplot(data = sf_df,x = 'Expenses',y = 'Income',hue = 'Cluster',palette = 'viridis');

sns.boxenplot(x = 'Cluster' , y ='Expenses' ,data = sf_df);

Group 1: high spending and average income Group 2: high spending and high income

# Spent vs Products
Product_vars = ['Wines',
       'Fruits', 'Meat', 'Fish', 'Sweets',
       'Gold']

for i in Product_vars:
    plt.figure(figsize = (7,7))
    sns.barplot(x  = 'Cluster' , y = i,data = sf_df)
    plt.show()

Personal_vars = ['Age','Education_Undergraduate','Education_Graduate','Education_Postgraduate','Kidhome','Teenhome','Kids','Is_Parent','Marital_Status_Single','Marital_Status_Taken']
plt.figure(figsize = (10,7))
for i in (Personal_vars):

    sns.catplot(data =sf_df,x='Cluster',y=i)

Place_vars = ['NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

for i in Place_vars:
    plt.figure(figsize = (7,7))
    sns.boxplot(x='Cluster',y=i,data= sf_df)
    plt.show()

Conclusion Segmentation¶ Write a short text of what is the key business takeaway of the recommendation. Group 1 high spending and high income More number of store purchases and catalog purchases Family size is atmost 3 Atmost 1 child Spend on all products Group 2 low spending and low income more web visits at most 2 children have only 1 teen

sns.countplot(x=sf_df["TotalAcceptedCmp"],hue=sf_df["Cluster"]);

sns.boxenplot(y = 'NumDealsPurchases',x = 'Cluster',data=sf_df);

sizes = dict(sf_df['Cluster'].value_counts())

plt.figure(figsize=(12, 8))
plt.title("Clusters proportions")
plt.pie(sizes.values(), labels=sorted(sizes.keys()), autopct="%.1f%%", pctdistance=0.85, shadow=True, colors=palette)
plt.legend(title="Customer's cluster", labels=sorted(sizes.keys()), bbox_to_anchor=(1, 1))

# add a circle at the center to transform it in a donut chart
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.show()

Cluster 1 is the biggets cluster, around 55 % of all customers. Clusters 45 %

plt.figure(figsize=(16,5))
plt.title(f'Customers income by cluster')
ax = sns.boxplot(data=sf_df, x='Cluster', y='Income', palette=palette, showfliers=False)
plt.show()

sf_df.query('Income > 140000')

And the income outliers are distributed in 3rd cluster

plt.figure(figsize=(16,5))
plt.title(f'Customers amount spent by clusters')
ax = sns.boxplot(data=sf_df, x='Cluster', y='NumWebVisitsMonth', palette=palette, showfliers=False)
plt.show()

plt.figure(figsize=(16,5))
plt.title(f'Customers amount spent by clusters')
ax = sns.boxplot(data=sf_df, x='Cluster', y='NumTotalPurchases', palette=palette, showfliers=False)
plt.show()

Cluster 1 is the most active and frequent buyers

plt.figure(figsize=(16,5))
plt.title(f'Countplot of education degrees by clusters')
sns.countplot(data=sf_df, x='Education_Undergraduate', hue='Cluster', palette=palette)
plt.show()

plt.figure(figsize=(16,5))
plt.title(f'Countplot of education degrees by clusters')
sns.countplot(data=sf_df, x='Education_Graduate', hue='Cluster', palette=palette)
plt.show()

plt.figure(figsize=(16,5))
plt.title(f'Countplot of education degrees by clusters')
sns.countplot(data=sf_df, x='Education_Postgraduate', hue='Cluster', palette=palette)
plt.show()

The Basic degree is presented mostly in 1st cluster

fig, axes = plt.subplots(2,2, figsize=(16, 10))
k = 0
for i in range(0, 2):
    for j in range(0, 2):
        k += 1
        sizes = dict(sf_df.query(f'Cluster == {k}')['Is_Parent'].value_counts().sort_index(ascending=False))
        axes[i, j].set_title(f"Cluster {k}")
        axes[i, j].pie(sizes.values(), labels=['Yes', 'No'], autopct="%.1f%%", pctdistance=0.75, shadow=True, colors=palette)

fig.suptitle('Having children in different clusters')
fig.legend(title="Does the customer have children", labels=['Yes', 'No'], bbox_to_anchor=(1, 1))
fig.show()

plt.figure(figsize=(16,5))
plt.title(f'Number of web visits per month by clusters')
ax = sns.boxplot(data=sf_df, x='Cluster', y='NumWebVisitsMonth', palette=palette, showfliers=False)
plt.show()

fig, axes = plt.subplots(4, 1, figsize=(16, 20))

for i in range(1, 5):
    ax = (sf_df.query(f'Cluster == {i}')[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']]
          .sum()
          .divide(sf_df.query(f'Cluster == {i}').shape[0]).multiply(100)
          .plot(kind='bar', figsize=(18,15), title=f'% of customers from Cluster {i} accepted different campaigns', ax=axes[i-1], color=palette[i-1]))
    ax.set_xticklabels(['Campaign 1', 'Campaign 2', 'Campaign 3', 'Campaign 4', 'Campaign 5', 'Last campaign'], rotation=0)

plt.tight_layout()

We see that:

The biggest interest in campaign 5 showed Cluster 1 and 2 Campaign acceptance is relatively lower than any campaign The biggest interest in campaign 3 showed Cluster 1 and 2 Campaign acceptance is relatively high than any campaign was relatively higher than any campaign The last campaign was succesfull in all clusters

complains_by_cluster = (sf_df.groupby(by='Cluster')['Complain'].sum()
                                      .divide(sf_df['Cluster'].value_counts())
                                      .multiply(100))

ax = complains_by_cluster.plot(kind='bar', figsize=(18, 8), color=palette[:4],
                               title='Percent of complained customers for the last 2 years in different clusters',
                               ylabel='%', xlabel='Cluster')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.show()

Cluster 1 purchases less, but complains more, thats interesting

The most successful campaigns: 3, the last The least successful campaigns: 1, 2, 4, 5 (0 acceptance in 2 and 5 campaigns) Complain the most

sns.scatterplot(data = sf_df,x=sf_df["Income"], y=sf_df["Expenses"],hue=sf_df["Cluster"])
plt.title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()

cluster1:high income, high spending

cluster2:high income, average spending

cl = ['#FAD3AE', '#855E46', '#FE800F', '#890000']
plt.figure(figsize=(14,8))
sns.countplot(x=sf_df['Cluster'], palette=cl);

# plotting the total amount spent by each cluster
plt.figure()
sns.swarmplot(x=sf_df["Cluster"], y=sf_df["Expenses"], color= "#CBEDDD", alpha=0.5 )
sns.boxenplot(x=sf_df["Cluster"], y=sf_df["Expenses"], palette=cl)
plt.title("Total Amount Spent")
plt.show()

cluster 2 spend more compared to cluster 1

#Plotting the number of deals purchased
plt.figure()
sns.boxenplot(y=sf_df["NumTotalPurchases"],x=sf_df["Cluster"])
plt.title("Number of Deals Purchased")
plt.show()

cluster 1 purchase more compared to cluster 2

Customer profile

plt.figure()

sns.countplot(x=sf_df["Education_Undergraduate"],hue=sf_df["Cluster"])
plt.title("Count Of Education")
plt.xlabel(" Undergraduate Education")
plt.show()

plt.figure()

sns.countplot(x=sf_df["Education_Graduate"],hue=sf_df["Cluster"])
plt.title("Count Of Education")
plt.xlabel(" Graduate Education")
plt.show()

plt.figure()

sns.countplot(x=sf_df["Education_Postgraduate"],hue=sf_df["Cluster"])
plt.title("Count Of Education")
plt.xlabel(" Postgraduate Education")
plt.show()

Majority of our population Have obtained graduate and postgraduate Education

Cluster 2 has relatively lower deals used per customer eventhough their spend is considerbly higher cluster 1 . This shows that Cluster 2 doesn't care as much about deals as compared to Cluster 1.

Clusters High Income highest Spend(Cluster 1): These are the customers that have high income and highest spend per customer. Their deals per customer is also the lowest indicating that they aren't as concerned with deals compared to other clusters. HIgh Income high Spend(Cluster 2): These are the customers that have high income and high spend per customer. Their deals per customer is lower 1.

sns.pairplot(sf_df, vars= ['Kids','Expenses','TotalAcceptedCmp','NumTotalPurchases'])
plt.show();

removed outliers

Age greater than 100 Income above 600,000

So we have the customers with all three education level in each cluster Cluster 0 is mostly consist of graduate and post graduate customers and 4or5 undergraduates Cluster 1 is similar to cluster 0 regarding education level of customers Cluster 0 has most number of graduates and post graduates and very few undergraduates Cluster 1 has least number of postgraduates and third estsmall number of undergraduate than any other cluster

# Now lets findout how  many customers from each cluster has partner and see if we find something interesting
plt.figure(figsize=(12, 8))
sns.countplot(x='Clusters', data=sf_df, hue='Marital_Status_Single')
plt.show()

# Now lets findout how  many customers from each cluster has partner and see if we find something interesting
plt.figure(figsize=(12, 8))
sns.countplot(x='Clusters', data=sf_df, hue='Marital_Status_Taken')
plt.show()

We got see similar trend among each cluster. All clusters consists of customers which have partner and which are single. They all have more number customers having partner in each cluster Cluster 0 has more number of customers with partner as compared to others

# Find out the customers which have kids or tenns in different clusters
plt.figure(figsize=(12, 8))
sns.barplot(x=sf_df["Clusters"], y=sf_df["Kids"])
plt.title("Kids Number vs Clusters", size=15)
plt.show()

Cluster 0 has the customers with most number of kids or teens in household Cluster 1 has the customers with least amount of kids or teens in household

# Now lets findout how  many customers from each cluster has exact number of kids or teens in household
plt.figure(figsize=(12, 8))
sns.countplot(x='Clusters', data=sf_df, hue='Kids')
plt.show()

All the customers from Cluster 0 have kids or teens in household. Most customers have 1 kid, some have 2 and few have 3 children in household Maximum number of customers from cluster 1 have 1 child in household. Hoever there are few with 3 children in household. Cluster 2 largly consist of customers having 1 child in household. some have no children at all in household and remaining have 2 children in household. Cluster 1 is similar to cluster 2. It consist of customers with 1 children and very few with no kids in household

# Now lets findout how  many customers from each cluster have complained
plt.figure(figsize=(12, 8))
sns.countplot(x='Clusters', data=sf_df,hue='Complain')
plt.show()

We already know there really few compaints, and this shouldn't be taken for granted Customers from cluster 1 have no complaints and cluster has only 2 complain

# Lets findout income of customers with in clusters
plt.figure(figsize=(12, 8))
sns.barplot(x=sf_df["Clusters"], y=sf_df["Income"])
plt.title("Income vs Clusters", size=15)
plt.show()

Even tho the number of customers in cluster 2 were very less than cluster 3, still the income of customers within cluste 2 is more than cluster 3. Despite the fact that there are more undergraduates in cluster 2. same goes for cluster 1 and 2. Cluster 1 customer has more income than cluster 2 . Cluster 1 has maximum number of customers than any other cluster.

# It will be interesting to know the income of customers in each clusters depending on the number of children they have in household
plt.figure(figsize=(12, 8))
sns.barplot(x=sf_df["Clusters"], y=sf_df["Income"], hue=df['Kids'])
plt.show()

For cluster 0 customers with no children earn the highest Still customers with 2 or 3 chidren earning slightly more. But the interesting to note is cluster 1 has very few customers with 3 children in household and still they are matching up with other customers. Same goes for the cluster 2. It largely consist of customers with 0 childrenand very few with 2 children but still their income is similar

# Lets see how customers spent money on different products depending on their income
MntColumns= ['Wines', 'Fruits',
       'Meat', 'Fish', 'Sweets',
       'Gold','Expenses']

_, ax1 = plt.subplots(4,2, figsize=(25,22))
plt.suptitle('Income Vs Amount Spent By Clusters')

for i, col in enumerate(MntColumns):
    sns.scatterplot(x='Income', y=col, data=sf_df, ax=ax1[i//2, i%2],hue='Clusters')

plt.show()

Cluster 0 and 1 have high income and have spent more money on different products Cluster 2 and 3 has low income and have spent less money We can see that clusters are overlapping , thats why clusters have customers with similar characteristics

# Barplot to see money spent by different customers on differen products
_, ax1 = plt.subplots(4,2, figsize=(25,22))
for i, col in enumerate(MntColumns):
    sns.barplot(x='Clusters', y=col, data=sf_df, ax=ax1[i//2, i%2])

plt.show()

Cluster 2 has spent more money than any other clusters on all products Cluster 0 is second in spending most money, cluster 2 has spent more money on wines and goldproducts than any other products Cluster 0 has spent less money as compared to 1 and 2 clusters. Still cluster 0 has spent more money on wines and goldproducts than 1 and 3 cluster 3 has spent least money as compared to other clusters

# Let's see if there are any customers withspecific educational background in clusters spending money on specific products
_, ax1 = plt.subplots(4,2, figsize=(25,22))
for i, col in enumerate(MntColumns):
    sns.barplot(x='Clusters', y=col, data=sf_df, ax=ax1[i//2, i%2], hue='Education_Undergraduate')

plt.show()

# Let's see if there are any customers withspecific educational background in clusters spending money on specific products
_, ax1 = plt.subplots(4,2, figsize=(25,22))
for i, col in enumerate(MntColumns):
    sns.barplot(x='Clusters', y=col, data=sf_df, ax=ax1[i//2, i%2], hue='Education_Graduate')

plt.show()

# Let's see if there are any customers withspecific educational background in clusters spending money on specific products
_, ax1 = plt.subplots(4,2, figsize=(25,22))
for i, col in enumerate(MntColumns):
    sns.barplot(x='Clusters', y=col, data=sf_df, ax=ax1[i//2, i%2], hue='Education_Postgraduate')

plt.show()

A very interesting thing we can see is, undergraduates from all clusters are spending more money on sweets, fish, gold and fruits than customers with any other education background

# Let's see if there's any difference between money spent by customers with partner and without partner within different clusters
_, ax1 = plt.subplots(4,2, figsize=(25,22))
for i, col in enumerate(MntColumns):
    sns.barplot(x='Clusters', y=col, data=sf_df, ax=ax1[i//2, i%2], hue='Marital_Status_Single')

plt.show()

# Let's see if there's any difference between money spent by customers with partner and without partner within different clusters
_, ax1 = plt.subplots(4,2, figsize=(25,22))
for i, col in enumerate(MntColumns):
    sns.barplot(x='Clusters', y=col, data=sf_df, ax=ax1[i//2, i%2], hue='Marital_Status_Taken')

plt.show()

Theres not much difference or trend that we can see. All customers within all clusters have similar trend. Money spent by singles and customers with partner is similar , theres not much

# Let's see if there's any difference between money spent by customers with partner and without partner within different clusters
_, ax1 = plt.subplots(4,2, figsize=(25,22))
for i, col in enumerate(MntColumns):
    sns.barplot(x='Clusters', y=col, data=sf_df, ax=ax1[i//2, i%2], hue='Age')

plt.show()

As we saw earlier all the clusters have very few senior citisens and youg adults, still the money spent by them is equal or more than customers from other age groups Senior citizens are more into buying wines and fish

_, ax1 = plt.subplots(4,2, figsize=(25,22))
for i, col in enumerate(MntColumns):
    sns.barplot(x='Clusters', y=col, data=sf_df, ax=ax1[i//2, i%2], hue='Kids')

plt.show()

we can see that customers with 3 children in household are more into buying fuits, meat and sweet products

plt.figure(figsize=(12, 8))
sns.boxenplot(x=sf_df["Clusters"], y=sf_df["NumTotalPurchases"])
plt.title("Purchase Number vs Clusters", size=15)
plt.show()

cluster 3 does the least shopping, followed by cluster 1 Cluster 2 and 0 does the most shopping

plt.figure(figsize=(12, 8))
sns.boxenplot(x=sf_df["Clusters"], y=sf_df["NumDealsPurchases"])
plt.show()

Cluster 2 has the highest number of purchases on discount deals

CUSTOMER PROFILING

Cluster 0

Well Educated
Majority have no Kids
High Income
High Expenditure
High number of purchases. More shopping

Cluster 1

Most number of customers
Graduates & Post Graduates
Has 1 kid or Teen
Highest Income
Above Average expenditure

Cluster 3

Least number of customers
Graduates & post graduates
Has 1 kid or teen
Average Income
Low expenditure
Leastshopping

Cluster 2

Graduates and Postgraduates
Low Income
Has 0 or 1 or 2 children
High expenditure
Highest shopping

sf_df['Education_Undergraduate'].value_counts()

plt.scatter(sf_df['Income'], sf_df['Expenses'], c = sf_df['Cluster'], cmap = cmap, alpha = 0.4)
plt.xlabel('Income')
plt.ylabel('Expenses')

Income and Spending¶ The following scatter plot shows that:

Cluster 3 (light blue) has an average income and does not spend much on our store. Cluster 0 (purple) has a very high income and spends a lot on our store. Cluster 2 (dark blue), on average, has an above average income and a moderate spending. Cluster 1 (pink) have a high income and spend a fair amount on our store.

fig, axs = plt.subplots(ncols = 2, figsize = (15, 5))
sns.boxplot(data = sf_df, x = 'Cluster', y = 'Income', ax = axs[0], palette = palette, flierprops={'marker': 'o'})
sns.boxplot(data = sf_df, x = 'Cluster', y = 'Expenses', ax = axs[1], palette = palette, flierprops={'marker': 'o'})
fig.tight_layout()

Family Dynamics¶ The following bar graphs show that:

Cluster 0 customers usually have one kid.

Cluster 1 customers have 1 and 2 Child.Either a kid and a teen, or only a teen.

Cluster 2 customers have one child. Either a kid and a teen, or only a teen.They are parents

Cluster 3 customers usually have one child.

fig = plt.figure(figsize = (15, 10))
columns = 3
rows = 2
x = ['Kids', 'Kidhome', 'Teenhome', 'Is_Parent', 'Marital_Status_Single','Marital_Status_Taken']
for i in range(0, 6):
    fig.add_subplot(rows, columns, i + 1)
    sns.countplot(data = sf_df, x = x[i], hue = 'Cluster', palette = palette)
fig.tight_layout()

Deals and Purchase Channels The following box plots indicate that:

Cluster 1 makes a lot of visits to the company website, but makes few purchases, some online and some at the store.

Cluster 2 does not make many deal purchases and does not visit the website much. They do make a fair amount of catalogue purchases and many store purchases.

Cluster 3 make few deal purchases. They make few store purchases and catalogue purchases. They make some deals purchses and visit the website quite frequently.They have above averageweb purchases

Cluster 0 make the most deal purchases, web purchases and store purchases and visit the website quite a lot. They make the most purchases out of all clusters.

fig = plt.figure(figsize = (15, 10))
columns = 3
rows = 2
y = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
for i in range(0, 5):
    fig.add_subplot(rows, columns, i + 1)
    sns.boxplot(data = sf_df, x = 'Cluster', y = y[i], palette = palette, flierprops={'marker': 'o'})
    plt.yticks(ticks = plt.yticks()[0], labels = plt.yticks()[0].astype(int))
    plt.ylim(bottom = -1)
plt.show()
fig.tight_layout()

Conclusion¶ We were able to identify 4 clusters and their characteristics. Here is a summary of what we have learned:

Cluster 3 - The Conservative 1-Child Spenders: Cluster 0 represents customers with average incomes who exhibit cautious spending habits.They are often parents with one child. They spend evenly among various categories except for web purchses. They make fewer purchases overall.

Cluster 0 - The Child High-Income Big Spenders: high-income individuals who are avid spenders, allocating a substantial portion of their income to purchases. and their active engagement in online and store purchases.

Cluster 2 - The Moderate Family Spenders: Customers with above-average incomes who exhibit moderate spending habits. They are often parents with one child. They mostly spend on wine. While they make fewer purchases compared to other clusters, they show a preference for web purchses.

Cluster 1 - The Wine Enthusiasts with Teenagers: High-income individual who come to our store mostly for wine. they are often parents of Either one or two children. They make the most purchases online .

from sklearn.metrics import confusion_matrix,classification_report
print("ConfusionMatrix \n",confusion_matrix(kmeans.labels_,pred))
print("classification report \n", classification_report(kmeans.labels_,pred))

ConfusionMatrix 

#kmeans

from sklearn.metrics import confusion_matrix,classification_report
print("ConfusionMatrix \n",confusion_matrix(kmeans.labels_,yhat_kmeans))
print("classification report \n", classification_report(kmeans.labels_,yhat_kmeans))

Recommendations:¶ Customers from cluster 2 and 3 spend the similar amount of money per item but just buy more. I would recommend to add more expensive products for such customers in categories wine and meat in order to increase sells.

Mini-Batch K-Means

from sklearn.cluster import MiniBatchKMeans
#Initiating the MiniBatchKMeans Clustering model
MP = MiniBatchKMeans(n_clusters=4)

# fit model and predict clusters
MP_df = MP.fit_predict(PCA_ds)
PCA_ds["Clusters"] = MP_df
#Adding the Clusters feature to the orignal dataframe.
sf_df["Clusters"]= MP_df

#Plotting countplot of clusters
pl = sns.countplot(x=sf_df["Clusters"])
pl.set_title("Distribution Of The Clusters")
plt.show()

pl = sns.scatterplot(data = sf_df,x=sf_df["Expenses"], y=sf_df["Income"],hue=sf_df["Clusters"])
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()

plt.figure()
pl=sns.swarmplot(x=sf_df["Clusters"], y=sf_df["Expenses"], color= "#CBEDDD", alpha=0.5 )
pl=sns.boxenplot(x=sf_df["Clusters"], y=sf_df["Expenses"])
plt.show()

#Creating a feature to get a sum of accepted promotions
sf_df["AcceptedCmp"] = sf_df["AcceptedCmp1"]+ sf_df["AcceptedCmp2"]+ sf_df["AcceptedCmp3"]+ sf_df["AcceptedCmp4"]+ sf_df["AcceptedCmp5"]
#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=sf_df["AcceptedCmp"],hue=sf_df["Clusters"])
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()

#Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=sf_df["NumDealsPurchases"],x=sf_df["Clusters"])
pl.set_title("Number of Deals Purchased")
plt.show()

Mean Shift

from sklearn.cluster import MeanShift
#Initiating the MeanShift Clustering model
MSP = MeanShift()

# fit model and predict clusters
MSP_df = MSP.fit_predict(PCA_ds)
PCA_ds["Clusters"] = MSP_df

#Adding the Clusters feature to the orignal dataframe.
sf_df["Clusters"]= MSP_df

#Plotting countplot of clusters
pl = sns.countplot(x=sf_df["Clusters"])
pl.set_title("Distribution Of The Clusters")
plt.show()

pl = sns.scatterplot(data = sf_df,x=sf_df["Expenses"], y=sf_df["Income"],hue=sf_df["Clusters"])
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()

plt.figure()
pl=sns.swarmplot(x=sf_df["Clusters"], y=sf_df["Expenses"], color= "#CBEDDD", alpha=0.5 )
pl=sns.boxenplot(x=sf_df["Clusters"], y=sf_df["Expenses"])
plt.show()

#Creating a feature to get a sum of accepted promotions
sf_df["AcceptedCmp"] = sf_df["AcceptedCmp1"]+ sf_df["AcceptedCmp2"]+ sf_df["AcceptedCmp3"]+ sf_df["AcceptedCmp4"]+ sf_df["AcceptedCmp5"]
#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=sf_df["AcceptedCmp"],hue=sf_df["Clusters"])
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()

#Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=sf_df["NumDealsPurchases"],x=sf_df["Clusters"])
pl.set_title("Number of Deals Purchased")
plt.show()

OPTICS

from sklearn.cluster import OPTICS
#Initiating the OPTICS Clustering model
OP = OPTICS(eps=0.8, min_samples=10)

# fit model and predict clusters
OP_df = OP.fit_predict(PCA_ds)
PCA_ds["Clusters"] = OP_df

#Adding the Clusters feature to the orignal dataframe.
sf_df["Clusters"]= OP_df

#Plotting countplot of clusters
pl = sns.countplot(x=sf_df["Clusters"])
pl.set_title("Distribution Of The Clusters")
plt.show()

pl = sns.scatterplot(data = sf_df,x=sf_df["Expenses"], y=sf_df["Income"],hue=sf_df["Clusters"])
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()

plt.figure()
pl=sns.swarmplot(x=sf_df["Clusters"], y=sf_df["Expenses"], color= "#CBEDDD", alpha=0.5 )
pl=sns.boxenplot(x=sf_df["Clusters"], y=sf_df["Expenses"])
plt.show()

#Creating a feature to get a sum of accepted promotions
sf_df["TotalAcceptedCmp"] = sf_df["AcceptedCmp1"]+ sf_df["AcceptedCmp2"]+ sf_df["AcceptedCmp3"]+ sf_df["AcceptedCmp4"]+ sf_df["AcceptedCmp5"]
#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=df["TotalAcceptedCmp"],hue=sf_df["Clusters"])
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()

#Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=sf_df["NumDealsPurchases"],x=sf_df["Clusters"])
pl.set_title("Number of Deals Purchased")
plt.show()
