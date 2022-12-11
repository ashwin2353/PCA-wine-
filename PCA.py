# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 00:03:43 2022

@author: ashwi
"""
import pandas as pd
df = pd.read_csv("wine.csv")

df.shape
df.dtypes
df.head()
#=============================================================
# checking there is any null values
df.isnull().sum()

# chaking there is any duplicated rows and columns 
df.duplicated()
(df.duplicated()).sum() # there is no duplicated rows 

df.columns.duplicated()
(df.columns.duplicated()).sum() # there is no duplicated columns

#==============================================================
import seaborn as sns 
sns.pairplot(df)

#=============================================================
# histogram 
df["Type"].hist()
df["Alcohol"].hist()
df["Malic"].hist()
df["Ash"].hist()
df["Alcalinity"].hist()
df["Magnesium"].hist()
df["Phenols"].hist()
df["Flavanoids"].hist()
df["Nonflavanoids"].hist()
df["Proanthocyanins"].hist()
df["Color"].hist()
df["Hue"].hist()
df["Dilution"].hist()
df["Proline"].hist()
#=============================================================
# scatter plot
df.plot.scatter(x =["Type"],y = ["Alcohol"] ,color = "black")
df.plot.scatter(x =["Malic"],y = ["Malic"] ,color = "black")
df.plot.scatter(x =["Malic"],y = ["Ash"] ,color = "black")
df.plot.scatter(x =["Ash"],y = ["Alcalinity"] ,color = "black")
df.plot.scatter(x =["Alcalinity"],y = ["Magnesium"] ,color = "black")
df.plot.scatter(x =["Phenols"],y = ["Ash"] ,color = "black")
df.plot.scatter(x =["Magnesium"],y = ["Flavanoids"] ,color = "black")
df.plot.scatter(x =["Flavanoids"],y = ["Nonflavanoids"] ,color = "black")
df.plot.scatter(x =["Color"],y = ["Proanthocyanins"] ,color = "black")
df.plot.scatter(x =["Hue"],y = ["Nonflavanoids"] ,color = "black")
df.plot.scatter(x =["Type"],y = ["Alcohol"] ,color = "black")
df.plot.scatter(x =["Nonflavanoids"],y = ["Proline"] ,color = "black")
df.plot.scatter(x =["Magnesium"],y = ["Hue"] ,color = "black")
#==========================================================================
# boxplot
import matplotlib.pyplot as plt
from scipy.stats import skew
import seaborn as sns
for col in df:
    print(col)
    print(skew(df[col]))
    plt.figure()
    sns.boxplot(df[col])
    plt.show()

list(df)

import matplotlib.pyplot as plt
def plot_boxplot(df,ft):
    df.boxplot(column=[ft])
    plt.grid(False)
    plt.show()

plot_boxplot(df, "Alcohol")
plot_boxplot(df, "Malic")
plot_boxplot(df, "Ash")
plot_boxplot(df, "Alcalinity")
plot_boxplot(df, "Magnesium")
plot_boxplot(df, "Phenols")
plot_boxplot(df, "Flavanoids")
plot_boxplot(df, "Nonflavanoids")
plot_boxplot(df, "Proanthocyanins")
plot_boxplot(df, "Color")
plot_boxplot(df, "Hue")
plot_boxplot(df, "Dilution")
plot_boxplot(df, "Proline")



def outliers(df,ft):
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)
    IQR = Q3-Q1
    
    lower_bound = Q1-1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    ls = df.index[(df[ft]<lower_bound) | (df[ft] > upper_bound)]
    return ls



index_list = []
for feature in ["Alcohol","Malic","Ash","Alcalinity","Magnesium","Phenols","Flavanoids","Nonflavanoids","Proanthocyanins","Color","Hue","Dilution","Dilution"]:
    index_list.extend(outliers(df,feature))


index_list
 
def remove(df,ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df

df_cleaned = remove(df,index_list)
df_cleaned.shape
# outliers are removed 
#=============================================================================================
# divide X variables
X = df.iloc[:,:]

#==============================================================================================
# Standerdising the data
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X = SS.fit_transform(X)
X = pd.DataFrame(X)
X
#==============================================================================================
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA()
pcs = pca.fit_transform(X)
d2 = pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)

new_data = pd.DataFrame(data=pcs,columns=["P1","P2","P3","P4","P5","P6","P7","P8","P9","P10","P11","P12","P13","P14"])
new_data 

d1 ={"var":pca.explained_variance_ratio_,"PCnames":["P1","P2","P3","P4","P5","P6","P7","P8","P9","P10","P11","P12","P13","P14"]}
d1

t1 = pd.DataFrame(d1)
t1

import seaborn as sns
sns.barplot(x="PCnames",y="var",data=t1)
# Therefore from the above new_data set we get to now that the maximum information is compressed in 1st three PCs so we have to take that three PCs apply clustering on it

#===============================================================================================
# Clustering

new_data.shape
new_data.head()

df1 =new_data.iloc[:,0:3]
df1

%matplotlib qt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.Figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2])
plt.show()
#########################   Hierarchical clustering  ##############################
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="complete")
Y = cluster.fit_predict(df1)
Y = pd.DataFrame(Y,columns=["cluster"])
Y.value_counts()
new_data_1 = pd.concat([df1,Y],axis=1)
new_data_1 

# now we can apply this new_data to any classifier techniques for making model

#===================================================================================
###################################   KMeans  #######################################
from sklearn.cluster import KMeans
km = KMeans()

inertia = []
for i in range(1,11):
    km = KMeans(n_clusters=i,random_state=(0))
    km.fit(df1)
    inertia.append(km.inertia_)

print(inertia)

%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.plot(range(1,11),inertia)
plt.title("Elbow Method")
plt.xlabel("No clusters")
plt.ylabel("inertia")
plt.show()


# scree plot
import seaborn as sns
d1 = {"kvalue":range(1,11),"inertiavalues":inertia}
d2 = pd.DataFrame(d1)
sns.barplot(x="kvalue",y="inertiavalues",data= d2,)

# therefore by seing the Elbow mehtod and screen plot i have decided that 3 clusters the best for this data set

KM = KMeans(n_clusters=3, n_init=30)
Y = KM.fit_predict(df1)
Y
Y = pd.DataFrame(Y,columns=["cluster1"])
Y.value_counts()

new_data = pd.concat([df1,Y],axis=1)
new_data 

# now we can apply this new_data to any classifier techniques for making model

# therefore K-Means are prividing better results

















