# First step, import the arff files, converted to numpy and pandas data frames:
import numpy as np
import pandas as pd
from scipy.io import arff

data1 = arff.loadarff('Autism-Child-Data.arff')
np_data1 = pd.DataFrame(data1[0]).values
pd_data1 = pd.DataFrame(data1[0])
DataSet = pd.read_csv('ASD1_data.csv')
#First step to cleaning, identify the missing data via a dot value count for each attribute

Example: pd_data1['ethnicity'].value_counts()

# For the child dataset, 14% of ethnicity data was labeled with a ‘?’,  small enough to replace via the mode technique. ‘White European’ ethnicity was identified as being most frequent, thus, the ethnicity labeled ‘?’ was replaced with it.

# For the adolescent dataset, 6% of ethnicity data was labeled with a ‘?’,  small enough to replace via the mode technique. ‘White European’ ethnicity was identified as being most frequent, thus, the ethnicity labeled ‘?’ was replaced with it.

# For the adult dataset, 13% of ethnicity data was labeled with a ‘?’,  small enough to replace via the mode technique. ‘White European’ ethnicity was identified as being most frequent, thus, the ethnicity labeled ‘?’ was replaced with it.

# Code used to  replace missing labels:


DataSet['ethnicity']=DataSet['ethnicity'].replace(b'?',b'White-European')

# The ten behavioral features identified via the questionnaire were designed differently for each group. It was realized, the adult questionnaire was answered by the patient. Therefore, the questions were unchanged.

# The questionnaire for the child and adolescent group was taken by a third-party individual and had six similar questions. Before combining all three datasets,  6 ‘common’ columns were created for the similar questions for each group. The remaining four were left as is.


# So that each row of questions for each group was accounted for in the dataset, additional columns were created and assigned a -1 value to not affect the results for the actual values of the 10 question for each group. The code to replace the column names , values when applicable and to combine all the created columns to the dataset is as follows:

p1=pd_data1[['A2_Score','A3_Score','A5_Score','A6_Score','A8_Score','A10_Score']]

p1=p1.rename(columns={'A2_Score':'A2_common','A3_Score':'A3_common','A5_Score':'A5_common','A6_Score':'A6_common','A8_Score':'A8_common','A10_Score':'A10_common'})

p2=pd_data1[['A1_Score','A4_Score','A7_Score','A9_Score']]

p2=p2.rename(columns={'A1_Score':'A1_child','A4_Score':'A4_child','A7_Score':'A7_child','A9_Score':'A9_child'})

p3=pd.DataFrame(np.full((len(pd_data1),4),1),columns=['Adolescent_only_A1','Adolescent_only_A4','Adolescent_only_A7','Adolescent_only_A9'])

p4=pd.DataFrame(np.full((len(pd_data1),10),1),columns=['Adult_A1','Adult_A2','Adult_A3','Adult_A4','Adult_A5','Adult_A6','Adult_A7','Adult_A8','Adult_A9','Adult_A10'])

pd_data1=pd.concat([p1,p2,p3,p4,pd_data1.drop(pd_data1.columns[0:10],axis=1)],axis=1)

# After missing values were replaced and columns regarding the questionnaire were added and combined, all three data sets were combined imputing the following code:

DataSet=pd.concat([data1,data2,data3],axis=0,ignore_index=True)

# To find the any values labeled ‘NaN’, a code was imputed to identify and replace the missing value via the mean technique of the attribute.

# The code to identify ‘NaN’ values:

DataSet.isna().sum()

# 1% in the ‘age’ attribute was valued as ‘NaN’ and was replaced.

DataSet['age']=DataSet['age'].fillna(DataSet['age'].mean())

# Converting Categorical Features
# Categorical features such as ‘ethnicity’ and ‘country of residence’ were encoded to dummy variables. If not imputed, the models would not have been able to directly take in those features as inputs. The following code was imputed to convert.

dummy=pd.get_dummies(DataSet['contry_of_res'],prefix='country')

dummy1=pd.get_dummies(DataSet['ethnicity'],prefix='ethnicity')

DataSet=DataSet.drop(['contry_of_res','ethnicity'],axis=1)

DataSet=pd.concat([DataSet,dummy,dummy1],axis=1)

DataSet=DataSet.reset_index(drop=True)




# Removing irrelevant attribute columns:

# The attribute ‘relation’ pertaining to who completed the test and age description were removed due to the group agreeing the value did not have relevancy to our results. Especially age description considering the age group is identified via the questionnaire.

DataSet=DataSet.drop(['relation','age_desc'],axis=1)

# To replace the remaining binary or boolean attribute values to numeric:

DataSet=DataSet.replace(to_replace=["b'f'","b'm'","b'no'","b'yes'","b'1'","b'0'","b'NO'","b'YES'"],value=[1,0,0,1,1,0,0,1])

DataSet.replace(to_replace=["b'YES'","b'NO'"],value=[1,0])

# Dataset attribute values were converted to float then converted to a CSV file:

DataSet=DataSet.astype(float)
DataSet.to_csv('ASD_data.csv',index=False)
