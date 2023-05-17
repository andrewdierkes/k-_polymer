#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pathlib
from IPython.display import display
import re
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


# In[149]:


#add all predictions
mM_y_predict_list = []

#add all REF value lists
REF_list = []
na_list = []

#add all DES values
DES_list = []

#add all SR values
k_wet_SR_list_final = []
k_dry_SR_list_final = []
na_wet_SR_list_final = []
na_dry_SR_list_final = []

#RFU outputs for k+
k_wet_senRFU_final = []
k_wet_refRFU_final = []

k_dry_senRFU_final = []
k_dry_refRFU_final = []

#RFU outputs for na+
na_wet_senRFU_final = []
na_wet_refRFU_final = []

na_dry_senRFU_final = []
na_dry_refRFU_final = []


#path is in current working directory
_path = pathlib.Path.cwd()
#print(_path)

#for each file in the iterdir search for TSV
for _filepath in _path.iterdir():
    #print(_filepath)
    
    if _filepath.suffix != '.tsv':
        #print(_filepath)
        continue
    
    elif _filepath.suffix == '.tsv':
        with open(_filepath, 'rt') as file:
            data = file.read()
            
            
            
            #create a search parameter with digit as d as many digits then a decimal with any number behind it 1 or more times
            QOS_search = re.compile(r'QOS Optical Signal:.(\d*.[0-9]+)')
            REF_search = re.compile(r'@REF:.*')
            DES_search = re.compile(r'@DES:.*')
            Na_search = re.compile(r'@STR:.*')
            
            QOS_value = re.findall(QOS_search, data)
            REF_value = re.findall(REF_search, data)
            DES_value = re.findall(DES_search, data)
            Na_value = re.findall(Na_search, data)
            

            na_dry_SR_list = []
            k_dry_SR_list = []            
            na_wet_SR_list = []
            k_wet_SR_list = []
            
            k_wet_senRFU_list = []
            k_wet_refRFU_list = []
            
            k_dry_senRFU_list = []
            k_dry_refRFU_list = []
            
            na_wet_senRFU_list = []
            na_wet_refRFU_list = []
            
            na_dry_senRFU_list = []
            na_dry_refRFU_list = []
            
            #check we have 16 scans, or channel values won't align:
            
            if len(QOS_value) == 16:
                
                SR_1400_Dry = round(float(QOS_value[0]) / float(QOS_value[1]), 4)
                SR_2200_Dry = round(float(QOS_value[2]) / float(QOS_value[3]), 4)
         
                SR_3000_Dry = round(float(QOS_value[4]) / float(QOS_value[5]), 4)
                SR_3800_Dry = round(float(QOS_value[6]) / float(QOS_value[7]), 4)
                
                
                SR_1400_Wet = round(float(QOS_value[8]) / float(QOS_value[9]), 4)
                SR_2200_Wet = round(float(QOS_value[10]) / float(QOS_value[11]), 4)
                
                SR_3000_Wet = round(float(QOS_value[12]) / float(QOS_value[13]), 4)
                SR_3800_Wet = round(float(QOS_value[14]) / float(QOS_value[15]), 4)
                                          
                #print(k_wet_SR_list)
                
                def fluorescence(file_list, dataset_list, arg1, arg2):
                    file_list.append(arg1)
                    file_list.append(arg2)
                    
                    for var in file_list:
                        dataset_list.append(var)
                    return dataset_list
                
                
                #function for S/R
                fluorescence(k_wet_SR_list, k_wet_SR_list_final, SR_3000_Wet, SR_3800_Wet)
                fluorescence(na_wet_SR_list, na_wet_SR_list_final, SR_1400_Wet, SR_2200_Wet)
                fluorescence(k_dry_SR_list, k_dry_SR_list_final, SR_3000_Dry, SR_3800_Dry)
                fluorescence(na_dry_SR_list, na_dry_SR_list_final, SR_1400_Dry, SR_2200_Dry)
                
                #function for wet RFU
                fluorescence(k_wet_senRFU_list, k_wet_senRFU_final, QOS_value[12], QOS_value[14])
                fluorescence(k_wet_refRFU_list, k_wet_refRFU_final, QOS_value[13], QOS_value[15])
                
                fluorescence(na_wet_senRFU_list, na_wet_senRFU_final, QOS_value[8], QOS_value[10])
                fluorescence(na_wet_refRFU_list, na_wet_refRFU_final, QOS_value[9], QOS_value[11])
                
                
                #function for dry RFU
                fluorescence(k_dry_senRFU_list, k_dry_senRFU_final, QOS_value[4], QOS_value[6])
                fluorescence(k_dry_refRFU_list, k_dry_refRFU_final, QOS_value[5], QOS_value[7])
                
                fluorescence(na_dry_senRFU_list, na_dry_senRFU_final, QOS_value[0], QOS_value[2])
                fluorescence(na_dry_refRFU_list, na_dry_refRFU_final, QOS_value[1], QOS_value[3])
                
                #print(channel_assign)
                #search for this parameter in each string from REF_value & sub with nothing
                REF_removal = re.compile(r'@REF:.')
                String_REF = [REF_removal.sub('', string) for string in REF_value]
                
                DES_removal = re.compile(r'@DES:.')
                String_DES = [DES_removal.sub('', string) for string in DES_value]
                
                STR_removal = re.compile(r'@STR:.')
                String_na = [STR_removal.sub('', string) for string in Na_value]
                
                #Add REF to list outside loop
                REF_list.append(String_REF)
                
                DES_list.append(String_DES)
                
                na_list.append(String_na)

            #if more than 16 scans taken we must adjust the divison above float(QOS_value[15] etc)    
            else:
                print(f' Length of QOS_value list is {len(QOS_value)}, this does not match target 16 for proper parsing, {print(_filepath)}')


# In[150]:


#create class for linear regression data
class regression_data():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def regression_function(self):
        '''Must convert X into np.array'''
        regress_x = np.array(self.x).reshape(-1,1)
        model = LinearRegression().fit(regress_x, self.y)
        r2 = round(model.score(regress_x, self.y),4)
        intercept = round(model.intercept_, 4)
        slope = model.coef_
        equation = f'y={slope}x+{intercept}, r2 = {r2}'
        
        return slope, intercept, r2, equation


# In[151]:


def delist(args):
    'Taking mutliple lists within a list and returning a single list of values'
    delist = [var for small_list in args for var in small_list]
    return(delist)


# In[193]:


DES_list_final = delist(DES_list)
capital = re.compile(r'N')
#print(DES_list_final)

for var in DES_list_final:
 #   print(var)
    if capital.search(var):
        fix = capital.sub('n', var)
        
REF_final_list = [float(var) for var in delist(REF_list)]


# In[121]:


#potassium

#variables to fit in regression
Xkwet = k_wet_SR_list_final
Xkdry = k_dry_SR_list_final

Yk = [np.log10(var) for var in np.repeat(REF_final_list, 2)]

#call method in regression_data class to generate a regression plot
kwet_regression = regression_data(Xkwet, Yk)
kwet_slope, kwet_intercept, kwet_r2, kwet_equation= kwet_regression.regression_function()


#b = Regression_Function(Xkdry, Yk)
#print(slope_kwet, intercept_kwet, equation)


#wet scan plotter
fig, ax = plt.subplots()
plt.scatter(Xkwet, Yk, color='r')
ax.grid()
ax.set(xlabel = 'S/R value', ylabel = 'log([K+])', title = 'Linear Regression Model for wet S/R values vs Log([K+])');

#dry scan plotter
fig, ax = plt.subplots()
plt.scatter(Xkdry, Yk, color='black')
ax.grid()
ax.set(xlabel = 'S/R value', ylabel = 'log([K+])', title = 'Linear Regression Model for dry S/R values vs Log([K+])');



# In[116]:


DES_final_list = delist(DES_list)
REF_final_list = [float(var) for var in delist(REF_list)]
na_mm_final_list = [float(var) for var in delist(na_list)]

#Linear Regression Model

#variables to fit in regression
Xnawet = na_wet_SR_list_final
Xnadry = na_dry_SR_list_final

Yna = [np.log10(var) for var in np.repeat(na_mm_final_list, 2)]

nawet_regression = regression_data(Xnawet, Yna)
nawet_slope, nawet_intercept, nawet_r2, nawet_equation = nawet_regression.regression_function()

fig, ax = plt.subplots()
plt.scatter(Xnawet, Yna)
ax.grid()
ax.set(xlabel='S/R Value', ylabel='log([Na+])', title= 'Linear Regression Model for wet S/R values vs Log([Na+])')

fig, ax = plt.subplots()
plt.scatter(Xnadry, Yna)
ax.grid()
ax.set(xlabel='S/R Value', ylabel='log([Na+])', title= 'Linear Regression Model for dry S/R values vs Log([Na+])');


# In[62]:


assay_num = int(len(k_dry_senRFU_final)/2)
print(f'Data taken from {assay_num} assays')

#combining x and y of the entire dataset... X=S/R & Y=NOVA mM
tuples_xy = list(zip(Xkwet, Yk))


# In[63]:


#multi-index for both df


tuple_channel_na = (1400, 2200)
tuple_channel_k = (3000, 3800)

def multi_index_DF(outer_index, inner_index, assay_num, outer_index_name, inner_index_name):
        'This provides a way to index a pandas dataframe with multiple indexes. You should use outer_index for the largest grouping & inner for labeling discrete datapoints.'
        #n = the number of inner values you want for each outer_index value (IE tuple_channel value)
        n = len(inner_index)
        #match outer_index values to inner_index
        outer_index_list = list(np.repeat(outer_index, n))
        
        #makes len of list = num assays
        inner_index_list = tuple(inner_index*assay_num)
        
        #make a list of tuples out of the first two parameters
        list_tuple_for_index = list(zip(outer_index_list, inner_index_list))
        
        #generate multi index and assign index column categories
        return pd.MultiIndex.from_tuples(list_tuple_for_index, names = [outer_index_name, inner_index_name]) 

indexer_k = multi_index_DF(DES_final_list, tuple_channel_k, assay_num,  'Sample Name', 'Channel Position')
indexer_na = multi_index_DF(DES_final_list, tuple_channel_na, assay_num, 'Sample Name', 'Channel Position')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[135]:


#K dry and wet RFU OUTPUTS
def floater(arg):
    'To convert a list of strings into a list of floats'
    floater_list = [float(var) for var in arg]
    return floater_list

try: 
    if len(k_dry_senRFU_final) == (assay_num*2):
        
        k_dry_senRFU_df = pd.DataFrame(floater(k_dry_senRFU_final), columns = ['K+ Dry Sensor RFU'])

        k_dry_refRFU_df = pd.DataFrame(floater(k_dry_refRFU_final), columns = ['K+ Dry Reference RFU'])

        k_wet_senRFU_df = pd.DataFrame(floater(k_wet_senRFU_final), columns = ['K+ Wet Sensor RFU'])

        k_wet_refRFU_df = pd.DataFrame(floater(k_wet_refRFU_final), columns = ['K+ Wet Reference RFU'])
        
        k_wet_mM_df = pd.DataFrame(Yk, columns = ['log([K+])'])

        sample_name_df = pd.DataFrame(np.repeat(DES_final_list, 2), columns = ['Sample Name'])
        
        k_RFU_df = pd.concat([sample_name_df, k_wet_mM_df, k_dry_senRFU_df, k_dry_refRFU_df, k_wet_senRFU_df, k_wet_refRFU_df], axis=1)
        
        
        
        k_RFU_df.insert(2,'K+ Dry S/R', k_RFU_df['K+ Dry Sensor RFU'] / k_RFU_df['K+ Dry Reference RFU'])
        k_RFU_df.insert(2,'K+ Wet S/R', k_RFU_df['K+ Wet Sensor RFU'] / k_RFU_df['K+ Wet Reference RFU'])
        k_RFU_df.insert(2, 'K+ Wet Calculated', (k_RFU_df['K+ Wet S/R']*kwet_slope + kwet_intercept))
        
        
        k_RFU_df.set_index(indexer_k, inplace=True)
        k_RFU_df.sort_index(inplace=True)
        
        display(k_RFU_df)
        k_RFU_df.to_excel('k_50.xlsx')
except:
    print(f'something is wonky with your lists, check the length of all RFU_final lists should be equal to {assay_num}')


# In[189]:


def chunk_cv(df_col, chunk=10):
    '''This function will iterate over a df_col and return the average and stdev, using chunk_iterator.. so if you'd like the average & stdev values occuring every 3 rows... use 3 as your chunk iterator'''

    offset_mean = 0
    offset_stdev = 0
    
    number_list = [var for var in range(len(df_col))]
    
    dataset_average = []
    dataset_stdev = []
    
    
    while offset_mean < len(number_list):
        i_mean = number_list[offset_mean:chunk+offset_mean]
        average = df_col.iloc[i_mean].mean(axis=0)
        
        dataset_average.append(average)
        #dataset_array.append(_array)
        
        offset_mean += chunk
    
    while offset_stdev < len(number_list):
        i_stdev = number_list[offset_stdev:chunk+offset_stdev]
        stdev = df_col.iloc[i_stdev].std(ddof=1)
        
        dataset_stdev.append(stdev)
        
        offset_stdev += chunk
    
    return dataset_average, dataset_stdev


# In[190]:


kwet_avg, kwet_stdev = chunk_cv(k_RFU_df['K+ Wet Calculated'])
k_cv = pd.DataFrame()
k_cv.insert(0,'AVG', kwet_avg)
k_cv.insert(1,'StDev', kwet_stdev)
k_cv.insert(2,'CV', round(((k_cv['StDev']/k_cv['AVG'])*100),4))
display(k_cv)



# In[136]:


#Sodium dry and wet RFU outputs

try: 
    if len(na_dry_senRFU_final) == (assay_num*2):
        
        na_dry_senRFU_df = pd.DataFrame(floater(na_dry_senRFU_final), columns = ['Na+ Dry Sensor RFU'])

        na_dry_refRFU_df = pd.DataFrame(floater(na_dry_refRFU_final), columns = ['Na+ Dry Reference RFU'])

        na_wet_senRFU_df = pd.DataFrame(floater(na_wet_senRFU_final), columns = ['Na+ Wet Sensor RFU'])

        na_wet_refRFU_df = pd.DataFrame(floater(na_wet_refRFU_final), columns = ['Na+ Wet Reference RFU'])

        na_wet_mM_df = pd.DataFrame(Yna, columns = ['log([Na+])'])

        sample_name_df = pd.DataFrame(np.repeat(DES_final_list, 2), columns = ['Sample Name'])
        
       # na_RFU_df = pd.concat([sample_name_df, k_wet_mM_df, k_dry_senRFU_df, k_dry_refRFU_df, k_wet_senRFU_df, k_wet_refRFU_df], axis=1)
        na_RFU_df = pd.concat([sample_name_df, na_wet_mM_df, na_dry_senRFU_df, na_dry_refRFU_df, na_wet_senRFU_df, na_wet_refRFU_df], axis=1)
        
        na_RFU_df.insert(2,'Na+ Dry S/R', na_RFU_df['Na+ Dry Sensor RFU'] / na_RFU_df['Na+ Dry Reference RFU'])
        na_RFU_df.insert(2,'Na+ Wet S/R', na_RFU_df['Na+ Wet Sensor RFU'] / na_RFU_df['Na+ Wet Reference RFU'])
        na_RFU_df.insert(2, 'Na+ Wet Calculated', (na_RFU_df['Na+ Wet S/R']*nawet_slope + nawet_intercept))
        
        na_RFU_df['Na+ Dry S/R'] = na_RFU_df['Na+ Dry Sensor RFU'] / na_RFU_df['Na+ Dry Reference RFU']
        na_RFU_df['Na+ Wet S/R'] = na_RFU_df['Na+ Wet Sensor RFU'] / na_RFU_df['Na+ Wet Reference RFU']

        na_RFU_df.set_index(indexer_na, inplace=True)
        na_RFU_df.sort_index(inplace=True)
        display(na_RFU_df)
        na_RFU_df.to_excel('na_50.xlsx')
        
except:
    print(f'something is wonky with your lists, check the length of all RFU_final lists should be equal to {assay_num}')


# In[ ]:


nawet_avg, kwet_stdev = chunk_cv(k_RFU_df['K+ Wet Calculated'])
k_cv = pd.DataFrame()
k_cv.insert(0,'AVG', kwet_avg)
k_cv.insert(1,'StDev', kwet_stdev)
k_cv.insert(2,'CV', round(((k_cv['StDev']/k_cv['AVG'])*100),4))
display(k_cv)



# In[ ]:





# In[ ]:





# In[40]:


#plotting different colors for each assay


##USE PD DATAFRAME AFTER INDEX SORTED AND ALSO NEED TO ADD K+ to IT AS WELL AS CALCV...


chunk = 2
offset_x = 0
offset_y = 0

Xk_dataset = []
Yk_dataset = []
assay_num_list = [var for var in range(assay_num*2)]



#generate lists of S/R for each assay
while offset_x < len(assay_num_list):
    i_x = assay_num_list[(offset_x+chunk)-1] #bc index starts at 0
    #Xkwet_assay = k_RFU_df['K+ Wet S/R'].iloc[offset_x:i_x+1] #3000 & 3800 combined
    Xkwet_assay = Xkwet[offset_x:i_x+1]
    Xk_dataset.append(Xkwet_assay)
    offset_x += chunk

while offset_y < len(assay_num_list):
    
    i_y = assay_num_list[(offset_y +chunk)-1] #bc index starts at 0
    #Ykwet_assay = k_RFU_df['log([K+])'].iloc[offset_y:i_y+1] #3000 & 3800 combined
    Ykwet_assay = Yk[offset_y:i_y+1]
    Yk_dataset.append(Ykwet_assay)
    offset_y += chunk


# In[158]:


#x will be duplicated to be as long as Y
get_ipython().run_line_magic('matplotlib', 'inline')

X = REF_final_list
Ys = np.array(Xk_dataset)

#in this case x would be the Y mM conc... 
nCols = len(X)
nRows = Ys.shape[0]
#print(nRows)

#this will you X amount of colors.. last digit is = to num colors
colors = cm.rainbow(np.linspace(0, 1, (assay_num*2)))


#create a color for i in range of assay channels
cs = [colors[i] for i in range(assay_num*2)]

sample = np.repeat(DES_final_list, 2)

legend = list(zip(sample, cs))

#print(legend)
#repeat mM for channel duplicates
Xs = np.repeat(X, 2)

#flatten Y out into length of assay_num*2 (all channels)
Yflat = Ys.flatten()

#Length of sample, Xs, Ys.flatten & colors should all be the same
    #print(len(sample))
    #print(len(Xs))
    #print(len(colors))
    #print(len(Ys.flatten()))

fig, ax = plt.subplots()
for x, y, c, lb in zip(Xs, Yflat, colors, sample):
    plt.scatter(x, y, color = c, label = lb)
    plt.legend(bbox_to_anchor=(1, 1.05))
plt.grid()
ax.set(xlabel = 'mM [K+]', ylabel = 'S/R value', title = 'Wet SR values vs Log([K+])');

