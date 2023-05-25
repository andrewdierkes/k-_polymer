#!/usr/bin/env python
# coding: utf-8

# In[232]:


import pathlib
from IPython.display import display
import re
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


# In[259]:


#add all predictions
mM_y_predict_list = []

#add all REF value lists
REF_list = []
DES_list = []
DES_DEP_list = []
DEP_list = []

#add all SR values
k_wet_SR_list_final = []
k_dry_SR_list_final = []


#RFU outputs for k+
k_wet_senRFU_final = []
k_wet_refRFU_final = []

k_dry_senRFU_final = []
k_dry_refRFU_final = []

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
            QOS_search = re.compile(r'(?<=QOS Optical Signal: )\d*.[0-9]+')
            REF_search = re.compile(r'(?<=@REF: )[0-9,.]+')
            DES_search = re.compile(r'(?<=@DES: )[0-9,_,a-z].*')
            DEP_search = re.compile(r'(?<=@DEP: )[0-9]')
            
            QOS_value = QOS_search.findall(data)
            REF_value = REF_search.findall(data)
            DES_value = DES_search.findall(data)
            DEP_value = DEP_search.findall(data)
            
   
            #combine assay name and kl245 %
            DES_DEP = ''.join(DEP_value) + '_' + ''.join(DES_value)
            
            DES_list.append(DES_value)
            DEP_list.append(DEP_value)
            REF_list.append(REF_value)
            DES_DEP_list.append(DES_DEP)
            

            k_dry_SR_list = []            
            k_wet_SR_list = []
            
            k_wet_senRFU_list = []
            k_wet_refRFU_list = []
            
            k_dry_senRFU_list = []
            k_dry_refRFU_list = []
            
            #check we have 16 scans, or channel values won't align:
            
            if len(QOS_value) == 16:
         
                SR_3000_Dry = round(float(QOS_value[4]) / float(QOS_value[5]), 4)
                SR_3800_Dry = round(float(QOS_value[6]) / float(QOS_value[7]), 4)

                
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
                fluorescence(k_dry_SR_list, k_dry_SR_list_final, SR_3000_Dry, SR_3800_Dry)
                
                #function for wet RFU
                fluorescence(k_wet_senRFU_list, k_wet_senRFU_final, QOS_value[12], QOS_value[14])
                fluorescence(k_wet_refRFU_list, k_wet_refRFU_final, QOS_value[13], QOS_value[15])

                
                
                #function for dry RFU
                fluorescence(k_dry_senRFU_list, k_dry_senRFU_final, QOS_value[4], QOS_value[6])
                fluorescence(k_dry_refRFU_list, k_dry_refRFU_final, QOS_value[5], QOS_value[7])
               

            #if more than 16 scans taken we must adjust the divison above float(QOS_value[15] etc)    
            else:
                print(f' Length of QOS_value list is {len(QOS_value)}, this does not match target 16 for proper parsing, {print(_filepath)}')


# In[260]:


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


# In[261]:


def delist(args):
    'Taking mutliple lists within a list and returning a single list of values'
    delist = [var for small_list in args for var in small_list]
    return(delist)


# In[262]:


#various data parsing
DES_list_final = delist(DES_list)
REF_list = [float(var) for var in delist(REF_list)]
assay_num = int(len(k_dry_senRFU_final)/2)


# In[263]:


#potassium

#variables to fit in regression
Xkwet = k_wet_SR_list_final
Xkdry = k_dry_SR_list_final

#[0:assay_num] = for first half of data
#[assay_num+1:assay_num*2] for second half

Yk = [np.log10(var) for var in np.repeat(REF_list[0:assay_num],2)]

#call method in regression_data class to generate a regression plot
kwet_regression = regression_data(Xkwet[0:assay_num], Yk[0:assay_num])
kwet_slope, kwet_intercept, kwet_r2, kwet_equation= kwet_regression.regression_function()



#wet scan plotter for 0%
fig, ax = plt.subplots()
plt.scatter(Xkwet[0:assay_num], Yk[0:assay_num], color='r')
ax.grid()
ax.set(xlabel = 'S/R value', ylabel = 'log([K+])', title = 'Linear Regression Model for 0% KL245 wet S/R values vs Log([K+])');

#dry scan plotter for 0%
fig, ax = plt.subplots()
plt.scatter(Xkdry[0:assay_num], Yk[0:assay_num], color='black')
ax.grid()
ax.set(xlabel = 'S/R value', ylabel = 'log([K+])', title = 'Linear Regression Model for 0% KL245 dry S/R values vs Log([K+])');


#wet scan plotter for 1%
fig, ax = plt.subplots()
plt.scatter(Xkwet[assay_num+1:assay_num*2], Yk[assay_num+1:assay_num*2], color='r')
ax.grid()
ax.set(xlabel = 'S/R value', ylabel = 'log([K+])', title = 'Linear Regression Model for 1% KL245 wet S/R values vs Log([K+])');

#dry scan plotter for 1%
fig, ax = plt.subplots()
plt.scatter(Xkdry[assay_num+1:assay_num*2], Yk[assay_num+1:assay_num*2], color='black')
ax.grid()
ax.set(xlabel = 'S/R value', ylabel = 'log([K+])', title = 'Linear Regression Model for 1% KL245 dry S/R values vs Log([K+])');




# In[265]:


assay_num = int(len(k_dry_senRFU_final)/2)
print(f'Data taken from {assay_num} assays')

#combining x and y of the entire dataset... X=S/R & Y=NOVA mM
tuples_xy = list(zip(Xkwet, Yk))


# In[266]:


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

indexer_k = multi_index_DF(DES_DEP_list, tuple_channel_k, assay_num,  'Sample Name', 'Channel Position')
indexer_na = multi_index_DF(DES_DEP_list, tuple_channel_na, assay_num, 'Sample Name', 'Channel Position')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[267]:


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

        sample_name_df = pd.DataFrame(np.repeat(DES_DEP_list, 2), columns = ['Sample Name'])
        
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


# In[268]:


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


# In[269]:


def unique(df_col, chunk=4):
    '''This function will iterate over a df_col and return only unique values using chunk_iterator.. so if you'd like the unique value of something occuring every 3 rows... use 3 as your chunk iterator'''

    offset = 0
    
    number_list = [var for var in range(len(df_col))]
    
    dataset_array = []
    dataset_list = []
    
    while offset < len(number_list):
        i = number_list[offset:chunk+offset]
        _array = df_col.iloc[i].unique()
        
        dataset_array.append(_array)
        
        offset += chunk
    
    for var in dataset_array:
        dataset_list.append(var.tolist())
    
    unique = delist(dataset_list)
    
    return unique


# In[270]:


#find unique names from Assay Name row, slice each name to get rid of the rep number and then pass through unique function
#name_len = length of names we want to slice

def slice_name(df, row, name_len, end_offset):
    '''Function SLICE_NAME slices strings (usually assay name) that are similar in nature (df row). It works to 
    remove the ends (end_offset) which are different (usually rep number) to allow the UNIQUE function to work
    df = dataframe to use
    row = the index of the column you want to slice
    name_len = the length of each assay name, they all should be the same
    end_offset = what position you want to end with'''
    
    assay__name = [var for var in df.iloc[:,row]]

    unique_name = []


    for var in assay__name:
        if len(var) == name_len:
            unique_name.append(var[0:end_offset])

        else:
            print(file)
            print(f'labeling error for assay:{var}, length {len(var)}')
            for count, var2 in enumerate(var):
                print(count, var2)
    
    return pd.DataFrame(unique_name)


# In[272]:


sample_id = slice_name(k_RFU_df,0,15,13)
sample_unique = unique(sample_id.iloc[:,0],10)
print(sample_unique)
kwet_avg, kwet_stdev = chunk_cv(k_RFU_df['K+ Wet Calculated'])
k_wet_sr_avg, k_wet_sr_stdev = chunk_cv(k_RFU_df['K+ Wet S/R'])
k_cv = pd.DataFrame()
k_cv.insert(0, 'assay_name', sample_unique)
k_cv.insert(1,'AVG', kwet_avg)
k_cv.insert(2,'StDev', kwet_stdev)
k_cv.insert(3,'CV', round(((k_cv['StDev']/k_cv['AVG'])*100),4))
k_cv.insert(4, 's/r', k_wet_sr_avg)

display(k_cv)
cv = k_cv.iloc[:,3].mean()


# In[273]:


k_RFU_df.insert(9, '% KL_245', np.repeat(DEP_list,2))


# In[274]:


class multicolor_plotter():
    '''df = dataframe with your information, x_df_pos = column position of your x variables, x_label = x axis label, 
    y_df_pos = column position of your y variables, y_label = y axis label, label_df_pos = the label you want associated with each data point, 
    title = title of chart, chunk = corresponds to how many of the same variable you have & want plotted
    name_png = png naming'''
    
    #imports
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    def __init__(self, df, x_df_pos, x_label, y_df_pos, y_label, label_df_pos, title, chunk):
        self.df = df
        self.x_df_pos = x_df_pos
        self.x_label = x_label
        self.y_df_pos = y_df_pos
        self.y_label = y_label
        self.label_df_pos = label_df_pos
        self.title = title
        self.chunk = chunk
        #self.name_png = name_png 
        
    def plotter(self):
        
        def delist(args):
            'Taking mutliple lists within a list and returning a single list of values'
            delist = [var for small_list in args for var in small_list]
            return(delist)
        
        assay_num = [var for var in range(len(self.df.iloc[:,self.x_df_pos]))]
        
        offset_x = 0
        offset_y = 0
        offset_label = 0
        
        xf = []
        yf = []
        labelf = []
        
        while offset_x < len(assay_num):
            i_x = assay_num[offset_x:offset_x+self.chunk]
            x = self.df.iloc[i_x, self.x_df_pos]
            xf.append(x)
            
            offset_x += self.chunk
        
        while offset_y < len(assay_num):
            i_y = assay_num[offset_y:offset_y+self.chunk]
            y = self.df.iloc[i_y, self.y_df_pos]
            yf.append(y)
            
            offset_y += self.chunk
            
        while offset_label < len(assay_num):
            i_label = assay_num[offset_label:offset_label+self.chunk]
            label = self.df.iloc[i_label, self.label_df_pos]
            labelf.append(label)
            
            offset_label += self.chunk
        
        label_unique = []
        for var in labelf:
            labelu = var.unique()
            label_list = labelu.tolist()
            label_unique.append(label_list)
        
        labeld = delist(label_unique)
        
        colors = cm.rainbow(np.linspace(0,100, (len(labeld))))
        
        xylc_zip = zip(xf,yf, labeld, colors)
        
        fig, ax = plt.subplots(figsize=(6,4))
        for x, y, l, c in xylc_zip:
            plt.scatter(x,y, color = c, label = l)
            ax.grid()
            ax.legend()
            ax.set(xlabel = self.x_label, ylabel = self.y_label, title = self.title)
            plt.legend(bbox_to_anchor =(1.1, 1))


# In[275]:


graph = multicolor_plotter(k_RFU_df, 1, 'log([k+])', 2, 'wet s/r', 9, 'SR 3000 & 3800 comparing 0% & 1% KL245', 40)
graph.plotter()


# In[ ]:




