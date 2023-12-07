import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn import metrics
from datetime import datetime
from llm_openai import model_test_observation
from llm_opensource import model_test_observation_opensource
import matplotlib.pyplot as plt
write_output_path = './output/'

# Read excel sheet specific for each test
def read_excel_sheet(file_upload):
    sheet_dict = pd.read_excel(file_upload)
    return sheet_dict

# function to get output of df.info() that is sent to a print to a dataframe
def get_info_to_df(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    lines = buffer.getvalue().splitlines()
    df = (pd.DataFrame([x.split() for x in lines[5:-2]], columns=lines[3].split())
        .drop('Count',axis=1)
        .rename(columns={'Non-Null':'Non-Null Count'}))
    return df
#=======================================#

# function to calculate metrics for data1 and data2
def calculate_stats(data1,data2):
    # data1=vintage1
    df1=data1.describe(include='all').T
    df1=df1[["count","mean","std","min","max"]]
    df2=pd.DataFrame(index=list(data1.columns),columns=["count_missing"])
    for i,v in enumerate(list(data1.columns)):
        df2.loc[v,"count_missing"]=data1[v].isna().sum()
    df_vintage1=pd.concat([df1,df2],axis=1)
    df_vintage1["%missing"]=(df_vintage1["count_missing"]*100)/df_vintage1["count"]
    df_vintage1=df_vintage1[["count","count_missing","%missing","mean","std","min","max"]]
    df_vintage1["unique_val"]=np.NaN
    for i,v in enumerate(list(data1.columns)):
        if data1[v].dtype=='O':
            df_vintage1.loc[v,"unique_val"]=data1[v].nunique()
    df_vintage1.rename(columns={"count":"count_dev","count_missing":"count_missing_dev","%missing":"%missing_dev",
                                "mean":"mean_dev","std":"std_dev","min":"min_dev","max":"max_dev","unique_val":"unique_val_dev"} ,inplace=True)


    # data2=vintage2
    df3=data2.describe(include='all').T
    df3=df3[["count","mean","std","min","max"]]
    df4=pd.DataFrame(index=list(data2.columns),columns=["count_missing"])
    for i,v in enumerate(list(data2.columns)):
        df4.loc[v,"count_missing"]=data2[v].isna().sum()
    df_vintage2=pd.concat([df3,df4],axis=1)
    df_vintage2["%missing"]=(df_vintage2["count_missing"]*100)/df_vintage2["count"]
    df_vintage2=df_vintage2[["count","count_missing","%missing","mean","std","min","max"]]
    df_vintage2["unique_val"]=np.NaN
    for i,v in enumerate(list(data2.columns)):
        if data2[v].dtype=='O':
            df_vintage2.loc[v,"unique_val"]=data2[v].nunique()
    df_vintage2.rename(columns={"count":"count_val","count_missing":"count_missing_val","%missing":"%missing_val",
                                "mean":"mean_val","std":"std_val","min":"min_val","max":"max_val","unique_val":"unique_val_val"} ,inplace=True)
    df=pd.concat([df_vintage1,df_vintage2],axis=1)
    col_list=[]
    for i,v in enumerate(list(df_vintage1.columns)): 
        for j,k in enumerate(list(df_vintage2.columns)):
            if i==j:
                col_list.append(v)
                col_list.append(k)
    df=df[col_list]
    df.drop(["count_missing_dev","count_missing_val","min_dev","min_val","max_dev","max_val"],axis=1,inplace=True)

    combined_df = df # combined data of both vintages
    df.to_csv(os.path.join(os.getcwd(), 'output', "Stats Report.csv"))
    return df
#=======================================#

def get_KS(y_true, y_pred, vintage=None, save=False,flag=True):
    """
    Calculates KS values, measuring the consistency between two samples
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: KS Table for the given data
    """
    data = pd.DataFrame({'target': y_true, 'prob': y_pred})
    data['target0'] = 1 - data['target']
    data['bucket'] = pd.qcut(data['prob'], 10, duplicates='drop')
    grouped = data.groupby('bucket', as_index=False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()['prob']
    kstable['max_prob'] = grouped.max()['prob']
    kstable['events'] = grouped.sum()['target']
    kstable['nonevents'] = grouped.sum()['target0']
   

    kstable = kstable.sort_values(by="min_prob", ascending=flag).reset_index(drop=True)

    kstable['event_rate'] = kstable['events'].div(kstable[['events','nonevents']].sum(axis=1), axis=0)#.apply('{0:.2%}'.format)
    kstable['event_capture_rate'] = (kstable.events / data['target'].sum())#.apply('{0:.2%}'.format)
    kstable['nonevent_capture_rate'] = (kstable.nonevents / data['target0'].sum())#.apply('{0:.2%}'.format)
    kstable['cum_event_capture_rate'] = (kstable.events / data['target'].sum()).cumsum()
    kstable['cum_nonevent_capture_rate'] = (kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['KS'] = abs(np.round(kstable['cum_event_capture_rate'] - kstable['cum_nonevent_capture_rate'], 3) * 100)
    kstable['cum_event_capture_rate'] = kstable['cum_event_capture_rate'].apply('{0:.2%}'.format)
    kstable['cum_nonevent_capture_rate'] = kstable['cum_nonevent_capture_rate'].apply('{0:.2%}'.format)
    kstable.index.rename('Decile', inplace=True)
    kstable.index = kstable.index + 1

    return kstable
def get_RO(y_true, y_pred, vintage=None, save=False,flag=True):
    """
    Calculates KS values, measuring the consistency between two samples
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: KS Table for the given data
    """
    data = pd.DataFrame({'target': y_true, 'prob': y_pred})
    data['target0'] = 1 - data['target']
    data['bucket'] = pd.qcut(data['prob'], 10, duplicates='drop')
    grouped = data.groupby('bucket', as_index=False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()['prob']
    kstable['max_prob'] = grouped.max()['prob']
    kstable['events'] = grouped.sum()['target']
    kstable['nonevents'] = grouped.sum()['target0']
   

    kstable = kstable.sort_values(by="min_prob", ascending=flag).reset_index(drop=True)

    kstable['event_rate'] = kstable['events'].div(kstable[['events','nonevents']].sum(axis=1), axis=0).apply('{0:.2%}'.format)
    kstable['event_capture_rate'] = (kstable.events / data['target'].sum())#.apply('{0:.2%}'.format)
    kstable['nonevent_capture_rate'] = (kstable.nonevents / data['target0'].sum())#.apply('{0:.2%}'.format)
    kstable['cum_event_capture_rate'] = (kstable.events / data['target'].sum()).cumsum()
    kstable['cum_nonevent_capture_rate'] = (kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['KS'] = abs(np.round(kstable['cum_event_capture_rate'] - kstable['cum_nonevent_capture_rate'], 3) * 100)
    kstable['cum_event_capture_rate'] = kstable['cum_event_capture_rate'].apply('{0:.2%}'.format)
    kstable['cum_nonevent_capture_rate'] = kstable['cum_nonevent_capture_rate'].apply('{0:.2%}'.format)
    # kstable.index.rename('Decile', inplace=True)
    kstable.index = kstable.index + 1
    kstable=kstable[['min_prob','max_prob','events','nonevents','event_rate']]
    kstable.rename(columns={'min_prob':"Minimum Probability",'max_prob':"Maximum Probability"},inplace=True)
    kstable.index.rename('Decile', inplace=True)
    return kstable
#====================================================================================================#

def get_rmse(df, actual_vars, pred_vars):
    # Calculate AUC Score For given model on data
    rmse_value = np.sqrt(np.square((1-(df[pred_vars]/1000))-df[actual_vars]).sum()/df.shape[0])

    return pd.DataFrame({"RMSE VALUE": [rmse_value]})
#====================================================================================================#

def get_auc(df, actual_vars, pred_vars):
    # Calculate AUC Score For given model on data
    auc_score = round(roc_auc_score(df[actual_vars], (1-(df[pred_vars]/1000))),2)
    
    return pd.DataFrame({"AUC Score": [auc_score]})

def get_gini(df, actual_vars, pred_vars):
    # Calculate AUC Score For given model on data
    auc_score = round(roc_auc_score(df[actual_vars], (1-(df[pred_vars]/1000))),2)
    gini_index=round(((2*auc_score)-1),2)

    return pd.DataFrame({"GINI Index": [gini_index]})
#========================================================================================#

# function to round values
def round_df(df, num_places):
    for c in df.columns:
        try:
            na_mask = df[c].notnull()
            df.loc[na_mask, c] = df.loc[na_mask, c].astype(float).round(num_places)
        except Exception as e:
            print(e)
            pass
    return df

# Function to read the file
# Function to read all Excel files in the folder and return a dictionary of DataFrames
def read_all_excel_files(folder_location):
    excel_files = [f for f in os.listdir(folder_location) if f.endswith('.xlsx')]

    if not excel_files:
        st.warning("No Excel files (.xlsx) found in the folder.")
        return {}

    dataframes = {}
    
    for filename in excel_files:
        file_path = os.path.join(folder_location, filename)
        try:
            df = pd.read_excel(file_path)
            dataframes[filename] = df
        except pd.errors.EmptyDataError:
            st.warning(f"{filename} is empty.")
        except Exception as e:
            st.error(f"Error reading {filename}: {str(e)}")
    return dataframes

def write_dataframes_to_excel_merged(df1, df2, excel_filename):
    # Concatenate the DataFrames vertically
    merged_df = pd.concat([df1, df2], axis=0)
    
    # Create an Excel writer
    output_path = './output/'
    excel_writer = pd.ExcelWriter(output_path + excel_filename, engine='xlsxwriter')
    
    # Write the merged DataFrame to the Excel sheet
    merged_df.to_excel(excel_writer, index=False)
    
    # Save the Excel file
    excel_writer.close()

# # write one dataframe to excel
def write_dataframe_to_excel(df, excel_filename):
    # Create an Excel writer
    excel_writer = pd.ExcelWriter(write_output_path + excel_filename, engine='xlsxwriter')
    # Write the merged DataFrame to the Excel sheet
    df.to_excel(excel_writer, index=False)    
    # Save the Excel file
    excel_writer.close()
    return write_output_path + excel_filename


# function to read from sparse excel sheet based on template
def extract_tables(sheet_dfs,template_test_dict):
    extracted_tables = {}
    try:
        for file_name, test_sheet in sheet_dfs.items():

            for section_name in template_test_dict.keys():
            
                # Get limits of columns and rows
                lim_x = template_test_dict[section_name]['lim_x']
                lim_y = template_test_dict[section_name]['lim_y']

                # Get the DataFrame for the corresponding sheet name          
                
                # if test_sheet is None:
                #     raise ValueError(f"No sheet named '{test_name}' found.")

                # Get the header row and assign it as the column names
                extracted_tables[section_name] = test_sheet.iloc[lim_x[0]:lim_x[1], lim_y[0]:lim_y[1]]
                extracted_tables[section_name] = extracted_tables[section_name].reset_index()
                extracted_tables[section_name].pop('index')
                extracted_tables[section_name].columns = extracted_tables[section_name].iloc[0]
                extracted_tables[section_name] = extracted_tables[section_name].iloc[1:, :]

            # print("*"*20, extracted_tables)
            
    except Exception as e:
        print("Error occurred while extracting tables:")
        raise e

    return extracted_tables

#===============================================================================#

def testing_results_full_text(mt_model_name, mt_methods, data_vin_1, data_vin_2, target_var, predicted_var, model_test_funk):

    # Write the opening line of testing results
    testing_results_desc = "The results of various tests are shown in this section."

    # if no tests were selected then return
    if mt_methods == []:
        return testing_results_desc, [], []

    # append the name of first test
    # testing_results_desc += "\n\nResults and Conclusion of " + mt_methods[0] + " test\n"
    vin_1_results ={}
    vin_2_result = {}

    # calculate output of each test
    test_results_list = []
    test_obs_list = []
    for tst in mt_methods:
        if tst == "KS Statistic":
            results_data_vin1 = get_KS(data_vin_1[target_var], data_vin_1[predicted_var], 1).round(2)
            results_data_vin2 = get_KS(data_vin_2[target_var], data_vin_2[predicted_var], 2).round(2)
            vin_1_results[tst] = results_data_vin1["KS"]
            vin_2_result[tst] = results_data_vin2["KS"]

            # append results to list of list of tables
            test_results_list.append(results_data_vin1)
            test_results_list.append(results_data_vin2)
            write_dataframes_to_excel_merged(results_data_vin1, results_data_vin2, str(tst) +'.xlsx')
            # get observations
            observation = model_test_funk(results_data_vin1, results_data_vin2, mt_model_name, tst)
            test_obs_list.append(observation)
            
        elif tst == "AUC":
            results_data_vin1 = get_auc(data_vin_1, target_var,predicted_var).round(2)
            results_data_vin2 = get_auc(data_vin_2, target_var, predicted_var).round(2)
            vintage_type_1 = ["Development"]
            vintage_type_2 = ["Validation"]
            results_data_vin1.insert(0, 'Sample', vintage_type_1)
            results_data_vin2.insert(0, 'Sample', vintage_type_2)
            vin_1_results[tst] = results_data_vin1
            vin_2_result[tst] = results_data_vin2
            # append results to list of list of tables
            test_results_list.append(results_data_vin1)
            test_results_list.append(results_data_vin2)
            write_dataframes_to_excel_merged(results_data_vin1, results_data_vin2, str(tst) +'.xlsx')
            # get observations
            observation = model_test_funk(results_data_vin1, results_data_vin2, mt_model_name, tst)
            test_obs_list.append(observation)
            
        elif tst == "GINI":
            results_data_vin1 = get_gini(data_vin_1, target_var,predicted_var).round(2)
            results_data_vin2 = get_gini(data_vin_2, target_var, predicted_var).round(2)
            vintage_type_1 = ["Development"]
            vintage_type_2 = ["Validation"]
            results_data_vin1.insert(0, 'Sample', vintage_type_1)
            results_data_vin2.insert(0, 'Sample', vintage_type_2)
            vin_1_results[tst] = results_data_vin1
            vin_2_result[tst] = results_data_vin2
            # append results to list of list of tables
            test_results_list.append(results_data_vin1)
            test_results_list.append(results_data_vin2)
            write_dataframes_to_excel_merged(results_data_vin1, results_data_vin2, str(tst) +'.xlsx')
            # get observations
            observation = model_test_funk(results_data_vin1, results_data_vin2, mt_model_name, tst)
            test_obs_list.append(observation)
        
        elif tst == "Rank Ordering":
            results_data_vin1 = get_RO(data_vin_1[target_var], data_vin_1[predicted_var], 1).round(2)
            results_data_vin2 = get_RO(data_vin_2[target_var], data_vin_2[predicted_var], 1).round(2)
            results_data_vin1.rename(columns={'event_rate':"Development Event Rate"},inplace=True)
            results_data_vin2.rename(columns={'event_rate':"Validation Event Rate"},inplace=True)
            test_results_list.append(results_data_vin1)
            test_results_list.append(results_data_vin2)
            vin_1_results[tst] = results_data_vin1
            vin_2_result[tst] = results_data_vin2
            write_dataframes_to_excel_merged(results_data_vin1, results_data_vin2, str(tst) +'.xlsx')
            # get observations
            observation = model_test_funk(results_data_vin1, results_data_vin2, mt_model_name, tst)
            test_obs_list.append(observation)

        elif tst == "RMSE":
            results_data_vin1 = get_rmse(data_vin_1, target_var,predicted_var).round(2)
            results_data_vin2 = get_rmse(data_vin_2, target_var, predicted_var).round(2)
            vintage_type_1 = ["Development"]
            vintage_type_2 = ["Validation"]
            results_data_vin1.insert(0, 'Sample', vintage_type_1)
            results_data_vin2.insert(0, 'Sample', vintage_type_2)
            vin_1_results[tst] = results_data_vin1
            vin_2_result[tst] = results_data_vin2
            # append results to list of list of tables
            test_results_list.append(results_data_vin1)
            test_results_list.append(results_data_vin2)
            write_dataframes_to_excel_merged(results_data_vin1, results_data_vin2, str(tst) +'.xlsx')
            # get observations
            observation = model_test_funk(results_data_vin1, results_data_vin2, mt_model_name, tst)
            test_obs_list.append(observation)

    return testing_results_desc, test_results_list, test_obs_list, vin_1_results, vin_2_result

def summarize_result(testing_result):
    #calling test result function
    testing_result_list=testing_result[1]
    summarize_test_table=pd.DataFrame(columns=["Test","Development","Validation"])
    summarize_test_table["Test"]=["KS","AUC","GINI","Rank Ordering Break","RMSE"]
    summarize_test_table.iloc[0,1]=testing_result_list[0]["KS"].max()
    summarize_test_table.iloc[0,2]=testing_result_list[1]["KS"].max()
    summarize_test_table.iloc[1,1]=testing_result_list[2]["AUC Score"].sum()
    summarize_test_table.iloc[1,2]=testing_result_list[3]["AUC Score"].sum()
    summarize_test_table.iloc[2,1]=testing_result_list[4]["GINI Index"].sum()
    summarize_test_table.iloc[2,2]=testing_result_list[5]["GINI Index"].sum()
    summarize_test_table.iloc[3,1]="NO"
    summarize_test_table.iloc[3,2]="NO"
    table1_list=[]
    table2_list=[]
    for i in list(testing_result_list[6].iloc[:,-1]):
        i=i.replace("%","")
        table1_list.append(float(i))
    for i in list(testing_result_list[7].iloc[:,-1]):
        i=i.replace("%","")
        table2_list.append(float(i))

    for i,v in enumerate(table1_list):
        if i<(len(table1_list)-1):
            if table1_list[i]<table1_list[i+1]:
                summarize_test_table.iloc[3,1]="Yes"
                break
    for i,v in enumerate(table2_list):
        if i<(len(table2_list)-1):
            if table2_list[i]<table2_list[i+1]:
                summarize_test_table.iloc[3,2]="Yes"
                break
    summarize_test_table.iloc[4,1]=testing_result_list[8]["RMSE VALUE"].sum()
    summarize_test_table.iloc[4,2]=testing_result_list[9]["RMSE VALUE"].sum()

    return summarize_test_table

# function to create bench mark analysis
def benchmark_analysis(model_testing_result,vantage_testing_result):
    #calling test result function
    model_result_list=model_testing_result[1]
    vantage_result_list=vantage_testing_result[1]

    test="KS"
    benchmark_table1=pd.DataFrame()
    for i,v in enumerate(["Development","Validation"]):
        benchmark_table=pd.DataFrame()
        benchmark_table["Sample"]=[v]
        benchmark_table["Benchmark model KS"]=[vantage_result_list[i][test].max()]
        benchmark_table["Model KS"]=[model_result_list[i][test].max()]
        benchmark_table["% Change in KS"]=((benchmark_table["Model KS"]-benchmark_table["Benchmark model KS"])/benchmark_table["Benchmark model KS"]).apply('{0:.2%}'.format)
        benchmark_table1=pd.concat([benchmark_table1,benchmark_table],axis=0)

    return benchmark_table1

#Function to convert dd/mm/yyy to Month-YY
def convert_to_desired_format(input_date):
    parsed_date = datetime.strptime(input_date, "%m/%d/%Y")
    formatted_date = parsed_date.strftime("%b-%Y")
    return formatted_date

def vintage_level_bad_rate(development, validation,unique_id_variable, approval_variable, vintage_variable, target_variable):
    
# Aggregate applications for development data calculate %goods, %bads and total applications
    approval_counts_dev = development.loc[development[approval_variable] == 'YES'].groupby(vintage_variable)[approval_variable].count().reset_index()
    bad_counts_dev = development.loc[development[target_variable] == 1].groupby(vintage_variable)[target_variable].count().reset_index()
    approval_counts_dev['Sample'] = 'Development'
    
    approval_counts_dev = approval_counts_dev.rename(columns={approval_variable: 'total_approved'})
    bad_counts_dev =bad_counts_dev.rename(columns={target_variable:'total_bads'})
    total_counts_dev = development.groupby(vintage_variable)[approval_variable].count().reset_index()
    
    total_counts_dev = total_counts_dev.rename(columns={approval_variable: '#Applications'})
    approval_counts_dev = pd.merge(approval_counts_dev, total_counts_dev, on=[vintage_variable], how='left')
    bad_counts_dev = pd.merge(bad_counts_dev, total_counts_dev, on=[vintage_variable], how='left')
    approval_counts_dev['Approval Rate'] = approval_counts_dev['total_approved']/approval_counts_dev['#Applications'] 
    approval_counts_dev['Bad Rate'] = bad_counts_dev['total_bads']/approval_counts_dev['#Applications']

    #st.write(approval_counts_dev)

# Aggregate applications for development data calculate %goods, %bads and total applications
    approval_counts_val = validation.loc[validation[approval_variable] == 'YES'].groupby(vintage_variable)[approval_variable].count().reset_index()
    bad_counts_val = validation.loc[validation[target_variable] == 1].groupby(vintage_variable)[target_variable].count().reset_index()
    approval_counts_val['Sample'] = 'Validation'
    approval_counts_val = approval_counts_val.rename(columns={approval_variable: 'total_approved'})
    bad_counts_val =bad_counts_val.rename(columns={target_variable:'total_bads'})
    total_counts_dev = validation.groupby(vintage_variable)[approval_variable].count().reset_index()
    total_counts_dev = total_counts_dev.rename(columns={approval_variable: '#Applications'})
    approval_counts_val = pd.merge(approval_counts_val, total_counts_dev, on=[vintage_variable], how='left')
    approval_counts_val['Approval Rate'] = approval_counts_val['total_approved']/approval_counts_val['#Applications']
    approval_counts_val['Bad Rate'] = bad_counts_val['total_bads']/approval_counts_val['#Applications']
    #Concat Final dfs containing both validation and development data
    vintage_rates = pd.concat([approval_counts_dev, approval_counts_val], ignore_index=True)
    vintage_rates = vintage_rates.rename(columns ={'application_month': 'Vintage'})
    vintage_rates = vintage_rates[['Sample', 'Vintage', '#Applications', 'Approval Rate', 'Bad Rate']]
    vintage_rates['Vintage'] = vintage_rates['Vintage'].dt.strftime('%b-%Y')
    vintage_rates['Approval Rate'] = vintage_rates['Approval Rate'].apply(lambda x: '{:.1%}'.format(x))
    vintage_rates['Bad Rate'] = vintage_rates['Bad Rate'].apply(lambda x: '{:.1%}'.format(x))
#     st.write(vintage_rates)
    return vintage_rates


def generate_dependent_variable_text(dependant_variable_definition):
    final_text = " The dependent variable definition for the model is " + dependant_variable_definition +"\n\n"
                 
    return final_text

def plot_percentage_Bad_Rate_Approval_Rate(data, output_file):
    """
    Plot %Bad Rate and %Approval Rate for Development and Validation over vintage months.


    Parameters:
    data (dict): Dictionary containing data.
    output_file (str): Output file path to save the plot as a PNG.

    Returns:
    None
    """
     # Create a DataFrame
    df = pd.DataFrame(data)


    # Convert 'Approval Rate' and 'Bad Rate' to float
    df['Approval Rate'] = df['Approval Rate'].str.replace('%', '').astype(float)
    df['Bad Rate'] = df['Bad Rate'].str.replace('%', '').astype(float)

    # Calculate the maximum values for setting y-axis limits
    max_approval_rate = max(df['Approval Rate'])
    max_bad_rate = max(df['Bad Rate'])

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Group the data by "Sample" (Development and Validation)
    grouped = df.groupby("Sample")

    # Plot the Approval Rate for Development and Validation
    axs[0].bar(grouped.get_group("Development")["Vintage"], grouped.get_group("Development")["Approval Rate"], label="Development", alpha=0.7)
    axs[0].bar(grouped.get_group("Validation")["Vintage"], grouped.get_group("Validation")["Approval Rate"], label="Validation", alpha=0.7)

    # Plot the Bad Rate for Development and Validation
    axs[1].bar(grouped.get_group("Development")["Vintage"], grouped.get_group("Development")["Bad Rate"], label="Development", alpha=0.7)
    axs[1].bar(grouped.get_group("Validation")["Vintage"], grouped.get_group("Validation")["Bad Rate"], label="Validation", alpha=0.7)

    # Set y-axis limits based on your data
    max_approval_rate = max(df["Approval Rate"])
    max_bad_rate = max(df["Bad Rate"])
    axs[0].set_ylim(0, max_approval_rate + 10)
    axs[1].set_ylim(0, max_bad_rate + 10)

    # Add x-axis labels
    axs[0].set_xticklabels(df["Vintage"])
    axs[1].set_xticklabels(df["Vintage"])

    # Add legends
    axs[0].legend()
    axs[1].legend()

    # Set labels and title
    axs[0].set_ylabel("Approval Rate (%)")
    axs[1].set_ylabel("Bad Rate (%)")
    plt.xlabel("Vintage")
    fig.suptitle("Approval Rate and Bad Rate for Development and Validation Samples")
    #axs[1].title("Bad Rate for Development and Validation Samples")

    plt.tight_layout()    # Save the plot as a PNG file
    plt.savefig(output_file)

def generate_exclusion_table(dev,val,Exclusion_column=None):
    Exclusion_df=pd.DataFrame(columns=['Exclusion Details','Development Acct','% of Development Acct','Validation Acct','% of Validation Acct'],index=['TOTAL ACCT']+list(dev[Exclusion_column].unique()))
    Exclusion_df['Exclusion Details']=['TOTAL ACCT']+list(dev[Exclusion_column].unique())
    
    #Creating observation exclusion df
#     observation_df=Exclusion_df
    Exclusion_df.loc['TOTAL ACCT','Development Acct']=dev.shape[0]
    Exclusion_df.loc['TOTAL ACCT','Validation Acct']=val.shape[0]
    for i in list(dev[Exclusion_column].unique()):
        Exclusion_df.loc[i,'Development Acct']=dev.loc[dev[Exclusion_column]==i].shape[0]
        Exclusion_df.loc[i,'Validation Acct']=val.loc[val[Exclusion_column]==i].shape[0]
    Exclusion_df['% of Development Acct']=Exclusion_df['Development Acct']/dev.shape[0]
    Exclusion_df['% of Development Acct']=Exclusion_df['% of Development Acct'].apply(lambda x: '{:.1%}'.format(x))
    Exclusion_df['% of Validation Acct']=Exclusion_df['Validation Acct']/val.shape[0]
    Exclusion_df['% of Validation Acct']=Exclusion_df['% of Validation Acct'].apply(lambda x: '{:.1%}'.format(x))
    Exclusion_df=pd.concat([pd.DataFrame(Exclusion_df.iloc[0]).T,Exclusion_df.iloc[2:].sort_values(by=['Development Acct'],ascending=False),pd.DataFrame(Exclusion_df.iloc[1]).T],axis=0)

    return Exclusion_df

def plot_rank_ordering_chart(test_result,output_file):
    #calling test result function
    test="Rank Ordering"
    vin1_results=test_result[3][test]
    vin2_results=test_result[4][test]
    vin1_results=vin1_results['Development Event Rate'].str.replace('%', '').astype(float)
    vin2_results=vin2_results['Validation Event Rate'].str.replace('%', '').astype(float)
    df=pd.DataFrame({"Decile":[1,2,3,4,5,6,7,8,9,10],
                     'Development':list(vin1_results),
                     'Validation':list(vin2_results)})
    
    df.plot(x="Decile", y=["Development", "Validation"])
    # plt.show()
    plt.title('Rank Ordering Chart')
    plt.ylabel("Bad rate in %")
    plt.tight_layout()    # Save the plot as a PNG file
    plt.savefig(output_file)


  





