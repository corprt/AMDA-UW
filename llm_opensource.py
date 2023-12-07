import os
import langchain
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import textwrap
import pandas as pd
#import textract
from transformers import pipeline
import streamlit as st
import pandas as pd
import os
from openpyxl import load_workbook

#Open source LLM block
def open_source_api(template, value):
    huggingfacehub_api_token="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"
    repo_id = "HuggingFaceH4/zephyr-7b-beta"
    llama_llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                         repo_id=repo_id,
                         model_kwargs={"temperature":0.1, "max_new_tokens":1000})
    prompt = PromptTemplate(template=template, input_variables=['input_file_text'])
    llm_chain = LLMChain(prompt=prompt, llm=llama_llm)
    response = llm_chain.run({'input_file_text':value})
    return response

def open_source_params(prompt):
    huggingfacehub_api_token="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"

    repo_id = "HuggingFaceH4/zephyr-7b-beta"
    zephyr_llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                        repo_id=repo_id,
                        model_kwargs={"temperature":0.1, "max_new_tokens":1000})
    
    llm_chain = LLMChain(prompt=prompt, llm= zephyr_llm)

    return llm_chain

# def data_overview_vintage_description_opensource(vintage_level_bad_df, mt_model_name):
    
#     df_dict = vintage_level_bad_df.to_dict(orient='records')
#     template = """
#     You are a machine learning modeler creating a summary for data overview selected
#     This is a dataframe converted to a list, with each element as one row comtaining summary of different samples used in the model: \n {df_dict} \n   
#     Create a summary in 100 words drawing insights from the table, comment on the values across vintages.
#     Offer insights and compare  the approval rate and bad rates across vintages of each sample and why it is important to consider different samples when building a model. Use conscise and statistical language in present tense.    
#  """


#     prompt = PromptTemplate(template=template, input_variables=["df_dict"])
    
#     return prompt

#====================================================================================================================================================================================================#

df=pd.DataFrame({"A":[1,2,3],"B":[4,5,6],"C":["1%","2%","3%"]})
model_name="UW"
def data_overview_vintage_description_opensource(df, model_name ):
    # Convert dataframe rows to a list for better results through Open AI 
    data_list = df.to_dict(orient='records')
    data_str = '\n'.join([', '.join(f"{k}: {v}" for k, v in item.items()) for item in data_list])
    #prompt
    prompt_desc = "You are a machine learning engineer writing Model Development Document for "+str(model_name)+". Write a section about data overview. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about data. " + \
    "This table contain data overview result: \n {input_file_text} \n" + \
    "Explain importance of this data overview for modeling and Explain briefly with numbers." + \
    "Explain in briefly. " + \
    "Do not give any suggetion in the output." +\
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "
        

    # prompt_desc = "You are a machine learning engineer writing Model Development Document for "+str(model_name)+". Write a section about Data Overview." +\
    #     "The table which contain summary of the data : \n {input_file_text} \n" + \
    #     "This is what you know about data. Do not repeat the information as it is and start with good sentence. Learn from the information and paraphrase in technical and succinct way." + \
    # "Keep the language in present tense"+\
    # "Start paragraph with proper sentence"+\
    #     "The table lists the summaries of vintages leveraged for the new score model development." +\
    #     "Start with explanation of the timeframe for development and validation vintages. In the overview mention why it is necessary to consider different population for model development"
    # prompt_desc = "You are a machine learning modeler writing Model Development Document for " +str(model_name)+ ". write a section for data overview selected." + \
    #     "This is a dataframe converted to a list, with each element as one row comtaining summary of different samples used in the model: \n {input_file_text} \n" + \
    # "Create a summary in 100 words drawing insights from the table, comment on the values across vintages."+\
    #     "Offer insights and compare  the approval rate and bad rates across vintages of each sample and why it is important to consider different samples when building a model. Use conscise and statistical language in present tense."
    
    # prompt_desc="You are a machine learning modeler creating a summary for data overview." +\
    #     "This is a dataframe converted to a list, with each element as one row comtaining summary of different samples used in the model: \n {input_file_text} \n" +\
    #     "Create a summary in 100 words drawing insights from the table, comment on the values across vintages." +\
    #     "Offer insights and compare  the approval rate and bad rates across vintages of each sample and why it is important to consider different samples when building a model. Use conscise and statistical language in present tense."
    
    data_overview_desc = open_source_api(prompt_desc, data_str)
    print('*'*20)
    print('Response received:\n', data_overview_desc)
    # add a line to introduce df.info() that will be printed
    return data_overview_desc

def data_overview_vintage_description_2_opensource(df, model_name, data_list):
    # data_list = df.to_dict(orient='records')
    # data_str = '\n'.join([', '.join(f"{k}: {v}" for k, v in item.items()) for item in data_list])

    
    prompt_desc =  "You are a machine learning engineer writing Model Development Document for "+str(model_name)+". Write a section about Data Overview in 4/5 sentences." + \
    "Keep the language in present tense"+\
        "Please mention about count of variables and rows present in data." +\
    "This is what you know about data. Do not repeat the information as it is. Learn from the information and paraphrase in technical and succinct way." + \
        "The number of variables are" + str(df.shape[1]) +"and the number of observations are" +str(df.shape[0]) +\
        "Pick the important variables from"+ str(data_list) +"which are relevant to th report and mention them "
    data_overview_desc_2 = open_source_api(prompt_desc,df)
    return data_overview_desc_2

#====================================================================================================================================================================================================#
def data_quality_check_description_opensource(stats_df, data_dict):

    data_list = stats_df.to_dict(orient='records')
    data_str = '\n'.join([', '.join(f"{k}: {v}" for k, v in item.items()) for item in data_list])

    sliced_dict = {key: value for key, value in list(data_dict.items())[:10]}
    prompt_desc = ""
    
    for column, description in sliced_dict.items():
        columns = f"{description}\n"

    prompt_desc = "You are a machine learning engineer Write a section about data quality check. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about data. " + \
    "This table contain data quality result: \n {input_file_text} \n" + \
    "Eaplain overall result and The variables present in the data are " +str(columns) +\
    "Explain importance of this data quality check for modeling and Explain briefly with numbers." + \
    "Explain in briefly. " + \
    "Do not give any suggetion in the output." +\
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "
        
    data_quality_check_desc = open_source_api(prompt_desc,data_str)

    return data_quality_check_desc

#====================================================================================================================================================================================================#

def data_exclusion_check_description_opensource(excl_df):
    data_list = excl_df.to_dict(orient='records')
    data_str = '\n'.join([', '.join(f"{k}: {v}" for k, v in item.items()) for item in data_list])
    
    prompt_desc = "You are a machine learning engineer writing Model Development Document. Write a section about Data Exclusion. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about data. There are two vintages of data, development and validation. " + \
    "The data regarding exclusions is in one table. The table is for Exclusion. Please Expalin this tables in detail as possible. Start with explaining these exclusions very briefly" + \
    "This is table which contain stats of the data : \n {input_file_text} \n" + \
    "Summarize the findings in tables. Highlight the kind of exclusions performed and why they were excluded use the percentages in development and validation account columns for data, do not use" + \
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "

    table_value = str(excl_df.to_string(index = False))

    # data_exclusion_check_desc = open_ai_API(prompt_desc)
    data_exclusion_check_desc = open_source_api(prompt_desc, data_str)
    return data_exclusion_check_desc

def exclusion_types_description_opensource(Exclusion_df):
    prompt_desc = "You are a machine learning engineer writing Model Development Document. Write a section about exclusion types. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about exclusion types. " + \
    "This table contains exlusion types to exclude from the data please explain them in details: \n {input_file_text} \n" + \
    "Explain in briefly. " + \
    "Do not give any suggetion in the output." +\
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "

    table_value = str(Exclusion_df.iloc[1:-1,0].to_string(index = False)) #+ "\n" + str(gini_result_df.to_string(index = False))
    # table_value = str(Exclusion_df.to_string(index = False))
    exclusion_type_call = open_source_api(prompt_desc, table_value)
    return exclusion_type_call

#====================================================================================================================================================================================================#

def feature_importance_description_opensource(feature_importance_table):

    prompt_desc = "You are a machine learning engineer writing Model Development Document. Write a section about feature importance. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about feature importance table. " + \
    "This table contains informantion about the important features used for the modeling please explain them in details: \n {input_file_text} \n" + \
    "Explain in briefly. " + \
    "Do not give any suggetion in the output." +\
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "

    table_value = str(feature_importance_table.to_string(index = False)) #+ "\n" + str(gini_result_df.to_string(index = False))

    feature_importance_call = open_source_api(prompt_desc, table_value)
    return feature_importance_call

#====================================================================================================================================================================================================#

def create_description_opensource(value, model):
    prompt_desc= "You are a machine learning modeller, describe : \n {input_file_text} \n" +\
    "Explain in 100 words do not give any sort of formulas in output. " + \
    "test in 8 lines used to calculate statistical performance of a " + str(model) + ". Be descriptive and include definition in a scientific manner.keep the summary statistically relevant and limited to 10 sentences. Do not explain the columns in much detail, explain the relevance of the scores themselves.Use present tense "
    # table_value = str(value.to_string(index = False))
    test_description = open_source_api(prompt_desc, value)
    return test_description

def testing_plan_description_opensource(model_name, model_testing_methods):

    # description of all tests
    test_desription = []
    for tst in model_testing_methods:
        test_desc = create_description_opensource(tst, model_name)
        test_desription.append(test_desc)    
    return test_desription 

#====================================================================================================================================================================================================#
def model_test_observation_opensource(data1, data2, model, stat_test):

    if stat_test == "KS Statistic":
        prompt_test_obs = "You are a machine learning engineer writing Model Development Document. Write a section about Performance test. " + \
    "Test performed on a " +str(model)+" predictor model." +\
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about testing result. " + \
    "This table contain KS Score for development and validation data: \n {input_file_text} \n" + \
    "Maximum KS for the devlopement and validation data are" +str(data1['KS'].max())+ " and " +str(data2['KS'].max())+ " respectively" + \
    "Explain the defination of KS Statistics as well " + \
    "Explain in briefly. " + \
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "
        
    elif stat_test == "AUC":
        prompt_test_obs = "You are a machine learning engineer writing Model Development Document. Write a section about Performance test. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about testing result. " + \
    "This table contain performance or result of model on development and validation data: \n {input_file_text} \n" + \
    "Please consider validation drop up to 10% " + \
    "Explain the defination of AUC Score as well. " + \
    "Explain in briefly. " + \
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "
        
    elif stat_test == "GINI":
        prompt_test_obs = "You are a machine learning engineer writing Model Development Document. Write a section about Performance test. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about testing result. " + \
    "This table contain performance or result of model on development and validation data: \n {input_file_text} \n" + \
    "Please consider validation drop up to 10% and do not add this validation drop in output" + \
    "Explain the defination of GINI Score as well. " + \
    "Explain in briefly. " + \
    "Do not give any suggetion in the output." +\
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "

    elif stat_test == "Rank Ordering":
        prompt_test_obs = "You are a machine learning engineer writing Model Development Document. Write a section about Performance test. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about testing result. " + \
    "This table contain rank ordering or result of model on development and validation data: \n {input_file_text} \n" + \
    "Explain Rank Ordering based on Bad rate column that is there break in bad rate decile wise. " + \
    "Explain Rank Ordering break based on Bad rate column " + \
    "Explain the defination of Rank Ordering as well. " + \
    "Explain in briefly. " + \
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "
        
    elif stat_test == "RMSE":
        prompt_test_obs = "You are a machine learning engineer writing Model Development Document. Write a section about Performance test. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about testing result. " + \
    "This table contains RMSE value of model for developmnet and validation data: \n {input_file_text} \n" + \
    "explain the RMSE result in detail." + \
    "Explain the defination of RMSE Score as well." + \
    "Explain in briefly." + \
    "Do not give any suggetion in the output." +\
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "


    else:
        prompt_test_obs = "You are a machine learning engineer writing Model Development Document. Write a section about Performance test. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about testing result. " + \
    "This table contain performance or result of model on development and validation data: \n {input_file_text} \n" + \
    "Drop in validation performance is within thresholds. " + \
    "Explain in briefly. " + \
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "
                      
    data_list1 = data1.to_dict(orient='records')
    data_str1 = '\n'.join([', '.join(f"{k}: {v}" for k, v in item.items()) for item in data_list1])
    data_list2 = data2.to_dict(orient='records')
    data_str2 = '\n'.join([', '.join(f"{k}: {v}" for k, v in item.items()) for item in data_list2])
    # table_to_summarize_1 = data1.to_string()
    # table_to_summarize_2 =  data2.to_string()
    table_to_summarize = data_str1 + data_str2
    observation_summary = open_source_api(prompt_test_obs, table_to_summarize)
    return observation_summary

def summarize_table_result_description_opensource(result_df):
    prompt_desc = "You are a machine learning engineer writing Model Development Document. Write a section about Performance test. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about testing result. " + \
    "This is the table which contain performance of all test : \n {input_file_text} \n" + \
    "This table contain sammarize result of all performance test or result of model on development and validation sample. " + \
    "Explain in briefly. " + \
    "Do not give any suggetion in the output." +\
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "

    table_value = str(result_df.to_string(index = False)) #+ "\n" + str(gini_result_df.to_string(index = False))

    auc_gini_result_desc = open_source_api(prompt_desc, table_value)
    return auc_gini_result_desc

def benchmark_result_description_opensource(result_df):
 
    data_list = result_df.to_dict(orient='records')
    data_str = '\n'.join([', '.join(f"{k}: {v}" for k, v in item.items()) for item in data_list])
    #prompt
    prompt_desc = "You are a machine learning engineer Write a section about Benchmark analysis. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about data. " + \
    "This table contain benchmark analysis KS Statistics result: \n {input_file_text} \n" + \
    "Explain in briefly. " + \
    "Do not give any suggetion in the output." +\
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "
        
    benchmark_analysis_call = open_source_api(prompt_desc, data_str)
    return benchmark_analysis_call