import openai
import io
import pandas as pd


openai.api_type = "azure"
openai.api_base = "https://exl-isc-minerva-openai-svcs.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "28c49a2bd351436d8b85ac8693e25bd4"

# import functions from other modules
from utils import *


import os
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import textwrap
from langchain import HuggingFaceHub

# Setting Env
if st.secrets["OPENAI_API_KEY"] is not None:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

#huggingfacehub_api_token=OPENAI_API_KEY

#Open source LLM block
def open_source_api(template, value):
    
    repo_id = "HuggingFaceH4/zephyr-7b-beta"
    llama_llm = HuggingFaceHub(
                         repo_id=repo_id,
                         model_kwargs={"temperature":0.1, "max_new_tokens":1000})
    prompt = PromptTemplate(template=template, input_variables=['input_file_text'])
    llm_chain = LLMChain(prompt=prompt, llm=llama_llm)
    response = llm_chain.run({'input_file_text':value})
    return response

#Open AI LLM block
def open_ai_API(prompt):

    # return "Temporary Response from OpenAI"

    response = openai.ChatCompletion.create(
    engine="exl-isc-minerva-openai-svc-gpt01",    
    messages = [{"role":"user","content":prompt}],
    temperature=0.0,
    max_tokens=4000,
    top_p=1,
    frequency_penalty=1,
    presence_penalty=0,
    stop=None)
    return response.choices[0].message.content.strip()

#Open AI LLM block => designed by Shambhavi
def open_ai_API_2(prompt, value):

    # return "Temporary Response from OpenAI"

    response = openai.ChatCompletion.create(
    engine="exl-isc-minerva-openai-svc-gpt01",    
    messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": value}
],
    temperature=0.0,
    max_tokens=4000,
    top_p=1,
    frequency_penalty=1,
    presence_penalty=0,
    stop=None)
    return response.choices[0].message.content.strip()


#Description fetch and poulate for each tets
#report description
def report_description(value, model,sample_names):
    prompt_desc= "Give brief introduction of model evaluation test report for a" + str(model) +"model for following" +str(sample_names)+ "for the following tests performed" + str(value) +".The report contains review and recommendations."
    report_dsc = open_ai_API(prompt_desc)
    return report_dsc    

# Table description for each test
def table_description(model, table, test):
    table_string = table.to_string(index=False)
    prompt_desc = "You are a machine learning modeller, summarize the results in the tables which have been convertes " +str(table_string) + "for" +str(test) +" test for a " +str(model)    
    table_desc = open_ai_API(prompt_desc)
    return table_desc 




#==========================================================Section 3.1 Data Overview===========================================================================================#

# function to get commentary for data overview subsection
def data_overview_vintage_description(df, model_name ):
    # Convert dataframe rows to a list for better results through Open AI 
    data_list = df.to_dict(orient='records')
    data_str = '\n'.join([', '.join(f"{k}: {v}" for k, v in item.items()) for item in data_list])
    prompt_desc = "You are a machine learning engineer writing Model Development Document for "+str(model_name)+". Write a section about Data Overview in 4/5 sentences." + \
    "Keep the language in present tense"+\
        "This is what you know about data. Do not repeat the information as it is. Learn from the information and paraphrase in technical and succinct way." + \
        "The table lists the summaries of vintages leveraged for the new score model development." +\
        "Please explain with the numbers as well" +\
        "Start with explanation of the timeframe for development and validation vintages. In the overview mention why it is necessary to consider different population for model development. " +\
        "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents."
    print("Sending prompt : ", prompt_desc)
    data_overview_desc = open_ai_API_2(prompt_desc, data_str)
    print('*'*20)
    print('Response received:\n', data_overview_desc)
    # add a line to introduce df.info() that will be printed
    return data_overview_desc 

def data_overview_vintage_description_2(df, model_name, data_list):

    
    prompt_desc =  "You are a machine learning engineer writing Model Development Document for "+str(model_name)+". Write a section about Data Overview in 4/5 sentences." + \
    "Keep the language in present tense"+\
        "This is what you know about data. Do not repeat the information as it is. Learn from the information and paraphrase in technical and succinct way." + \
        "The number of variables are" + str(df.shape[1]) +"and the number of observations are" +str(df.shape[0]) +\
        "Pick the important variables from"+ str(data_list) +"which are relevant to th report and mention them "
    data_overview_desc_2 = open_ai_API(prompt_desc)
    return data_overview_desc_2

#==========================================================Section 3.2 Data Quality Check==================================================================================#
def data_quality_check_description(stats_df, data_dict):

    data_list = stats_df.to_dict(orient='records')
    data_str = '\n'.join([', '.join(f"{k}: {v}" for k, v in item.items()) for item in data_list])

    sliced_dict = {key: value for key, value in list(data_dict.items())[:10]}
    prompt_desc = ""
    
    for column, description in sliced_dict.items():
        columns = f"{description}\n"
    st.write(prompt_desc)
    prompt_desc = "You are a machine learning engineer writing the data quality check section in a modelling document. The content should be insightful, in present tense and should be worded formally succint, concise, not more that 200 words. No assumptions or unnecessary reccomendations are required"+\
    data_str + "The columns of the data are" +str(columns)+\
    "This is the table which contains comparison of stats like count, mean, standard dev, missing for two vintages development and validation"+\
    "Do not use column names in the content anywhere, when referencing use development and validation for comparison between the different statistics across columns for the two samples"
    "In your summary highlight what these differences mean statistically. Comment on differences exceeding 10%." +\
    " Do not provide column names where the insights are not valuable or th evaluables are categorical and unique identifiers etc. Describe inconsistencies if any."+\
    "Compare and contrast for the development and validation samples across each statistic. Comment statistically and draw insights on distributions in the columns prefixed with std which are for standard deviation for the two samples "
    
    #print("Sending prompt : ", prompt_desc)
    data_quality_check_desc = open_ai_API(prompt_desc)
    print('*'*20)
    print('Response received:\n', data_quality_check_desc)
    
    return data_quality_check_desc 


#===============================================================Section 3.3 Exclusions================================================================================================#


def data_exclusion_check_description(excl_df):
    prompt_desc = "You are a machine learning engineer writing Model Development Document. Write a section about Data Exclusion. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about data. There are two vintages of data, development and validation. " + \
    "The data regarding exclusions is in one table. The table is for Exclusion. Please Expalin this tables in detail as possible. Start with explaining these exclusions very briefly" + \
    "Summarize the findings in tables. Highlight the kind of exclusions performed and why they were excluded use the percentages in development and validation account columns for data, do not use" + \
    "This table contains exlusion types to exclude from the data please explain them in details" + \
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "

    table_value = str(excl_df.to_string(index = False))

    # data_exclusion_check_desc = open_ai_API(prompt_desc)
    data_exclusion_check_desc = open_ai_API_2(prompt_desc, table_value)
    return data_exclusion_check_desc

def exclusion_types_description(Exclusion_df):
    prompt_desc = "You are a machine learning engineer writing Model Development Document. Write a section about exclusion types. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about exclusion types. " + \
    "This table contains exlusion types to exclude from the data please explain them in details" + \
    "Explain in briefly. " + \
    "Do not give any suggetion in the output." +\
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "

    table_value = str(Exclusion_df.iloc[1:-1,0].to_string(index = False)) #+ "\n" + str(gini_result_df.to_string(index = False))

    exclusion_type_call = open_ai_API_2(prompt_desc, table_value)
    return exclusion_type_call

#====================================================================================================================================================================================================#

def feature_importance_description(feature_importance_table):

    prompt_desc = "You are a machine learning engineer writing Model Development Document. Write a section about feature importance. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about feature importance table. " + \
    "This table contains informantion about the important features used for the modeling please explain them in details" + \
    "Explain in briefly. " + \
    "This all features are important for the modeling. " +\
    "Do not give any suggetion in the output." +\
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "

    table_value = str(feature_importance_table.to_string(index = False)) #+ "\n" + str(gini_result_df.to_string(index = False))

    feature_importance_call = open_ai_API_2(prompt_desc, table_value)
    return feature_importance_call

#====================================================================================================================================================================================================#

def data_Sampling_check_description(data_sampling_df):
    prompt_desc = "You are a machine learning engineer writing Model Development Document. Write a section about Data Sampling. " + \
    "Do not add any heading of section. Do not use the word project use model instead. Use samples to refer to the different vintages. Do not repeat the above information, use this knowledge and your experience to express and paraphrase." + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents." + \
    "This is what you know about data. There are two vintages of data. The data used for sampling check is " + str(data_sampling_df.to_string(index = False)) + ". "+\
    "Explain the difference in means for the two vintages and why the sampling was carried out" +\
    "Use development and validation to refer to vintages"
    data_Sampling_check_desc = open_ai_API(prompt_desc)
    return data_Sampling_check_desc
#========================================================================================================================================================#

#Description fetch and poulate for each tets
def create_description(value, model):
    prompt_desc= "You are a machine learning modeller, describe " + str(value) + " test in 8 lines used to calculate statistical performance of a " + str(model) + ". Be descriptive and include definition in a scientific manner.keep the summary statistically relevant and limited to 10 sentences. Do not explain the columns in much detail, explain the relevance of the scores themselves.Use present tense "
    test_description = open_ai_API_2(prompt_desc, value)
    return test_description
#=============================================================Section 5 Testing ===============================================================#

def model_test_observation(data1, data2, model, stat_test):

    if stat_test == "KS Statistic":
        prompt_test_obs = "You are a machine learning modeller, summarize the tables which have been extracted as text to be more readable, the table contains the details of "+str(stat_test)+\
        " test performed for a" +str(model)+" predictor model. Keep the summary statistically relevant and limited to 10 sentences." +\
        "The tables are comparison across two vintages, table 1 is for Development and tables 2 for Validation"+\
             " Do not explain the columns in much detail, explain the relevance of the scores themselves." +\
             "If it is KS test then the values of KS are" + str(data1['KS'].max()) + "and" + str(data2['KS'].max())+ "respectively for Development and Validation" +\
             "Keep decile level observation limited for the deciles with maximum score of each test as above. Make sure the final values captured are accurate for each test"+\
             "Comment on these scores and their statistical relevance for KS test and what it signifies"
        
    elif stat_test == "AUC":
        prompt_test_obs = "You are a machine learning engineer writing Model Development Document. Write a section about Performance test. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about testing result. " + \
    "This table contain performance or result of model on development and validation data. " + \
    "Drop in validation performance is within thresholds. " + \
    "Explain in briefly. " + \
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "
    
    elif stat_test == "Rank Ordering":
        prompt_test_obs = "You are a machine learning engineer writing Model Development Document. Write a section about Performance test. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about testing result. " + \
    "This tables contain bad rate of development and validation data. do the rank ordering based on the bad rate of development and validation data result." + \
    "Explain the rank ordering based on the event rate columns present in two tables and explain about any discrepancies found in event rate columns" + \
    "Explain in briefly. " + \
    "Do not give any suggetion in the output." +\
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "

    elif stat_test == "RMSE":
        prompt_test_obs = "You are a machine learning engineer writing Model Development Document. Write a section about Performance test. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about testing result. " + \
    "This table contains RMSE value of model for developmnet and validation data. explain the RMSE result in detail." + \
    "Explain what should be rmse value and its importance" + \
    "Explain in briefly." + \
    "Do not give any suggetion in the output." +\
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "


    else:
        prompt_test_obs = "You are a machine learning engineer writing Model Development Document. Write a section about Performance test. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about testing result. " + \
    "This table contain performance or result of model on development and validation data. " + \
    "Drop in validation performance is within thresholds. " + \
    "Explain in briefly. " + \
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "
        
#     else:        
#         prompt_test_obs = "You are a machine learning modeller, summarize the tables which have been extracted as text to be more readable, the table contains the details of "+str(stat_test)+\
#         " test performed for a" +str(model)+" predictor model. Keep the summary statistically relevant and limited to 10 sentences." +\
#         "The tables are comparison across two vintages, table 1 is for development and tables 2 for Validation"+\
#              " Do not explain the columns in much detail, explain the relevance of the scores themselves." +\
#                  " Summary should clearly highlight the difference of scores for Table 1 and Table 2, use Table 1 and Table 2 to reference the respective tables. Comment statistically on these differences and what they mean, Use present tense."+\
#                     "Keep decile level observation limited for the deciles with highest score of each test. Make sure the final values captured are accurate for each test"+\
#                  "Mention that the overall performance of the model has increased between vintages and is good "
                           
    
    table_to_summarize_1 = data1.to_string()
    table_to_summarize_2 =  data2.to_string()
    table_to_summarize = table_to_summarize_1 + table_to_summarize_2
    observation_summary = open_ai_API_2(prompt_test_obs, table_to_summarize)
    return (observation_summary)
#===============================================================================#

def summarize_table_result_description(result_df):
    prompt_desc = "You are a machine learning engineer writing Model Development Document. Write a section about Performance test. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about testing result. " + \
    "This table contain sammarize result of all performance test or result of model on development and validation sample. " + \
    "Explain in briefly. " + \
    "Do not give any suggetion in the output." +\
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "

    table_value = str(result_df.to_string(index = False)) #+ "\n" + str(gini_result_df.to_string(index = False))

    auc_gini_result_desc = open_ai_API_2(prompt_desc, table_value)
    return auc_gini_result_desc

def benchmark_result_description(result_df):
    prompt_desc = "You are a machine learning engineer writing Model Development Document. Write a section about Benchmark analysis. " + \
    "Do not add any heading of section. Do not use the word project use model instead. This is what you know about Benchmark analysis. " + \
    "This table contain Benchmark analysis of KS Statistics Explain in details percentage change in model KS over vantage KS" + \
    "Please Give Appropriate answer only" + \
    "This analyis results are within threshold" + \
    "Explain in briefly. " + \
    "Please Do not give any suggetion in the output." +\
    "Do not repeat the above information, use this knowledge and your experience to express and paraphrase. " + \
    "Do not use words that express emotions like unfortunately or fortunately, or luckily while writing the contents. "

    table_value = str(result_df.to_string(index = False)) #+ "\n" + str(gini_result_df.to_string(index = False))

    benchmark_analysis_call = open_ai_API_2(prompt_desc, table_value)
    return benchmark_analysis_call

def testing_plan_description(model_name, model_testing_methods):

    # description of all tests
    test_desription = []
    for tst in model_testing_methods:
        test_desc = create_description(tst, model_name)
        test_desription.append(test_desc)    
    return test_desription 
#==========================================================Section 5 Charts===========================================================================#

def create_charts_openai(df1, df2, test):
    current_directory = os.getcwd()
    # chart_path = os.path.join(current_directory, 'output', 'chartimages')
    # not using cwd, otherwise it doesn't work if there is a space in the folder name
    chart_path = os.path.join('output', 'chartimages')
    prompt_chart = "These instructions are to create a code for charts which will be used to embed into a risk model documentation report" +\
    "You have the following columns" +str(df1) + "and" + str(df2) +"which are the results of" + str(test) +"for two vintages of the same model" +\
    ".Import all the necessary modules and generate a code to plot the results contained in the columns above. Make sure they there are not syntax errors"+\
    "Close all brackets, make sure all variables which are used are defined, avoid whitespaces, avoid adding any comments, it should be a clean well written code which can be easily compiled"+\
    "Adjust the label size and axis-units to ensure clear visibility along the x-axis. There should be clear disticntion between two vintages in the graph" +\
    "Include an appropriate title for the figure corresponding to the test, and column name for axis labels. Use a font size of 18 for the title and 15 for the axis labels. Utilize a white background for the graphs" +\
    "When plotting, use colors to highlight different metrics within the chart. Do not show the graph instead save it as a png file as "+str(test)+".png in the following location" +str(chart_path) # + \
    # " in the current working directory that can be fetched by using os.getcwd(). " + \
    # " While writing code pay special attention to variable names and suffixes to make sure no runtime error is thrown."

    chart_code = open_ai_API(prompt_chart)
    # Get the current directory

# Create the file path
    file_path = os.path.join(current_directory, 'output', 'chartscripts', "chart_for_" +str(test) +".py")
    f= open(file_path, 'w')
    f.write(chart_code)
    f.close
    
#===============================================================================#
