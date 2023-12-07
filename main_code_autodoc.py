import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from docx import Document
import traceback
import os
from pathlib import Path
import yaml
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

# import functions from other modules
from utils import *
from document_functions import *
from llm_openai import *
import definitions 
from streamlit_space import space
import datetime
from llm_opensource import *
path = "" 

# init docum
# temp_doc= Document()
document = Document()
obj_styles = document.styles
# obj_charstyle = obj_styles.add_style(definitions.css_inputs['style'], WD_STYLE_TYPE.PARAGRAPH)
obj_charstyle = obj_styles.add_style('CommentsStyle', WD_STYLE_TYPE.PARAGRAPH)
obj_font = obj_charstyle.font
#obj_font.size = Pt(12)
obj_font.name = 'Calibri'

# Document Data
document_title = "Model Development Document"


#================================================================================#
#=====================================SC==========================================#
# Start of the main code
st.set_page_config(layout="wide", page_title= "Auto Document Creator")
#Logo and heading
st.image("logo.png", use_column_width=False)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# st.title("Auto Document Creator")
st.title("Automated Model Documentation Assistant ")
#st.write("(Automated Model Documentation Assistant)")

# Create an empty dictionary to store the files
# check if variables are present in memory so that they are not reinitialized
if 'file_dictionary' not in st.session_state:
    st.session_state.file_dictionary = {}
if 'vintages_dataframes_dict' not in st.session_state:
    st.session_state.vintages_dataframes_dict = {}
if 'exclusion_dataframes_dict' not in st.session_state:
    st.session_state.exclusion_dataframes_dict = {}
if 'section_subsec_dict' not in st.session_state:
    st.session_state.section_subsec_dict = {}
if 'optional_input_data' not in st.session_state:
    st.session_state.optional_input_data={}
            
st.markdown(definitions.markdown_inputs['Spacing_input'], unsafe_allow_html=True)

# Template dictionary for x and y limit of each table

template_test_dict_exclusion = {'Observation_Exclusion': {'lim_y': (0,6), 'lim_x': (2,8)},
                       'Performance_Exclusion': {'lim_y': (0,6), 'lim_x': (11,18)}}

tab1, tab2, tab3= st.tabs(['Model Taxonomy','Model Inputs','Document Template'])

# Inputs to increase tab font size
css = definitions.markdown_inputs['Tab size']
st.markdown(css, unsafe_allow_html=True)

#=====================================Input Model==========================================#
with tab1:
    space(container=None, lines=1)
    col1, col2 = st.columns([3, 3])
    with col1:
#         space(container=None, lines=1)
        #Choose the model
        st.markdown(definitions.markdown_inputs['Model Name'], unsafe_allow_html=True)
        mt_model_name = st.text_input(':black', "Enter name of the model",label_visibility='collapsed')

        #Model Type 
        st.markdown(definitions.markdown_inputs['Model Type'], unsafe_allow_html=True)
        model_specs = st.selectbox('Select type of model', definitions.tab_1_inputs['Model Type'], label_visibility='collapsed')
        #Model Algoritham
        st.markdown(definitions.markdown_inputs['Model Algorithm'], unsafe_allow_html=True)
        algo_options = st.selectbox('Select type of algorithm',definitions.tab_1_inputs['Algorithm'] ,label_visibility='collapsed')
        #=====SC======#
        
    with col2:
        #Document Type
        st.markdown(definitions.markdown_inputs['Document Type'], unsafe_allow_html=True)
        report_options = st.selectbox('Select type of report', definitions.tab_1_inputs['Document Type'], label_visibility='collapsed')
        #functiona area
        st.markdown(definitions.markdown_inputs['Functional area'],unsafe_allow_html=True)
        functional_options = st.selectbox('Functional Area',definitions.tab_1_inputs['Functional Area'],label_visibility='collapsed' )
        if algo_options=="XGBoost":
            st.markdown(definitions.markdown_inputs['Optimization Technique'],unsafe_allow_html=True)
            sub_functional_options = st.selectbox('Select Optimization Technique', definitions.tab_1_inputs['Optimization Technique 1'],label_visibility='collapsed')
        else:
            st.markdown(definitions.markdown_inputs['Optimization Technique'],unsafe_allow_html=True)
            sub_functional_options = st.selectbox('Select Optimization Technique', definitions.tab_1_inputs['Optimization Technique 2'],label_visibility='collapsed')

                
                           

#=====================================Input Data==========================================================#
with tab2:
    col1, col2, col3 = st.columns([3, 3, 3])
    columns_for_selectbox = ['Placeholder']
    with col1:
        # Creating Header for Model Development Datasets
        st.markdown(definitions.markdown_inputs['Model Development Datasets'], unsafe_allow_html=True)
        # Creating Header and user input for Data Location
        st.markdown(definitions.markdown_inputs['Data Location'], unsafe_allow_html=True)
        input_data_location=st.text_input(':black', "Enter data location",label_visibility='collapsed')
        
        # Creating Header and user input for Development Dataset
        st.markdown(definitions.markdown_inputs['Development Dataset'], unsafe_allow_html=True)
        folder_input1 = st.text_input(':black', 'Development Data',label_visibility='collapsed')
        folder_input1=os.path.join(path, input_data_location,folder_input1)
        
        # Creating Header and user input for Testing Dataset
        st.markdown(definitions.markdown_inputs['Validation Dataset'], unsafe_allow_html=True)
        folder_input2 = st.text_input(':black','Validation Data',label_visibility='collapsed')
        folder_input2=os.path.join(path, input_data_location,folder_input2)
        
        # Creating Header for Additional Validation Data
        st.markdown(definitions.markdown_inputs['Additional Validation Data'], unsafe_allow_html=True)
        selected1 = st.checkbox('Additional Validation Data')
        if selected1:
            additional_validation_data = st.text_input(':black','Additional Validation Data',label_visibility='collapsed')
            additional_validation_data=os.path.join(path, input_data_location,additional_validation_data)
        
        # Creating Header for Data Dictionary
        st.markdown(definitions.markdown_inputs['Data Dictionary'], unsafe_allow_html=True)
        data_dictinary = st.text_input(':black','Data Dictionary',label_visibility='collapsed')
        data_dictinary=os.path.join(path, input_data_location,data_dictinary)
        
        # Option for Model Variables
        # Creating Header and user input for Out Of Time Data and data dictionary
        st.markdown(definitions.markdown_inputs['Model Variables'], unsafe_allow_html=True)
        optional_input=['Model Variables']
        optional_input_dict = {}

        #Display checkbox for oot data.
        for option in optional_input:
            selected = st.checkbox(option)
            if selected:
                optional_input_dict[option] = st.text_input(':black',option,label_visibility='collapsed')
                optional_input_dict[option]= os.path.join(path, input_data_location, optional_input_dict[option])

#========================================== READ FILES BUTTON PROCESSING START =====================================#              
        read_files = st.button("Read Files")
        preview = st.button("Preview files")
        # File flag to detect if files have been read 
        if 'button_state' not in st.session_state:
            st.session_state['button_state'] = False
        if read_files:
            # init state variables
            st.session_state.file_dictionary = {}
            st.session_state.vintages_dataframes_dict = {}
            st.session_state.optional_input_dict = {}
            st.session_state.addtional_validation_data_dict = {}
            st.session_state.data_dict = {}
            st.session_state['button_state'] = not st.session_state['button_state']

            # Read and store input vintages based on folder location
            input_vintage_files = {}
            for i,v in enumerate([folder_input1,folder_input2]):
                df = read_excel_sheet(v)
                st.session_state.vintages_dataframes_dict['Vintage_Data_' + str(i+1)] = df
                input_vintage_files[v.split('\\')[-1]] = df
            st.session_state.file_dictionary["Input Vintages"] = input_vintage_files
            
            # Read and store input validation(oot) data based on folder location
            additional_validation_file = {}
            # selected1 Variable taken from additional validation data user input options
            if selected1:
                df = read_excel_sheet(additional_validation_data)
                st.session_state.addtional_validation_data_dict[additional_validation_data.split('\\')[-1].split('.')[0]] = df
                additional_validation_file[additional_validation_data.split('\\')[-1]] = df
                st.session_state.file_dictionary["Additional validation data"] = additional_validation_file
            
            # Read and store input data dictionary based on folder location
            data_dictinary_file = {}
            df = read_excel_sheet(data_dictinary)
            st.session_state.data_dict[data_dictinary.split('\\')[-1].split('.')[0]] = df
            data_dictinary_file[data_dictinary.split('\\')[-1]] = df
            st.session_state.file_dictionary["Data Dictinary"] = data_dictinary_file
        

             # Read and store input oot and data dictionary based on file name without extension in optional_input_dict
            optional_input_files = {}
            for option in optional_input:
                if option in optional_input_dict.keys():
                    file_path = optional_input_dict[option]
                    df = read_excel_sheet(file_path)
                    st.session_state.optional_input_dict[file_path.split('\\')[-1].split('.')[0]] = df
                    optional_input_files[file_path.split('\\')[-1]] = df
            st.session_state.file_dictionary["Optional Input"] = optional_input_files

                        
            # Optional Display the stored files
            if st.session_state.file_dictionary:
                st.success("Files read and stored successfully. \nClick on Preview to see the files."+ \
                            "\nSelect next tab from the top of the page.")
    
    #========================================== READ FILES BUTTON PROCESSING END =====================================#

        # show preview if button clicked and data is present
        if st.session_state.file_dictionary and preview:
            for vintage, files in st.session_state.file_dictionary.items():
                st.write(vintage)
                for filename, df in files.items():
                    st.write(f"- {filename}")
                    st.write(df)
#=====================================Target Definition Inputs==========================================================#               
        with col2:
            # Creating Header for Target Definition
            st.markdown(definitions.markdown_inputs['Key Variables'], unsafe_allow_html=True)
            
            # Creating Header and selectbox for Dependant variable definition
            st.markdown(definitions.markdown_inputs['Dependant variable definition'],unsafe_allow_html=True)
            dependant_variable_definition=st.text_input(':black','enter dependent variable definition',label_visibility='collapsed')
                   
            if st.session_state['button_state']:
                columns_for_selectbox =list(st.session_state.vintages_dataframes_dict['Vintage_Data_1'].columns)
                # Creating Header and selectbox for Target Variable
                st.markdown(definitions.markdown_inputs['Dependant Variable Name'], unsafe_allow_html=True)
                target_col = st.selectbox("Select target variable column", ["select target variable column"]+columns_for_selectbox,label_visibility='collapsed')
        
                # Creating Header and selectbox for Model Score
                st.markdown(definitions.markdown_inputs['Final model output'], unsafe_allow_html=True)
                predicted_col = st.selectbox("Select model score column", ["select model score column"]+columns_for_selectbox,label_visibility='collapsed')
                
                # Creating Header and selectbox for Vintage variable
                st.markdown(definitions.markdown_inputs['Vintage'], unsafe_allow_html=True)
                vintage_col = st.selectbox("Select vintage column", ["select vintage column"]+columns_for_selectbox,label_visibility='collapsed') 
                
                # Creating Header and selectbox for Unique Identifier
                st.markdown(definitions.markdown_inputs['Unique Identifier'], unsafe_allow_html=True)
                unique_id_col = st.selectbox("Select unique ID column", ["select unique ID column"]+columns_for_selectbox,label_visibility='collapsed')


#=====================================Variable definition inputs==========================================================#   
        with col3:
            #If block controls if the files have been read once and the columns can be accessed
            if st.session_state['button_state']:
                # Creating Header for Additional Variables for Analysis
                st.markdown(definitions.markdown_inputs['Additional Variables for Analysis'], unsafe_allow_html=True)
                
                # Creating Header and selectbox for Exclusion Variable
                st.markdown(definitions.markdown_inputs['Exclusion Variable'], unsafe_allow_html=True)
                exclusion_col = st.selectbox("Select exclusion column", ["Select exclusion column"]+columns_for_selectbox,label_visibility='collapsed')
                
                # Creating Header and selectbox for Benchmark Score
                st.markdown(definitions.markdown_inputs['Benchmark Model'], unsafe_allow_html=True)
                benchmark_col = st.selectbox("Select benchmark score column", ["Select benchmark score column"]+columns_for_selectbox,label_visibility='collapsed')
                
                # Creating Header and selectbox for approval tag        
                st.markdown(definitions.markdown_inputs['Approval Tag'], unsafe_allow_html=True)
                approval_col = st.selectbox("Select approval tag column", ["Select approval tag column"]+columns_for_selectbox,label_visibility='collapsed')
                
                # Creating Header and selectbox for Segmentation Variable
                st.markdown(definitions.markdown_inputs['Segmentation'], unsafe_allow_html=True)
                segmentation_col = st.selectbox("Select segmentation column", ["Select segmentation column"]+columns_for_selectbox,label_visibility='collapsed')
                

   
#========================================================Documentation Section ==================================================================#
with tab3:
    col1, col2 = st.columns([2, 4]) # create two columns
    with col1:
        # Create a multiselect option button
        st.subheader("Select sections")
        #===== HB ====#
        section_options_orig = ['1. Model Scope, Purpose and Use','2. Limitations and Compensating Controls', '3. Model Data','4. Model Specification','5. Model Testing', '6. Model Implementation', '7. Operating and Control Environment', '8. Ongoing Monitoring and Governance Plan', '9. Reference', '10. Appendix']
        section_options = []
        for section in section_options_orig:
            st.session_state.section_subsec_dict[section] = []

        # Display checkboxes and collect selected options
        section_options_raw = [] # name of sections with number
# Display checkboxes with numbers and collect selected options
        for i, option in enumerate(section_options_orig, start=1):            
            checkbox_label = f"{i}.{option.split('. ', 1)[1]}"
            is_selected = st.checkbox(checkbox_label,value=True)
            if is_selected:
                section_options_raw.append(option)

        # remove the numerical index from section name
        # keep a dictionary to fetch section name with number when needed
        section_dict = {}
        section_options = []
        if section_options_raw:
            section_options = [s.split('.')[1].strip() for s in section_options_raw]
            for s,sec_raw in enumerate(section_options_raw):
                section_dict[section_options[s]] = sec_raw
                
#         if section_options_template: # at least one section selected
            #Submit button to create document
                
                
#========================================== DQ Inputs =====================================#
    with col2:
        dict_subsections = {} # dictionary to store subsections corresponding to sections
        subsections_dict = {} # keep a dictionary to fetch subsection name with number when needed      
        
        # show header for subsection selection
        list_implemented_sections = ['Model Data', 'Model Testing','Model Specification', 'Model Implementation', 'Ongoing Monitoring and Governance Plan'] # sections whose subsections are to be displayed
        st.subheader("Select sub-sections")
        # if any such sections are selected that have subsections applicable, then do not show below tip    
        if not any([s in list_implemented_sections for s in section_options]):
            st.write("(Select sections on the left to display applicable subsections)")

        if 'Model Data' in section_options:
            # pick subsections
            st.write("3.Model Data")
            subsections_options_raw = st.multiselect('Select subsections of Model Data section', definitions.sections_subsections['Model Data'],
                                                default=definitions.sections_subsections['Model Data'],label_visibility='collapsed')
            st.session_state.section_subsec_dict['3. Model Data'] = subsections_options_raw
            # remove the numerical index from subsection name
            subsections_options = []
            if subsections_options_raw:
                subsections_options = [s.split('.')[2].strip() for s in subsections_options_raw]
                for s,subsec_raw in enumerate(subsections_options_raw):
                    subsections_dict[subsections_options[s]] = subsec_raw
            dict_subsections['Model Data'] = subsections_options

        if 'Model Specification' in section_options:
            st.write('4.Model Specification')
            subsections_options_raw = st.multiselect('Select subsections of Model Specification section', definitions.sections_subsections['Model Specification'], 
                                                     default=definitions.sections_subsections['Model Specification'] ,label_visibility='collapsed')
            st.session_state.section_subsec_dict['4. Model Specification'] = subsections_options_raw
            #remove the numerical index from subsection name
            subsections_options = []
            if subsections_options_raw:
                subsections_options = [s.split('.')[2].strip() for s in subsections_options_raw]
                for s,subsec_raw in enumerate(subsections_options_raw):
                    subsections_dict[subsections_options[s]] = subsec_raw
            dict_subsections['Model Specification'] = subsections_options

        
#========================================== Testing Inputs =====================================#
        if 'Model Testing' in section_options:

            st.write("5.Model Testing")
            subsections_options_raw = st.multiselect('Select subsections of Model Testing section', definitions.sections_subsections['Model Testing'],
                                                label_visibility='collapsed', default= definitions.sections_subsections['Model Testing'])
            st.session_state.section_subsec_dict['5. Model Testing'] = subsections_options_raw
            #remove the numerical index from subsection name
            subsections_options = []
            if subsections_options_raw:
                subsections_options = [s.split('.')[2].strip() for s in subsections_options_raw]
                for s,subsec_raw in enumerate(subsections_options_raw):
                    subsections_dict[subsections_options[s]] = subsec_raw
            dict_subsections['Model Testing'] = subsections_options

            # input for testing (for section 5.1 and (5.2 + 5.3) as per model documentation format)
            if subsections_options:
                #Pick the model tests                
                mt_methods = ["KS Statistic", "AUC", "GINI","Rank Ordering","RMSE", "PSI", "CSI"]
                # add the test names that are applicable for charts to chart options
                applicable_test_methods = []
                chart_options = set(mt_methods).intersection(set(applicable_test_methods))
                #===== SC ====#
        if 'Model Implementation' in section_options:
            st.write('6.Model Implementation')
            subsections_options_raw = st.multiselect('Select subsections of Model Implementation section', definitions.sections_subsections['Model Implementation'], 
                                                     default=definitions.sections_subsections['Model Implementation'],label_visibility='collapsed')
            st.session_state.section_subsec_dict['6. Model Implementation'] = subsections_options_raw
            #remove the numerical index from subsection name
            subsections_options = []
            if subsections_options_raw:
                subsections_options = [s.split('.')[2].strip() for s in subsections_options_raw]
                for s,subsec_raw in enumerate(subsections_options_raw):
                    subsections_dict[subsections_options[s]] = subsec_raw
            dict_subsections['Model Implementation'] = subsections_options
            
        if 'Ongoing Monitoring and Governance Plan' in section_options:
            st.write('8.Ongoing Monitoring and Governance Plan')
            subsections_options_raw = st.multiselect('Select subsections of Ongoing Monitoring and Governance Plan section', definitions.sections_subsections['Ongoing Monitoring and Governance Plan'], label_visibility ='collapsed',default = definitions.sections_subsections['Ongoing Monitoring and Governance Plan'])
            st.session_state.section_subsec_dict['8. Ongoing Monitoring and Governance Plan'] = subsections_options_raw
            #remove the numerical index from subsection name
            subsections_options = []
            if subsections_options_raw:
                subsections_options = [s.split('.')[2].strip() for s in subsections_options_raw]
                for s,subsec_raw in enumerate(subsections_options_raw):
                    subsections_dict[subsections_options[s]] = subsec_raw
            dict_subsections['Ongoing Monitoring and Governance Plan'] = subsections_options
            
#==================================================template Creation ========================================================#
#Do not Use this for subsection content addition
with tab3:
    with col1:
        if st.button("Template Document"):
            add_titlepage(document, document_title, "Model Name")
            # add table of contents
            if section_options:
                add_toc(document, st.session_state.section_subsec_dict)
                # add sections in document
                sections_printed = 0 # to keep track of how many sections printed
                subsection_tables_list = [] # list of tables
                subsection_observations_dict = {} # list of observations
                subsection_observations_list = []
                section_description = ""
                subsection_description=""
                for i,section in enumerate(section_options):
                    if section == 'Limitations and Compensating Controls':
                        # write section
                        create_doc_section(document, document_title, section_dict[section], section_description, None, None, sections_printed,definitions.bylines[section]['main'])
                        sections_printed += 1

                    elif section == 'Model Scope, Purpose and Use':
                        # write section
                        create_doc_section(document, document_title, section_dict[section], section_description, None, None, sections_printed,definitions.bylines[section]['main'])
                        sections_printed += 1

    #==================================================Model Data ========================================================#          
                    elif section == 'Model Data':
                        # write section
                        create_doc_section(document, document_title, section_dict[section], section_description, None, None, sections_printed, definitions.bylines[section]['main'])
                        sections_printed += 1

                        # subsections
                        subsections_options = dict_subsections['Model Data']
                        if subsections_options:
                            subsections_list = subsections_options
                        else:
                            subsections_list = []
                        for subsec in subsections_list:
                            # init list of descriptions, list of tables and observations
                            create_doc_subsection(document, document_title,
                                               subsections_dict[subsec], subsection_description,definitions.bylines[section][subsec], subsection_tables_list, subsection_observations_list,
                                                print_charts=False,print_tables = False, table_names = [] )


    #==================================================Model Specification ========================================================#
                    elif section == 'Model Specification':
                        # write section
                        create_doc_section(document, document_title, section_dict[section], section_description, None, None, sections_printed, definitions.bylines[section]['main'])
                        sections_printed += 1

                        subsections_options = dict_subsections['Model Specification']
                        if subsections_options:
                            subsections_list = subsections_options
                        else:
                            subsections_list = []

                        for subsec in subsections_list:
                            # write section
                            create_doc_subsection(document, document_title,
                                               subsections_dict[subsec], subsection_description,definitions.bylines[section][subsec], subsection_tables_list, subsection_observations_list,
                                                print_charts=False,print_tables = False, table_names = [] )


    #==================================================Model Testing ========================================================#           
                    elif section == 'Model Testing': #and algo_options == "Classification":
                        # write section
                        create_doc_section(document, document_title, section_dict[section], section_description, None, None, sections_printed, definitions.bylines[section]['main'])
                        sections_printed += 1

                        # subsections
                        subsections_options = dict_subsections['Model Testing']
                        if subsections_options:
                            subsections_list = subsections_options
                        else:
                            subsections_list = []
                        for subsec in subsections_list:
                            # write subsection
                            create_doc_subsection(document, document_title,
                                               subsections_dict[subsec], subsection_description,definitions.bylines[section][subsec], subsection_tables_list, subsection_observations_list,
                                                print_charts=False,print_tables = False, table_names = [] )

    #===================================================Model Implementation======================================================#

                    elif section == 'Model Implementation':
                        # write section
                        create_doc_section(document, document_title, section_dict[section], section_description, None, None, sections_printed, definitions.bylines[section]['main'])
                        sections_printed += 1

                        # subsections
                        subsections_options = dict_subsections['Model Implementation']
                        if subsections_options:
                            subsections_list = subsections_options
                        else:
                            subsections_list = []
                        for subsec in subsections_list:
                            # write subsection
                            create_doc_subsection(document, document_title,
                                               subsections_dict[subsec], subsection_description,definitions.bylines[section][subsec], subsection_tables_list, subsection_observations_list,
                                                print_charts=False,print_tables = False, table_names = [] )                         

    #=================================================Ongoing Monitoring and Governance Plan============================================#

                    elif section == 'Ongoing Monitoring and Governance Plan':
                        # write section
                        create_doc_section(document, document_title, section_dict[section], section_description, None, None, sections_printed, definitions.bylines[section]['main'])
                        sections_printed += 1

                        # subsections
                        subsections_options = dict_subsections['Ongoing Monitoring and Governance Plan']
                        if subsections_options:
                            subsections_list = subsections_options
                        else:
                            subsections_list = []
                        for subsec in subsections_list:
                            # write subsection
                            create_doc_subsection(document, document_title,
                                               subsections_dict[subsec], subsection_description,definitions.bylines[section][subsec], subsection_tables_list, subsection_observations_list,
                                                print_charts=False,print_tables = False, table_names = [] )


                    else:  # for placeholder sections
                        # write section
                        create_doc_section(document, document_title, section_dict[section], ' ', None, None, sections_printed, definitions.bylines[section]['main'])
                        sections_printed += 1

    #========================================== Template saved =====================================#
            # Save and Download Template
            document.save('template.docx') 
            template_doc_file=os.path.join(path, Path.cwd(), "template.docx")
            with open(template_doc_file, "rb") as file:
                st.download_button(
                label="Download", 
                data=file, 
                file_name="template.docx"
                )
            
#========================================== SUBMIT BUTTON PROCESSING =====================================#
with tab3:
    with col2:
        #adding timestamp in demo doc
        now = datetime.datetime.now()
        date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        # filename = "demo_" + date_time_str + ".docx"
        filename = date_time_str + ".docx"
        if section_options: # at least one section selected
            st.markdown(definitions.markdown_inputs['Ouput folder Location'], unsafe_allow_html=True)
            output_location=st.text_input(':black', "Enter Output Folder Location",label_visibility='collapsed')
            #Submit button to create document
            # Checkbox for user to choose between OpenAI and open source
            use_model = st.radio("Choice of LLM",('Closed-Source','Open-Source'))
            # use_opensource = st.checkbox("Use Open Source")
            if st.button("Create Document"):
                #add the title page
                add_titlepage(document, document_title, mt_model_name)
                # add table of contents
                if section_options:
                    add_toc(document, st.session_state.section_subsec_dict)
#=========================================================DQ Processing==============================================================================
                # add sections in document
                sections_printed = 0 # to keep track of how many sections printed
                for i,section in enumerate(section_options):
                    if section == 'Limitations and Compensating Controls':
                        section_description = ""
                        df=pd.DataFrame(definitions.df_inputs['df_dict'],index=[0])
                        
                        # write section
                        create_doc_section(document, document_title, section_dict[section], section_description, df, None, sections_printed,definitions.bylines[section]['main'])
                        sections_printed += 1

                    elif section == 'Model Scope, Purpose and Use':
                        # write section
                        #Load Yaml file for section description
                        content = open("content.yaml", "rt")
                        subsec_dict = yaml.safe_load(content)
                        content.close()
                        section_description = subsec_dict[section]["Main"]
                        
                        # write section
                        create_doc_section(document, document_title, section_dict[section], section_description, None, None, sections_printed,definitions.bylines[section]['main'])
                        sections_printed += 1
                        
                        # Creating table for this section
                        data = {'Sr No': [1,2],
                            'Product': ['<user input>', '<user input>'],
                            'Description & Model Usage': ['<user input>', '<user input>']
                        }
                        df = pd.DataFrame(data)
                        
                        # Adding table in doc
                        create_table_ofdataframe(document,df,heading="Product and its Description Table")
                        
                        
                    elif section == 'Model Data':
                        # section description
                        section_description = ""

                        # write section
                        create_doc_section(document, document_title, section_dict[section], section_description, None, None, sections_printed, definitions.bylines[section]['main'])
                        sections_printed += 1

                        # subsections
                        subsections_options = dict_subsections[section]
                        if subsections_options:
                            subsections_list = subsections_options
                        else:
                            subsections_list = []

                        for subsec in subsections_list:
                            # init list of descriptions, list of tables and observations
                            subsection_description = ""
                            subsection_tables_list = [] # list of tables
                            subsection_observations_dict = {} # list of observations
                            subsection_observations_list = []
                            if subsec == 'Data Overview':
                                # generate info of dataframe to be printed as table in document
                                
                                #Function for calculating approval and bad rates
                                vintage_level_bad_df = vintage_level_bad_rate(st.session_state.vintages_dataframes_dict['Vintage_Data_1'],
                                                                              st.session_state.vintages_dataframes_dict['Vintage_Data_2'],
                                                                               	unique_id_col,approval_col,vintage_col, target_col)
                                
                            
                                subsection_tables_list.append(vintage_level_bad_df)
                                if use_model=="Open-Source":
                                    subsection_description = data_overview_vintage_description_opensource(vintage_level_bad_df, mt_model_name)
                                else:
                                    subsection_description = data_overview_vintage_description(vintage_level_bad_df, mt_model_name)
                                #st.write(subsection_description)
                                subsection_observations_dict[subsec + "1"] = subsection_description
                                #Generate visualizations for vintage based approval and bad rate
                                plot_percentage_Bad_Rate_Approval_Rate(vintage_level_bad_df, os.path.join(os.getcwd(), 'output', 'chartimages', str(subsec)+".png"))
#                           # # flag to print table and filenames for table to excel
                                print_tables = True
                                print_charts = True

                                # Extract the second key-value pair (dataframe2)
                                second_key, second_dataframe = list(st.session_state.file_dictionary["Optional Input"].items())[0]

                                # Convert the extracted DataFrame into a new DataFrame
                                second_dataframe_df = pd.DataFrame(second_dataframe)
                                #st.write(second_dataframe_df)
                                # Sort the DataFrame by 'Score' in descending order and extract the top 3 rows
                                top_3_descriptions = second_dataframe_df.sort_values(by='Relative Influence', ascending=False).head(3)
                                # Extract the 'Description' column values and store them in a list
                                top_3_description_list = top_3_descriptions['Description'].tolist()
                            # Fetch second description
                                if use_model=="Open-Source":
                                    subsection_description_2 = data_overview_vintage_description_2_opensource(st.session_state.vintages_dataframes_dict['Vintage_Data_1'], mt_model_name, top_3_description_list)
                                else:
                                    subsection_description_2 = data_overview_vintage_description_2(st.session_state.vintages_dataframes_dict['Vintage_Data_1'], mt_model_name, top_3_description_list)

                                subsection_observations_dict[subsec + "2"] = subsection_description_2
#                                 st.write(subsection_observations_dict["Data Overview2"])
                                # write subsection
                                create_doc_subsection_data_overview(document, document_title,
                                                    subsections_dict[subsec], subsection_description, definitions.bylines[section][subsec],
                                                    subsection_tables_list, subsection_observations_dict,print_charts, print_tables, [])

                            if subsec == 'Data Quality Check':
                                # calculate stats for the data of all vintages
                                # write subsection
                                

                                subsection_description=""
                                #pre_calculated_df = st.session_state.vintages_dataframes_dict['Vintage_Data_1'][target_col, ]

                                #Calculate stats for both the vintages and return a consolidated table
                                calculated_df = calculate_stats(st.session_state.vintages_dataframes_dict['Vintage_Data_1'], st.session_state.vintages_dataframes_dict['Vintage_Data_2'])
                                calculated_df = calculated_df[:10]
                                if use_model=="Open-Source":
                                    quality_check_description = data_quality_check_description_opensource(calculated_df, definitions.variable_description)
                                else:
                                    quality_check_description = data_quality_check_description(calculated_df, definitions.variable_description)

                                #st.write(quality_check_description)
                                #st.write(desc)
                                create_doc_subsection(document, document_title,
                                                   subsections_dict[subsec], quality_check_description,definitions.bylines[section][subsec], subsection_tables_list, subsection_observations_list,
                                                    print_charts=False,print_tables = False, table_names = [] )
                                
                            if subsec == 'Data Exclusions':
                                # write subsection
                                #calculate table for data exclusions
                                Exclusion_df = generate_exclusion_table(st.session_state.vintages_dataframes_dict['Vintage_Data_1'],                                                                   st.session_state.vintages_dataframes_dict['Vintage_Data_2'],Exclusion_column=exclusion_col)
                                subsection_tables_list=[Exclusion_df]
                                # table_names="Exclusion table"
#                                 subsection_description=""
                                if use_model=="Open-Source":
                                    subsection_description = data_exclusion_check_description_opensource(Exclusion_df)
                                else:
                                    subsection_description = data_exclusion_check_description(Exclusion_df)

                                create_doc_subsection(document, document_title,
                                                   subsections_dict[subsec], subsection_description,definitions.bylines[section][subsec], subsection_tables_list, subsection_observations_list,
                                                    print_charts=False,print_tables = True, table_names = ["Exclusion table"] )
                                
                                
                                
                                # Adding table in Document
                                # create_table_ofdataframe(document,Exclusion_df,heading="Exclusion Table")

                                if use_model=="Open-Source":
                                    subsection_description = exclusion_types_description_opensource(Exclusion_df)
                                else:
                                    subsection_description = exclusion_types_description(Exclusion_df)

                                create_doc_subsection(document, document_title,
                                                   None, subsection_description,None, subsection_tables_list, subsection_observations_list,
                                                    print_charts=False,print_tables = False, table_names = [] )
                                
                            #data exclusion
                            if subsec=="Vintage Selection & Sampling":
                                create_doc_subsection(document, document_title,
                                                    subsections_dict[subsec], subsection_description, definitions.bylines[section][subsec], subsection_tables_list, subsection_observations_list,print_charts=False,print_tables = False, table_names = [])
                                
#=====================================================================Model Specification =============================================================================================#
     
                    elif section == 'Model Specification':
                        section_description=""
                        
                        # write section
                        create_doc_section(document, document_title, section_dict[section], section_description, None, None, sections_printed, definitions.bylines[section]['main'])
                        sections_printed += 1

                        #subsections_options = definitions.sections_subsections[section]
                        subsections_options = dict_subsections['Model Specification']
                        if subsections_options:
                            subsections_list = subsections_options
                        else:
                            subsections_list = []

                        for subsec in subsections_list:
                            # init list of descriptions, list of tables and observations
                            subsection_description = ""
                            subsection_tables_list = [] # list of tables
                            subsection_observations_list = [] # list of observations
                            if subsec == 'Technical Summary':
                        
                                # write subsection
                                #Load Yaml file for section description
                                content = open("content.yaml", "rt")
                                subsec_dict = yaml.safe_load(content)
                                content.close()

                                subsection_description = subsec_dict[section][subsec]
                                
                                create_doc_subsection(document, document_title,
                                                   subsections_dict[subsec], subsection_description, definitions.bylines[section][subsec], subsection_tables_list, subsection_observations_list,
                                                    print_charts=False,print_tables = True, table_names = [] )
                                

                            if subsec == 'Dependent Variable':
                                
                                # write subsection

                                content = open("content.yaml", "rt")
                                subsec_dict = yaml.safe_load(content)
                                content.close()
                                
                                subsec_dict[section][subsec]['1. Target Variable Definition']+= generate_dependent_variable_text(dependant_variable_definition)

                                subsection_description = subsec_dict[section][subsec]
                                

                                create_doc_subsection(document, document_title,
                                                   subsections_dict[subsec], subsection_description,definitions.bylines[section][subsec], subsection_tables_list, subsection_observations_list,
                                                    print_charts=False,print_tables = True, table_names = [] )

                            if subsec == 'Variable transformation and selection':
                                
                                # write subsection
                                subsection_description = subsec_dict[section][subsec]
                                create_doc_subsection(document, document_title,
                                                   subsections_dict[subsec], subsection_description,definitions.bylines[section][subsec], subsection_tables_list, subsection_observations_list,
                                                    print_charts=False,print_tables = True, table_names = [] )

                            if subsec == 'Final Model Selection':
                                
                                # write subsection
                                # subsection_description = subsec_dict[section][subsec]

                                if optional_input_dict['Model Variables']:
                                    file_path = optional_input_dict['Model Variables']
                                    df = read_excel_sheet(file_path)
                                    df['Relative Influence'] = df['Relative Influence'].apply('{:.1%}'.format)
                                    subsection_tables_list.append(df)

                                if use_model=="Open-Source":
                                    subsection_description= feature_importance_description_opensource(df)
                                else:
                                    subsection_description= feature_importance_description(df)


                                create_doc_subsection(document, document_title,
                                                   subsections_dict[subsec], subsection_description,definitions.bylines[section][subsec], subsection_tables_list, subsection_observations_list,
                                                    print_charts=False,print_tables = True, table_names = [] )
                                
  #=========================================================Testing Processing =================================================================#
                    elif section == 'Model Testing': #and algo_options == "Classification":
                        # section description
                        section_description = ""

                        # write section
                        create_doc_section(document, document_title, section_dict[section], section_description, None, None, sections_printed, definitions.bylines[section]['main'])
                        sections_printed += 1

                        # subsections
                        subsections_options = dict_subsections['Model Testing']
                        if subsections_options:
                            subsections_list = subsections_options
                        else:
                            subsections_list = []
                        for subsec in subsections_list:
                            # init list of descriptions, list of tables and observations
                            subsection_description = ""
                            subsection_tables_list = [] # list of tables
                            subsection_observations_list = [] # list of observations

                            supported_tests = ["KS Statistic", "AUC", "GINI", "Rank Ordering","RMSE"]
                            mt_methods_supported = [m for m in mt_methods if m in supported_tests]
                            # get test descriptions, list of tables with results, list of observations, vintage wise test results
                            if use_model=="Open-Source":
                                subsec_description, subsec_tables_list, subsec_observations_list, vintage_1_results, vintage_2_results = testing_results_full_text(mt_model_name, mt_methods_supported, 
                                                                                                st.session_state.vintages_dataframes_dict['Vintage_Data_1'],#['Vintage_Data_1_Model'],
                                                                                                st.session_state.vintages_dataframes_dict['Vintage_Data_2'],target_col, predicted_col,model_test_observation_opensource )#['Vintage_Data_2_Model'])
                            else:
                                subsec_description, subsec_tables_list, subsec_observations_list, vintage_1_results, vintage_2_results = testing_results_full_text(mt_model_name, mt_methods_supported, 
                                                                                                st.session_state.vintages_dataframes_dict['Vintage_Data_1'],#['Vintage_Data_1_Model'],
                                                                                                st.session_state.vintages_dataframes_dict['Vintage_Data_2'],target_col, predicted_col,model_test_observation )#['Vintage_Data_2_Model'])
                

                            if subsec == "Testing Plan":
                                # generate description using openai or opensource
                                if use_model=="Open-Source":
                                    test_description_list = testing_plan_description_opensource(mt_model_name, mt_methods_supported)                                                                
                                else:
                                    test_description_list = testing_plan_description(mt_model_name, mt_methods_supported)                                                                
                                create_doc_testing_plan(document, document_title, subsections_dict[subsec], supported_tests, test_description_list, definitions.bylines[section][subsec])

                            if subsec == "Overall Performance":
#=====================================================================Charts =============================================================================================#
                                #Generate code for charts, each chart gets stored as testname.png
                                if use_model=="Open-Source":
                                    test_result=testing_results_full_text(mt_model_name, mt_methods,st.session_state.vintages_dataframes_dict['Vintage_Data_1'],st.session_state.vintages_dataframes_dict['Vintage_Data_2'],target_var=target_col,predicted_var=predicted_col, model_test_funk=model_test_observation_opensource)
                                else:
                                    test_result=testing_results_full_text(mt_model_name, mt_methods,st.session_state.vintages_dataframes_dict['Vintage_Data_1'],st.session_state.vintages_dataframes_dict['Vintage_Data_2'],target_var=target_col,predicted_var=predicted_col, model_test_funk=model_test_observation)
                                # We need to call all charts function below and add test in chart options
                                # Store chart for rank ordering
                                plot_rank_ordering_chart(test_result,os.path.join(os.getcwd(), 'output', 'chartimages','Rank Ordering.png'))

                                chart_options=["Rank Ordering"]
                                for chart_opt in chart_options:
                                    if chart_opt in supported_tests: # if the results for tests are returned
                                        chart_code_run_successful = True
                                        excep = None
                                        num_attempts = 0
                                        # run the code until a successful run or till maximum attempts have failed
                                        while (not chart_code_run_successful) and (num_attempts < 10):
                                            try:
                                                num_attempts += 1
                                                create_charts_openai(vintage_1_results[chart_opt], vintage_2_results[chart_opt], chart_opt)
                                                # Get the current directory
                                                current_directory = os.getcwd()

                                                # Construct the file path
                                                # file_path = os.path.join(current_directory, 'output', 'chartscripts', 'chart_for_' + str(chart_opt)+'.py')
                                                file_path = os.path.join(current_directory, 'output', 'chartimages','Rank Ordering.png')

                                                # Read the Python file and store the code in a variable
                                                with open(file_path, 'r') as file:
                                                    code = file.read()
                                                exec(code)
                                                chart_code_run_successful = True
                                                # print('Chart code for ', chart_opt, ' test is successful.')
                                            except Exception as e:
                                                excep = e
                                                print('Exception ', e, ' was encountered in running chart code for ', chart_opt , ' test.')
                                                traceback.print_exc()
                                                print('Attempting to write the code again.')
                                                chart_code_run_successful = False

                                
                               #Document call for testing results
                                # chart_options = True
                                create_doc_subsection_test_results(document, subsections_dict[subsec], definitions.bylines[section][subsec], subsec_tables_list, subsec_observations_list, 
                                                                mt_methods_supported, chart_options)
                                

                            if subsec == "Summarized Result":
                                # create all test result table df
                                if use_model=="Open-Source":
                                    testing_result=testing_results_full_text(mt_model_name, mt_methods,st.session_state.vintages_dataframes_dict['Vintage_Data_1'],st.session_state.vintages_dataframes_dict['Vintage_Data_2'],target_var=target_col,predicted_var=predicted_col, model_test_funk=model_test_observation_opensource)
                                    result_df = summarize_result(testing_result)
                                else:
                                    testing_result=testing_results_full_text(mt_model_name, mt_methods,st.session_state.vintages_dataframes_dict['Vintage_Data_1'],st.session_state.vintages_dataframes_dict['Vintage_Data_2'],target_var=target_col,predicted_var=predicted_col, model_test_funk=model_test_observation)
                                    result_df = summarize_result(testing_result)
                                    
                                subsection_description = summarize_table_result_description(result_df)
                                subsection_tables_list=[result_df]
                                table_names=["Summarized Test Result Table"]

                                create_doc_subsection(document, document_title,
                                                   subsections_dict[subsec], subsection_description,definitions.bylines[section][subsec], subsection_tables_list, subsection_observations_list,
                                                    print_charts=False,print_tables = True, table_names=[])

                            if subsec == "Benchmark Analysis":
                                # crate all test result table df

                                if use_model=="Open-Source":
                                    model_testing_result=testing_results_full_text(mt_model_name, mt_methods,st.session_state.vintages_dataframes_dict['Vintage_Data_1'],st.session_state.vintages_dataframes_dict['Vintage_Data_2'],target_var=target_col,predicted_var=predicted_col,model_test_funk=model_test_observation_opensource)
                                    vantage_testing_result=testing_results_full_text(mt_model_name, mt_methods,st.session_state.vintages_dataframes_dict['Vintage_Data_1'],st.session_state.vintages_dataframes_dict['Vintage_Data_2'],target_var=target_col,predicted_var=benchmark_col,model_test_funk=model_test_observation_opensource)
                                    result_df = benchmark_analysis(model_testing_result,vantage_testing_result)
                                    subsection_description = benchmark_result_description_opensource(result_df)
                                else:
                                    model_testing_result=testing_results_full_text(mt_model_name, mt_methods,st.session_state.vintages_dataframes_dict['Vintage_Data_1'],st.session_state.vintages_dataframes_dict['Vintage_Data_2'],target_var=target_col,predicted_var=predicted_col,model_test_funk=model_test_observation)
                                    vantage_testing_result=testing_results_full_text(mt_model_name, mt_methods,st.session_state.vintages_dataframes_dict['Vintage_Data_1'],st.session_state.vintages_dataframes_dict['Vintage_Data_2'],target_var=target_col,predicted_var=benchmark_col,model_test_funk=model_test_observation)
                                    result_df = benchmark_analysis(model_testing_result,vantage_testing_result)
                                    subsection_description = benchmark_result_description(result_df)

                                subsection_tables_list=[result_df]
                                table_names=["benchmark Analysis Table"]

                                # adding subsection in the doc
                                create_doc_subsection(document, document_title,
                                                   subsections_dict[subsec], subsection_description,definitions.bylines[section][subsec], subsection_tables_list, subsection_observations_list,
                                                    print_charts=False,print_tables = True, table_names=[])
                                

                            # if subsec == "Performance across Segments":
                            #     # create all test result table df
                            #     if use_model=="Open-Source":
                            #         testing_result=testing_results_full_text(mt_model_name, mt_methods,st.session_state.vintages_dataframes_dict['Vintage_Data_1'],st.session_state.vintages_dataframes_dict['Vintage_Data_2'],target_var=target_col,predicted_var=predicted_col, model_test_funk=model_test_observation_opensource)
                            #         result_df = summarize_result(testing_result)
                            #     else:
                            #         testing_result=testing_results_full_text(mt_model_name, mt_methods,st.session_state.vintages_dataframes_dict['Vintage_Data_1'],st.session_state.vintages_dataframes_dict['Vintage_Data_2'],target_var=target_col,predicted_var=predicted_col, model_test_funk=model_test_observation)
                            #         result_df = summarize_result(testing_result)
                                    
                            #     subsection_description = summarize_table_result_description(result_df)
                            #     subsection_tables_list=[result_df]
                            #     table_names=["Summarized Test Result Table"]

                            #     create_doc_subsection(document, document_title,
                            #                        subsections_dict[subsec], subsection_description,definitions.bylines[section][subsec], subsection_tables_list, subsection_observations_list,
                            #                         print_charts=False,print_tables = True, table_names=[])

#=====================================================================Model Implementation =============================================================================================#

                    elif section == 'Model Implementation':
                        # section description
                        section_description = ""

                        # write section
                        create_doc_section(document, document_title, section_dict[section], section_description, None, None, sections_printed, definitions.bylines[section]['main'])
                        sections_printed += 1

                        # subsections
                        subsections_options = dict_subsections['Model Implementation']
                        if subsections_options:
                            subsections_list = subsections_options
                        else:
                            subsections_list = []
                        for subsec in subsections_list:
                            # init list of descriptions, list of tables and observations
                            subsection_description = ""
                            subsection_tables_list = [] # list of tables
                            subsection_observations_list = [] # list of observations
                            if subsec == 'Implementation Overview':
                                # write subsection
                                create_doc_subsection(document, document_title,
                                                    subsections_dict[subsec], subsection_description, definitions.bylines[section][subsec],
                                                     subsection_tables_list, subsection_observations_list, print_charts=False, print_tables=False)
                            
                            if subsec == 'Implementation Testing & Results':
                                # write subsection
                                create_doc_subsection(document, document_title,
                                                    subsections_dict[subsec], subsection_description, definitions.bylines[section][subsec],
                                                    subsection_tables_list, subsection_observations_list, print_charts=False, print_tables=False)


    #=====================================================================Ongoing Monitoring and Governance Plan =============================================================================================#

                    elif section == 'Ongoing Monitoring and Governance Plan':
                        # section description
                        section_description = ""

                        # write section
                        create_doc_section(document, document_title, section_dict[section], section_description, None, None, sections_printed, definitions.bylines[section]['main'])
                        sections_printed += 1

                        # subsections
                        subsections_options = dict_subsections['Ongoing Monitoring and Governance Plan']
                        if subsections_options:
                            subsections_list = subsections_options
                        else:
                            subsections_list = []
                        for subsec in subsections_list:
                            # init list of descriptions, list of tables and observations
                            subsection_description = ""
                            subsection_tables_list = [] # list of tables
                            subsection_observations_list = [] # list of observations
                            if subsec == 'Monitoring Frequency & Components':
                                # write subsection
                                create_doc_subsection(document, document_title,
                                                    subsections_dict[subsec], subsection_description, definitions.bylines[section][subsec],
                                                    subsection_tables_list, subsection_observations_list, print_charts=False, print_tables=False)
                            
                            if subsec == 'Annual Model Review plan':
                                # write subsection
                                create_doc_subsection(document, document_title,
                                                    subsections_dict[subsec], subsection_description, definitions.bylines[section][subsec],
                                                    subsection_tables_list, subsection_observations_list, print_charts=False, print_tables=False)
                                
                    else:  # for placeholder sections
                        # write section
                        create_doc_section(document, document_title, section_dict[section], ' ', None, None, sections_printed, definitions.bylines[section]['main'])
                        sections_printed += 1
                        
#==================================================Save Final doc ========================================================#
                #Save final document
                if use_model=="Open-Source":
                    filename="demo_open_source_"+filename
                    document.save(os.path.join(path, output_location,filename))
                else:
                    filename="demo_openai_"+filename
                    document.save(os.path.join(path, output_location,filename))
