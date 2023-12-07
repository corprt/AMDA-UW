bylines ={
    'Model Scope, Purpose and Use' : { 'main' : 'Provide a summary of the product or portfolio to which the model will be implemented, ' +\
                                    'encompassing essential alterations in the business strategy and noteworthy events that exerted a substantial influence on ' +\
                                    'either the portfolio or the model during the period when modeling samples were generated.' },
    'Limitations and Compensating Controls' : { 'main' : 'List all potential/known limitations identified by the model sponsor/developer. '+\
      'For each limitation identified, identify what compensating control exists to mitigate the limitation.' },
    'Model Data' : { 'main' : 'NA' ,
                     'Data Overview' : 'Provide description of data used to develop and validate the model. Explain why the model data is appropriate for model development.' ,
                     'Data Quality Check' : 'Provide evidence of consistency and integrity checks and describe how was the data tested. Data should be analyzed for missing values, outlier values, inconsistent fields.' ,
                     'Data Exclusions' : 'Document exclusions that were performed during modeling exercise, including the reasons and number of observations.',
                     'Vintage Selection & Sampling' : 'Provide Sampling table based on the vintage selction and sampling analysis'},
    'Model Specification': { 'main' : 'NA' ,
                             'Technical Summary' : 'Provide a technical summary of model development process. Describe the design, theory, and logic of the model.' ,
                             'Dependent Variable' : 'Provide the definition of dependent variable with all technical details, along with supporting analysis.' ,
                             'Variable transformation and selection' : 'Describe how the final set of variables were selected over rest of the independent variables.' ,
                             'Final Model Selection' : 'Describe the final model specification, model output, list of independent variables, and descriptions of the variables.' },
    'Model Implementation' : { 'main' : 'NA',
                               'Implementation Overview' : 'Describe the implementation system/environment where the model will be implemented for the model scoring.',
                               'Implementation Testing & Results' : 'Describe the implementation testing plan along with metrics used and the expected outcome for succesful and accurate implementation. Document the results of testing to demonstrate correct implementation.' },
    'Model Testing' : { 'main' : 'NA' ,
                        'Testing Plan' : 'Evaluate whether the selected model performs as indented by conducting and documenting a range of performance tests.' ,
                        'Overall Performance' : 'Evaluate whether the selected model performs as indented by conducting and documenting a range of performance tests.' ,
                        'Summarized Result' : 'Evaluate all test performed on respective data smaples by this model.',
                        'Benchmark Analysis' : 'Evaluate benchmark analysis on all test performed. ',
                        'Performance across Segments' : 'Evaluate model performance across various segments to demonstrate performance is sufficient with respect to intended purpose.' },
    'Operating and Control Environment': { 'main' : 'Provide evidence of show that the model resides in a secured environment where no un-authorized changes can be made to the model.' },
    'Ongoing Monitoring and Governance Plan': { 'main' : 'NA',
                                                'Monitoring Frequency & Components' : 'Describe the frequency of the model monitoring and components that will be included in the monitoring reports.',
                                                'Annual Model Review plan' : 'Provide a plan of data and performance testing results that will be provided as part of annual model review.' },
    'Reference': { 'main' : 'Provide all relevant references.' },
    'Appendix': { 'main' : 'NA'}
}

tab_1_inputs = {'Document Type': ['Select Report Type','Model Development Report', 'Model Validation Report'],
          'Model Type': ['Select Model Type','Regulatory Model', 'Risk Decision Model', 'Marketing Model'],
          'Functional Area': ['Select Functional Area','Underwriting Model','Probability of default Model'],
          'Modelling Technique' :['Select Technique','Classification', 'Regression', 'Segmentation'],
          'Algorithm' :['Select Algorithm','XGBoost','GBM', 'LightGBM', 'Random Forest', 'Logistic Regression','Decision Tree','Neural Network'],
          'Optimization Technique 1': ['Select Optimization Technique','Bayesian Optimization', 'Random Search', 'Grid Search', 'Not applicable'],
          'Optimization Technique 2': ['Select Optimization Technique','Bayesian Optimization', 'Random Search', 'Grid Search', 'Not applicable']
          }

sections_subsections ={
    'Model Scope, Purpose and Use' : [],
    'Limitations and Compensating Controls' : [],
    'Model Data' : ['3.1. Data Overview', '3.2. Data Quality Check', '3.3. Data Exclusions', '3.4. Vintage Selection & Sampling'],
    'Model Specification': ['4.1. Technical Summary','4.2. Dependent Variable','4.3. Variable transformation and selection','4.4. Final Model Selection'],
    'Model Implementation' :['6.1. Implementation Overview','6.2. Implementation Testing & Results'],
    'Model Testing': ['5.1. Testing Plan', '5.2. Overall Performance','5.3. Summarized Result','5.4. Benchmark Analysis', '5.5. Performance across Segments'],
    'Operating and Control Environment': [],
    'Ongoing Monitoring and Governance Plan': ['8.1. Monitoring Frequency & Components','8.2. Annual Model Review plan'], 
    'Reference': [],
    'Appendix': []
}

df_inputs={'df_dict':{'Sr.No':'  ','Raised By':'  ','Limitation Type':'  ','Limitation Description':'  ','Proposed compensating control':'  ','Additional Comments':'  '}

}
           
# css_inputs={
#     'style':'CommentsStyle'
# }

markdown_inputs={
    'Spacing_input':"""
    <style>
     .css-ue6h4q{
     margin-bottom: 0rem;
     min-height: 0rem;
     }

     .css-1629p8f h1{
    scroll-margin-top:0rem;
    padding:0rem;
    line-height: 0;
     }

     .st-cy {
              padding-top: 0rem;
                      }
     .st-cx {
              padding-top: 0rem;
                      }
     .st-cz {
              padding-top: 0rem;
                      }
     .st-cm {
              padding-top: 0rem;
                      }-
    </style>

#                       """,
    'Tab size':"""<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.2rem;font-weight: 600;
    }
</style>""",
    'Document Type':"""<span style="font-size: 27px;font-weight: 600;color:#000000"> Document Type </span>""",
    'Model Type':"""<span style="font-size: 27px;font-weight: 600;color:#000000"> Model Type </span>""",
    'Functional area':"""<span style="font-size: 27px;font-weight: 600;color:#000000"> Functional Area </span>""",
    'Sub Functional area':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Sub Functional Area </span>""",
    'Model Name':"""<span style="font-size: 27px;font-weight: 600;color:#000000"> Model Name </span>""",
    'Modelling Technique':"""<span style="font-size: 27px;font-weight: 600;color:#000000"> Modelling Technique </span>""",
    'Model Algorithm':"""<span style="font-size: 27px;font-weight: 600;color:#000000"> Model Algorithm </span>""",
    'Algorithm Type':"""<span style="font-size: 20px;font-weight: 600;color:#000000"> Select type of algorithm </span>""",
    'Optimization Technique':"""<span style="font-size: 27px;font-weight: 600;color:#000000">Optimization Technique </span>""",
    'Model Development Datasets':"""<span style="font-size: 27px;font-weight: 600;color:#FC2403">Model Datasets </span>""",
    'Development Dataset':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Development Data </span>""",
    'Validation Dataset':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Validation Data </span>""",
    'Additional Validation Data':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Additional Validation Data </span>""",
    'Data Dictionary':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Data Dictionary </span>""",
    'Model Variables':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Model Variables </span>""",
    'Exclusion Variable':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Exclusion Variable</span>""",
    'Performance Exclusion Variable':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Performance Exclusion Variable</span>""",    'Segmentation':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Segmentation </span>""",
    'Dependant Variable Name':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Dependent Variable Name </span>""",
    'Predicted Variable':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Predicted Variable </span>""",
    'Final model output':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Final Model Output </span>""",
    'Benchmark Model':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Benchmark Model </span>""",
    'Unique Identifier':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Unique Identifier </span>""",    
    'Vintage':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Vintage </span>""",     
    'Additional Variables for Analysis':"""<span style="font-size: 27px;font-weight: 600;color:#FFFFFF">Additional Variables </span>""",
    'Approval Tag':"""<span style="font-size: 20px;font-weight: 600;color:#000000">Approval Indicator </span>""",
    
    
    'Key Variables':"""<span style="font-size: 27px;font-weight: 600;color:#FC2403"> Key Variables </span>""",
    'Dependant variable definition':"""<span style="font-size: 20px;font-weight: 600;color:#000000"> Dependent Variable Definition </span>""",
    'Delinquency Bucket':"""<span style="font-size: 20px;font-weight: 600;color:#000000"> Target DLQ Bucket </span>""",
    'Duration Type':"""<span style="font-size: 20px;font-weight: 600;color:#000000"> Target Duration </span>""",
    'Performance Window':"""<span style="font-size: 20px;font-weight: 600;color:#000000"> Target Performance Window </span>""",
    'Data Location':"""<span style="font-size: 20px;font-weight: 600;color:#000000"> Data Location </span>""",
    'Ouput folder Location':"""<span style="font-size: 27px;font-weight: 600;color:#000000"> Output Folder Location </span>""",
}

target_defination_input={
    "Delinquency Bucket":["Select Delinquency Bucket","30+ DPD","60+ DPD","90+ DPD","120+ DPD","150+ DPD","180+ DPD","Charge-off/Bankruptcy"],
    "Duration Type":["Select Duration Type","Ever","At the end"],
    "Performance Window":["Select Performance Window","6 Months","9 Months","12 Months","15 Months","18 Months","21 Months","24 Months"]
}


variable_description={'Application_id': 'Application id ', 'vantage_score': 'Vantage Score', 'g106s': 'Months on file', 'at104s': 'Percentage of all trades opened in past 24 months to all trades', 'of20s': 'Months since oldest credit union trade opened', 'in57s': 'Total past due amount of open installment trades verified in past 12 months', 're35s': 'Average balance of open revolving trades verified in past 12 months', 'of36s': 'Months since most recent credit union delinquency', 'in34s': 'Utilization for open installment trades verified in past 12 months', 'of34s': 'Utilization for open credit union trades verified in past 12 months', 'at36s': 'Months since most recent delinquency', 'agg901': 'Number of aggregate non-mortgage balance increases over last 3 months', 'at06s': 'Number of trades opened in past 6 months', 'cv14': 'Number of deduped inquiries', 's004s': 'Average number of months trades have been on file', 'g237s': 'Number of credit inquiries in past 6 months', 'mt20s': 'Months since oldest mortgage trade opened', 'at21s': 'Months since most recent trade opened', 'fi34s': 'Utilization for open finance installment trades verified in past 12 months', 'g208s': 'Percentage of individual debt', 'g001s': 'Number of 30 days past due ratings in past 12 months', 'g206s': 'Total monthly obligation for joint account verified in past 12 months', 'g218a': 'Number of trades verified in the past 12 months that are currently 30 days past due', 's114s': 'Number of deduped inquiries in past 6 months (excluding auto and mortgage inquiries)', 'bc34s': 'Utilization for open credit card trades verified in past 12 months', 'au36s': 'Months since most recent auto delinquency', 'bc103s': 'Average balance of all credit card trades with balance > $0 verified in past 12 months', 'of57s': 'Total past due amount of open credit union trades verified in past 12 months', 'cv24': 'Total payment amount of credit card trades verified in past 3 months', 'bc02s': 'Number of open credit card trades', 'aggs904': 'Max Aggregate Monthly Spend over last 12 Months', 'aggs902': 'Aggregate Monthly Spend over last 6 Months', 'of21s': 'Months since most recent credit union trade opened', 'at35b': 'Average balance of open trades verified in past 12 months (excluding mortgage and home equity)', 'br21s': 'Months since most recent bank revolving trade opened', 'agg908': 'Max aggregate bankcard balance over last 12 months', 'bc06s': 'Number of credit card trades opened in past 6 months', 'g990s': 'Number of deduped inquiries in past 12 months', 'of02s': 'Number of open credit union trades', 'in21s': 'Months since most recent installment trade opened', 's209s': 'Months since most recent third party collection', 'g205s': 'Total monthly obligation for individual account verified in past 12 months', 're20s': 'Months since oldest revolving trade opened', 'g980s': 'Number of deduped inquiries in past 6 months', 'cv26': 'Percent of trades from active to inactive status in the past 12 months', 'cv21': 'Total payment amount of trades verified in past 3 months', 'au20s': 'Months since oldest auto trade opened', 'trv01': 'Number of months since overlimit on a bankcard', 'at33b': 'Total balance of open trades verified in past 12 months (excluding mortgage and home equity)', 'rvlr29': 'Bankcard balance for bankcard with highest balance', 'rt21s': 'Months since most recent retail trade opened', 'cv04': 'Months since most recent third-party collection occurrence', 'br27s': 'Number of currently open and satisfactory bank revolving trades 24 months or older', 'au57s': 'Total past due amount of open auto trades verified in past 12 months', 'inap01': 'Total scheduled monthly payment for open installment trades verified in past 12 months', 'at27s': 'Number of currently open and satisfactory trades 24 months or older', 'bc36s': 'Months since most recent credit card delinquency', 'au51a': 'Terms in months of most recent auto trade', 'g207s': 'Total monthly obligation for all accounts', 'in101s': 'Total balance of all installment trades verified in past 12 months', 'g250c': 'Number of 30 days past due or worse items in the past 24 months (excluding medical collection items)', 'bc97b': 'Total open to buy of closed credit cards verified in past 3 months', 'of35s': 'Average balance of open credit union trades verified in past 12 months', 'g095s': 'Months since most recent public record', 'rvlr02': 'Retail revolver trade utilization last month', 's204a': 'Total balance of non-medical third party collections verified in past 12 months', 'bc101s': 'Total balance of all credit card trades verified in past 12 months', 'at35a': 'Average balance of open trades verified in past 12 months', 'bc107s': 'Number of 30 and 60 days past due ratings on credit card trades', 'reap01': 'Total scheduled monthly payment for all revolving trades verified in past 12 months', 'g250b': 'Number of 30 days past due or worse items in the past 12 months (excluding medical collection items)', 'g213a': 'Highest balance of third party collections verified in 24 months', 'br32s': 'Maximum balance owed on open bank revolving trades verified in past 12 months', 'bc24s': 'Number of currently open and satisfactory credit card trades 6 months or older', 'fi33s': 'Total balance of open finance installment trades verified in past 12 months', 'g230s': 'Number of 60 or more days past due trades (current MOP only) with balance > $0 opened in past 24 months', 'rvlr03': ' Total bankcard revolver trade balance', 're24s': 'Number of currently open and satisfactory revolving trades 6 months or older', 'g221a': 'Number of trades verified in the past 12 months that are currently 120 days past due', 'bc29s': 'Number of open credit card trades with balance > $0 verified in past 12 months', 'trv02': 'Number of months overlimit on a bankcard over last 12 months', 'g223s': 'Number of trades prior 60 days past due, now current, verified in past 12 months', 'cv13': 'Percentage of trades ever delinquent', 'md33s': 'Total balance of open medical trades verified in past 12 months', 'target_24m': 'Actual target variable which has categories for Good and Bad customers', 'prediction': 'Probabilties Given by the model', 'model_score': 'model score is the score calculated from model probability'}
                         


models_specification={
    'models details':{'Technical Summary' : 'Statistical Estimation Technique' +\
'\n\n'
'XGBoost Algorithm and Hyperparameters:' +\
                      '\n\n'
'Introduction:' +\
                      '\n\n'
'XGBoost, short for Extreme Gradient Boosting, is a highly acclaimed machine learning algorithm renowned' +\
'for its exceptional predictive accuracy and efficient handling of large datasets. In this documentation,' +\
'we delve into the intricacies of the XGBoost algorithm, its functioning, and the significance of its' +\
'essential hyperparameters.' +\
                      '\n\n'
'Understanding XGBoost:' +\
                      '\n\n'
'1.Gradient Boosting:' +\
                      '\n\n'
'XGBoost falls within the gradient boosting family of algorithms, which leverages an ensemble approach by' +\
'combining predictions from multiple weak learners, typically decision trees. The unique feature of gradient ' +\
# 'boosting lies in its sequential training of trees to rectify errors made by the ensemble of previously' +\ 
'trained trees, progressively enhancing the model overall performance.' +\
                      '\n\n'
'2.Core Features of XGBoost:' +\
                      '\n\n'
'XGBoost offers several key features that make it a favored choice for machine learning tasks:' +\
                      '\n\n'
'1.Regularization:' +\
'\nIncorporating L1 (Lasso) and L2 (Ridge) regularization techniques, XGBoost effectively ' +\
'combats overfitting, a common challenge in machine learning.' +\
                      '\n\n'
'2.Sparsity Awareness:' +\
'\nThis algorithm excels at handling sparse data by optimizing memory usage and processing speed.' +\
                      '\n\n'
'3.Customizable Objective Functions:' +\
'\nUsers can define their own loss functions and evaluation metrics, ' +\
'making it adaptable to a variety of problem domains.' +\
                      '\n\n'
'4.Parallel and Distributed Computing:' +\
'\nXGBoost capitalizes on multi-core processors and distributed computing ' +\
'frameworks, which accelerates model training on extensive datasets.' +\
                      '\n\n'
'5.Out-of-the-Box Support for Missing Values:' +\
'\nIt adeptly manages missing data, reducing the need for ' +\
'extensive data preprocessing.' +\
                      '\n\n'
'6.Cross-Validation:' +\
'\nXGBoost includes built-in support for cross-validation, simplifying the hyperparameter ' +\
'tuning process and performance assessment.' +\
                      '\n\n'
'3.How XGBoost Works:' +\
                      '\n\n'
'1.Initial Prediction:' +\
'\nXGBoost commences with an initial prediction, often set as the mean of the target' +\
'variable for regression or the class distribution for classification.' +\
                      '\n\n'
'2.Residual Calculation:' +\
'\nIt computes residuals, representing the differences between the actual target' +\
'values and the current model"s predictions.' +\
                      '\n\n'
'3.Building Trees:' +\
'\nXGBoost constructs decision trees to fit these residuals. Each iteration adds a new' +\
'tree with the goal of minimizing the loss function.' +\
                      '\n\n'
'4.Shrinkage:' +\
'\nThe algorithm employs a shrinkage parameter (learning rate) to control the step size during' +\
'tree construction, enhancing robustness against overfitting.' +\
                      '\n\n'
'5.Regularization:' +\
'\nL1 and L2 regularization techniques penalize large coefficients and enable tree pruning' +\
'to enhance model generalization.' +\
                      '\n\n'
'6.Ensemble Building:' +\
'\nThe final prediction is a weighted sum of predictions from all trees in the ensemble.' +\
                      '\n\n'
'4.Hyperparameters:' +\
                      '\n'
                      '\n'
'Hyperparameters in XGBoost play a pivotal role in tailoring the models behavior and optimizing its performance.' +\
'Here are some of the most crucial hyperparameters:' +\
                      '\n\n'
'1.n_estimators:' +\
'\nDefines the number of boosting rounds (trees) to be built, with higher values potentially' +\
'leading to overfitting.' +\
                      '\n\n'
'2.learning_rate:' +\
'\nThe learning rate, or shrinkage parameter, governs the step size during tree construction.' +\
'Smaller values require more boosting rounds but can enhance model generalization.' +\
                     '\n\n'
'3.max_depth:' +\
'\nThis hyperparameter determines the maximum depth of each decision tree, controlling model' +\
'complexity and guarding against overfitting.' +\
                      '\n\n'
'4.min_child_weight:' +\
'\nSpecifies the minimum sum of instance weight needed in a child node,' +\
'helping control overfitting.' +\
                      '\n\n'
'5.gamma:' +\
'\nA regularization parameter that sets a threshold for further node partitioning, with higher' +\
'values reducing the number of splits.' +\
                      '\n\n'
'6.subsample:' +\
'\nDenotes the fraction of samples used for growing trees, with smaller values mitigating overfitting.' +\
                      '\n\n'
'7.colsample_bytree:' +\
'\nDetermines the fraction of features utilized for tree building, aiding in feature' +\
'selection and overfitting prevention.' +\
                      '\n\n'
'8.lambda (L2 regularization term) and alpha (L1 regularization term):' +\
'\nControl the strength of regularization in the model.' +\
                      '\n\n'
'9.objective:' +\
'\nThe loss function to optimize, such as "reg:squarederro" for regression or "binary:logistic"' +\
'for binary classification.' +\
                      '\n\n'
'10.eval_metric:' +\
'\nThe evaluation metric used during training, like "rmse" for regression and "logloss" for classification.' +\
                      '\n\n'
'11.early_stopping_rounds:' +\
'\nIf specified, the model will halt training if no improvement is observed for a' +\
'specified number of rounds.'
                      '\n\n'
'These hyperparameters empower users to fine-tune the XGBoost model to match the specific' +\
'requirements of their machine learning tasks.' +\
                      '\n\n'
'Hyperparameter Optimization Technique: Bayesian Optimization' +\
                      '\n\n'
'Introduction:' +\
                      '\n\n'
'Bayesian Optimization is an advanced and effective optimization methodology that plays a pivotal' +\
'role in the realm of machine learning, specifically in the context of hyperparameter tuning.'+\
'By harnessing probabilistic modeling, it guides the selection of hyperparameters in a systematic' +\
'and intelligent manner, leading to superior model performance. This section provides a comprehensive' +\
'understanding of Bayesian Optimization, its mechanics, and its significant impact on the model development process.' +\
                      '\n\n'
'Bayesian Optimization stands out as a sequential model-based optimization technique, tailored for' +\
'the efficient exploration of hyperparameter configurations, particularly when the objective function' +\
'is expensive or lacks an explicit analytical form. In the domain of machine learning, this objective' +\
'function typically represents evaluation metrics that gauge a models performance. The central objective' +\
# 'of Bayesian Optimization is to identify the set of hyperparameters that maximizes the objective function,' +\ 
'thus elevating the model"s overall performance.' +\
                      '\n\n'
'Salient Features of Bayesian Optimization' +\
                      '\n\n'
'Bayesian Optimization is endowed with a host of features that make it an indispensable tool' +\
'in the model development process:' +\
                      '\n\n'
'1.Probabilistic Model:' +\
'\nBayesian Optimization capitalizes on probabilistic models, primarily' +\
'Gaussian processes, to capture the intricate relationships between hyperparameters' +\
'and the objective function.' +\
                      '\n\n'
'2.Acquisition Functions:' +\
'\nIt makes use of acquisition functions to intelligently decide the next' +\
'hyperparameter configuration to evaluate, striking a balance between exploration (discovering' +\
'uncharted territories) and exploitation (exploiting promising regions).' +\
                      '\n\n'
'3.Sequential Optimization:' +\
'\nThis optimization process unfolds sequentially, with the probabilistic' +\
'model of the objective function being built and updated iteratively, rendering it significantly' +\
'more efficient than rudimentary methods like grid search or random search.' +\
                      '\n\n'
'4.Model Selection:' +\
'\nBayesian Optimization extends its utility to the selection of the most suitable' +\
'machine learning model, optimizing hyperparameters for different model architectures.' +\
                      '\n\n'
'5.Parallelization:' +\
"\nIt's adaptable to parallelization, allowing simultaneous evaluation of multiple" +\
'hyperparameter configurations, thereby reducing optimization time.' +\
                      '\n\n'
'The Inner Workings of Bayesian Optimization:' +\
                      '\n\n'
'1.Initial Random Exploration:' +\
'\nBayesian Optimization commences by performing an initial random' +\
'exploration, generating a set of random hyperparameter configurations to collect data points' +\
'essential for constructing the initial probabilistic model.' +\
                      '\n\n'
'2.Probabilistic Modeling:' +\
'\nIt employs a probabilistic model, frequently a Gaussian process,' +\
'to model the distribution of the objective function across the hyperparameter space. The model' +\
'estimates the mean and the uncertainty of the objective function.' +\
                      '\n\n'
'3.Acquisition Function:' +\
'\nAn acquisition function, such as Expected Improvement (EI) or Probability' +\
'of Improvement (PI), takes center stage in deciding the subsequent hyperparameter configuration to' +\
'evaluate. This function diligently balances exploration, by targeting unexplored regions, and ' +\
'exploitation, by concentrating on regions with high expected improvement.' +\
                      '\n\n'
'4.Objective Function Evaluation:' +\
'\nThe selected hyperparameter configuration undergoes evaluation on ' +\
'the objective function, and the outcome is employed to refine the probabilistic model.' +\
                      '\n\n'  
'5.Iterative Procedure:' +\
'\nSteps 3 and 4 constitute an iterative loop, continuing until a predefined ' +\
'stopping criterion is met, which could be a maximum number of iterations or a ' +\
'convergence threshold.' +\
                      '\n\n'
'6.Final Optimal Configuration:' +\
'\nThe ultimate and optimal hyperparameter configuration is identified' +\
"based on the probabilistic model's predictions" +\
                      '\n\n'
'Harnessing the Power of Bayesian Optimization:' +\
                      '\n\n'
"Bayesian Optimization emerges as a sophisticated and highly proficient approach to the intricate task of hyperparameter optimization in machine learning models. By virtue of its probabilistic modeling and smart acquisition functions, it deftly navigates the challenging and high-dimensional hyperparameter space to pinpoint configurations that propel the model's performance to new heights. Whether you're finetuning a model's hyperparameters or scrutinizing various model architectures, Bayesian Optimization serves as an invaluable asset in your machine learning toolkit, eliminating the need for exhaustive and resource-intensive hyperparameter searches.",





                      'Dependent Variable' : '',
                      'Variable transformation and selection' : '',
                      'Final Model Selection' : ''
                     
                     }                     
}

model_scope = { 'main' : 'The new model will be used for underwriting new applications (i.e. making the approve/decline decisions).' +\
                        'Currently, underwriting policy usesscore. The new score is expected to replace the existing score along with other policy' +\
                        'criteria. The new score is developed using Extreme gradient boosting.'}
