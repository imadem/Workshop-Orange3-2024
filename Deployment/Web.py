## Created by I Made Murwantara for Orange3 Training 2024
## Faculty of Computer Science
## Universitas Pelita Harapan

import streamlit as st
import Orange
import pickle
import pandas as pd
import numpy as np
import plotly.figure_factory as ff

#########################  CONTAINERS  ##############################


# Sidebar
sd_bar = st.sidebar

# Container for the Header
header_cont = st.container()

# Container for the Dataset
dataset_cont = st.container()

# Container for the Features
features_cont = st.container()

# Container for Model Prediction
modelPrediction_cont = st.container()

############# LOAD MODEL ####################

with open("attrition.pkcls","rb") as model:
    loaded_model = pickle.load(model)

##### GET DATA CACHE FUNCTION ###############

@st.cache_data
def get_data():
    df = pd.read_csv("attrition.csv", index_col=0)
    return df



########################## SIDEBAR ###################################


with sd_bar:
    st.markdown("## User Input (Job Roles)")

def get_user_input():

    df = get_data()
    JobRole = np.array(df["JobRole"])
    JobRole_sorted = np.unique(JobRole)
    
    JobRole_val = sd_bar.selectbox(label = "Job Role", options = JobRole_sorted, index = 0)
 

 # define Orange domain
    JobRole = Orange.data.DiscreteVariable("JobRole",[JobRole_val])
    
    domain = Orange.data.Domain([JobRole])

    # input values X
    X = np.array([[0]])

    # in this format, the data is now ready to be fed to StackModel
    user_input = Orange.data.Table(domain, X)

    return user_input


df_userinput = get_user_input()

###########################   DATASET  ##################################

    
with dataset_cont:
    st.markdown("## Dataset")
    st.markdown("Employee Attrition"
             "The last column (churn) is the target showing whether the employee withdraw from the job role (yes) or not (no).")
    df = get_data()
    #df = df.drop("Unnamed: 0", axis=1)
    
    st.dataframe(df)
#############################   MODEL PREDICTION   #########################

with modelPrediction_cont:
    st.markdown("## Model Prediction")


    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### Input")
        st.write("JobRole:  ", df_userinput[0,0])
        

    probs = loaded_model(df_userinput[0], 1)
    prob_no = probs[0]
    prob_yes = probs[1]
        
    with right_col:
    
        st.markdown("### Prediction Probabilities")
        st.markdown("Given the data on the left, the "
                "probability this employee will not withdraw, is:")
        st.write("churn (no): ", round(prob_no*100, 1), "%")
        st.write("churn (yes): ", round(prob_yes*100,1), "%")

###### PLOT ############

hist_data = [df.Age]
group_labels = ['Age']
# Create distplot with custom bin_size
fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.5])

# Plot!
st.plotly_chart(fig, use_container_width=True)