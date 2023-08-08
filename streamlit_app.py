# Import neccessary packages
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import date, datetime, timedelta
import pickle
import json
import math
import plotly.graph_objects as go
import bz2 
import matplotlib.pyplot as plt
# Import Snowflake modules
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T
from snowflake.snowpark import Window
from snowflake.snowpark.functions import col, date_add, to_date, desc, row_number
from datetime import datetime

# Get account credentials from a json file
with open("account.json") as f:
    data = json.load(f)
    username = data["username"]
    password = data["password"]
    account = data["account"]

# Specify connection parameters
connection_parameters = {
    "account": account,
    "user": username,
    "password": password,
}

# Create Snowpark session
session = Session.builder.configs(connection_parameters).create()

# Define the app title and favicon
st.set_page_config(page_title='ICP ASG 3', page_icon="favicon.ico")

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Predicting Future Sales [Shi Wei]', 'Predicting Customer Spending [Ernest]', 'Predicting Customer Churn [Gwyneth]', 'Guo Fung', 'Demand Forecasting [Kok Kai]'])

with tab1:
    a = 1

with tab2:
    a=2

with tab3:
    a=3

with tab4: 
    a=4
    
with tab5:
    st.title('Predicting Customer Churn :worried:')
    # introduction of web tab
    st.write("In this web tab, Tasty Bytes management will have the ability to obtain details about churn customers across various customer segments. They can **leverage the insights and advice provided to take proactive measures** aimed at retaining these customers in order for Tasty Bytes to **reach their goal** of increasing Net Promoter Score (NPS) from 3 to 40 by year end 2023. By **effectively addressing churn**, this will ensure that customers are engaged and shows strong loyalty and satisfaction towards Tasty Bytes, signifying that the NPS score is **poised to increase**.")
    st.write("Additionally, the management will also be able to **experiment** with customer's details to **predict** whether they will be likely to churn or not.")
    st.write("Customers are likely to churn when their predicted days to next purchase is **more than 14 days**.")
    st.header('Details of Churn Customers :face_with_monocle:')

    # loading of dataset 
    def load_next_purchase_cust_seg():
        data = pd.read_csv("NextPurchaseCustSeg2.csv")
        return data
    
    # filter csv based on customer segment chosen 
    def filter_cust_seg(data):
        filtered_cust_seg = next_purchase_cust_seg[next_purchase_cust_seg['CLUSTER']==data]
        return filtered_cust_seg
    
    # filter csv based on whether customer likely to churn or not
    def filter_cust_churn(data):
        filtered_cust_churn = filtered_cust_seg[filtered_cust_seg["CHURN_STATUS"] == data]
        return filtered_cust_churn
    
    # convert dataframe to csv 
    def convert_df(df):
       return df.to_csv(index=False).encode('utf-8')

    next_purchase_cust_seg = load_next_purchase_cust_seg()
    next_purchase_cust_seg.rename(columns={'CHURN': 'CHURN_STATUS'}, inplace=True)

    cust_seg_label_mapping = {0: 'Middle Value', 1: 'Low Value', 2:'High Value'}
    next_purchase_cust_seg['CLUSTER'] = next_purchase_cust_seg['CLUSTER'].map(cust_seg_label_mapping)
    # select customer segment
    cust_seg = st.selectbox(
    'Select the information of the customer segment that you would like to view',
    options = ['Low Value (Customers who buy less frequently and generate lower sales)', 
                             'Middle Value (Customers who make average purchases)', 
                             'High Value (Customers who make frequent purchases and generate higher sales)'])
    
    # show percentage of churn and not churn of customer segment chosen using bar charts
    cust_seg_option = cust_seg.split('(')[0].strip()
    filtered_cust_seg = filter_cust_seg(cust_seg_option)
    churn_label_mapping = {0: 'Not Churn', 1: 'Churn'}
    filtered_cust_seg['CHURN_STATUS'] = filtered_cust_seg['CHURN_STATUS'].map(churn_label_mapping)
    cust_churn_bar = filtered_cust_seg['CHURN_STATUS'].value_counts()
    st.bar_chart(data = cust_churn_bar)

    # show details of cust likely to churn 
    st.write("Details of customers likely to churn")
    churn_cust = filter_cust_churn("Churn")
    not_churn_cust = filter_cust_churn("Not Churn")
    customer_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.raw_customer.customer_loyalty")
    us_customer_df_sf = customer_df.filter(F.col("COUNTRY")=="United States")
    us_customer_df = us_customer_df_sf.to_pandas()
    us_customer_df = us_customer_df[us_customer_df['CUSTOMER_ID'].isin(churn_cust['CUSTOMER_ID'])]
    us_customer_df = pd.merge(us_customer_df, churn_cust, on='CUSTOMER_ID', how='inner')
    us_customer_df = us_customer_df.sort_values(by='PREDICTED')
    us_customer_df = us_customer_df.reset_index(drop=True)
    cust_to_show = us_customer_df[["FIRST_NAME", "LAST_NAME", "GENDER", "MARITAL_STATUS", "CHILDREN_COUNT", "BIRTHDAY_DATE", "E_MAIL", "PHONE_NUMBER", "TOTAL_SPENT", "TOTAL_ORDER", "YEARS_WITH_US", "PREDICTED"]]
    cust_to_show.rename(columns={'PREDICTED': 'PREDICTED_DAYS_TO_NEXT_PURCHASE'}, inplace=True)
    cust_to_show['YEARS_WITH_US'] = cust_to_show['YEARS_WITH_US'].round().astype(int)
    cust_to_show['PREDICTED_DAYS_TO_NEXT_PURCHASE'] = cust_to_show['PREDICTED_DAYS_TO_NEXT_PURCHASE'].astype(int)
    st.dataframe(cust_to_show)

    # give insights about the churn customers
    avg_churn_recency = str(math.floor(churn_cust['RECENCY_DAYS'].mean()))
    avg_not_churn_recency = str(math.floor(not_churn_cust['RECENCY_DAYS'].mean())) 
    min_predicted = str(cust_to_show['PREDICTED_DAYS_TO_NEXT_PURCHASE'].min())
    max_predicted = str(cust_to_show['PREDICTED_DAYS_TO_NEXT_PURCHASE'].max())
    avg_predicted = str(math.floor(cust_to_show['PREDICTED_DAYS_TO_NEXT_PURCHASE'].mean()))
    st.subheader("Insights :mag_right:")
    st.write("Out of all the " + str(cust_seg_option) + " customers, "+ str(len(cust_to_show)) + " of them are **likely to churn** as the average time since their last order is approximately **" + str(avg_churn_recency) + " days**, compared to unlikely to churn customers of " + str(avg_not_churn_recency) + " days.")
    st.write("These customers have a predicted **"+ min_predicted + "-"+ max_predicted + " days** to next purchase range, with the average customers having a predicted " + avg_predicted + " days to next purchase.")
    csv = convert_df(cust_to_show)

    # give advice on how to retain the churn customers
    st.subheader("Advice to retain the churn customers :bulb:")
    if cust_seg_option == "High Value":
        st.write("Since these customers are of high value, it will be **crucial** to implement targeted retention strategies to address their potential churn as they **contribute to a significant portion of Tasty Bytes sales**.")
        st.write("The reasons behind **why** these customers are showing signs of potential churn despite contributing so much to Tasty Bytes’ sales and making frequent orders should be **investigated**.")
        st.write("To retain these customers, **exclusive menu items** can be offered to these customers to provide them with a **unique and premium experience**, creating a **sense of loyalty and making them less likely to switch to competitors**.")
        st.write("Another suggestion is to focus on the high value customers that are more likely to purchase in the next **"+ min_predicted + "-" + avg_predicted + " days**, rather than customers predicted to purchase in eg. " + max_predicted + " days. This range is derived from taking the minimum and the average number of predicted days until next purchase of customers in this segment, pinpointing a timeframe that **strikes a balance between immediate action and a reasonable lead time for retention**. This can impact overall retention rates more **effectively** and **generate quicker positive results** compared to those with longer predicted purchase timelines.")  
    elif cust_seg_option == "Middle Value":
        st.write("Even though these customers are of middle value, they still play a significant role in the overall business. It is still **essential** to address their potential churn to maintain a **healthy customer base**.")
        st.write("Feedback can be gathered from these customers through **surveys or feedback forms** to help **identify areas for improvement and tailor services to better meet their needs**. Responding to their concerns and suggestions can demonstrate that their **opinions are valued**, fostering a **positive customer experience**.")
        st.write("To retain these customers, implementing **personalised special offers and discounts based on their preferences and order history** can be a strategic approach. This will encourage **repeat business** and foster a **sense of appreciation** amongst these customers.")
        st.write("Another suggestion is to focus on the middle value customers that are more likely to purchase in the next **"+ min_predicted + "-" + avg_predicted + " days**, rather than customers predicted to purchase in eg. " + max_predicted + " days. This range is derived from taking the minimum and the average number of predicted days until next purchase of customers in this segment, pinpointing a timeframe that **strikes a balance between immediate action and a reasonable lead time for retention**. This can impact overall retention rates more **effectively** and **generate quicker positive results** compared to those with longer predicted purchase timelines.") 
    else:
        st.write("While low value customers may not contribute as much sales as high or middle value customers, it is still **important** to address their potential churn and **explore ways** to retain them as they still represent a portion of Tasty Bytes’ customer base.")
        st.write("Analysing these customer’s order history and feedback through **surveys or feedback forms** can help to identify **customer’s preferences, buying behaviour and pain points** to be addressed to improve the overall customer experience.")
        st.write("To retain these customers, **attractive discounts or promotions such as cost-effective deals** can be offered to **incentivize repeat purchases** and even an increase in order frequency. This may also potentially **convert some of them into higher value customers** in the long run, contributing positively to the overall business growth.")
        st.write("Another suggestion is to focus on the low value customers that are more likely to purchase in the next **"+ min_predicted + "-" + avg_predicted + " days**, rather than customers predicted to purchase in eg. " + max_predicted + " days. This range is derived from taking the minimum and the average number of predicted days until next purchase of customers in this segment, pinpointing a timeframe that **strikes a balance between immediate action and a reasonable lead time for retention**. This can impact overall retention rates more **effectively** and **generate quicker positive results** compared to those with longer predicted purchase timelines.") 
    st.write("With customer details, **targeted marketing strategies such as email marketing** can be implemented to deliver personalised messages, promotions and offers that resonate with each customer. This makes the emails become more **engaging and relevant**, fostering a sense of value and loyalty amongst customers.")

    st.download_button(
       "Press to Download Details of " + cust_seg_option + " Customers Likely to Churn",
       csv,
       "churn_cust_" + str(cust_seg_option) +".csv",
       "text/csv",
       key='download-csv')



    st.header('Predicting whether customers churn :face_with_one_eyebrow_raised:')

    # loading model
    with open('NextPurchase2.pkl', 'rb') as file:
        npm = pickle.load(file)

    # total spending input
    avg_spending = math.floor(us_customer_df['TOTAL_SPENT'].mean()) 
    spending_option = st.number_input("Input Total Spending of Customer", min_value=1, value = avg_spending)

    st.write('You selected:', spending_option)

    # years with us input
    max_year = datetime.today().year - 2019
    years_list = [str(year) for year in range(1, max_year + 1)]
    years_with_us_option = st.selectbox(
    'Select the Number of Years the Customer has been with Tasty Bytes',years_list)

    st.write('You selected:', years_with_us_option)

    # select no of orders
    total_orders_option = st.number_input("Input Number of Orders", min_value=1)

    st.write('You selected:', total_orders_option)

    # input last purchase date
    first_date = datetime(2019, 1, 1)
    today_date = datetime(2022, 11, 1)
    date = st.date_input("Enter the customer's last purchase date", first_date, first_date,today_date,today_date)

    st.write('You selected:', date)

    # predict churn button
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def click_button():
        st.session_state.clicked = True

    st.button('Predict whether Customer is Likely to Churn', on_click=click_button)

    # predict whether customer is likely to churn 
    if st.session_state.clicked:
        #calculate average of trans_datediff1
        trans_datediff1 = next_purchase_cust_seg['TRANS_DATEDIFF1'].mean()

        #calculate average of trans_datedif2
        trans_datediff2 = next_purchase_cust_seg['TRANS_DATEDIFF2'].mean()

        #calculate average of avg(days_between)
        avg_days_between = next_purchase_cust_seg['AVG(DAYS_BETWEEN)'].mean()
   
        #calculate average of min(days_between)
        min_days_between = next_purchase_cust_seg['MIN(DAYS_BETWEEN)'].mean()

        #calculate average of max(days_between)
        max_days_between= next_purchase_cust_seg['MAX(DAYS_BETWEEN)'].mean()

        #calculate monetary value 
        monetary = spending_option / int(years_with_us_option)

        #calculate frequency
        frequency = int(total_orders_option) / int(years_with_us_option)

        #calculate recency
        recency = (today_date.date() - date).days

        #calculate monetary cluster 
        if monetary <= 566:
            monetary_cluster = 0
        elif monetary <= 795:
            monetary_cluster = 1
        else:
            monetary_cluster = 2
        
        #calculate frequency cluster 
        if frequency <= 14:
            frequency_cluster = 0
        elif frequency <= 20:
            frequency_cluster = 1
        else:
            frequency_cluster = 2

        #calculate recency cluster 
        if recency <= 12:
            recency_cluster = 2
        elif recency <= 33:
            recency_cluster = 1
        else:
            recency_cluster = 0

        #calculate overall score 
        if recency <= 12 and frequency >= 30 and monetary >= 1259:
            overall_score = 6
        elif recency <= 33 and frequency >= 28 and monetary >= 1247:
            overall_score = 5
        elif recency <= 86 and frequency >= 24 and monetary >= 1006:
            overall_score = 4
        elif recency <= 96 and frequency >= 14 and frequency <20 and monetary >= 566 and monetary <794:
            overall_score = 1
        elif recency <= 98 and frequency >= 20 and monetary >= 794:
            overall_score = 3
        elif recency <= 115 and frequency >= 19 and monetary >= 776:
            overall_score = 2
        else:
            overall_score = 0 

        # making of dataframe to input to model 
        data = [[spending_option, years_with_us_option, monetary, frequency, total_orders_option, recency, max_days_between, min_days_between, avg_days_between, trans_datediff1, trans_datediff2, recency_cluster, frequency_cluster, monetary_cluster, overall_score]]
        final = pd.DataFrame(data, columns = ['TOTAL_SPENT','YEARS_WITH_US','MONETARY_VALUE','CUSTOMER_FREQUENCY','TOTAL_ORDER','RECENCY_DAYS','MAX(DAYS_BETWEEN)','MIN(DAYS_BETWEEN)','AVG(DAYS_BETWEEN)','TRANS_DATEDIFF1','TRANS_DATEDIFF2','CUST_REC_CLUSTER','CUST_FREQ_CLUSTER','CUST_MONETARY_CLUSTER','OVERALL_SCORE'])

        pred = npm.predict(final)
        pred = pred.round().astype(int)

        # show prediction results 
        if pred[-1] <= 14:
            st.write("Customer is not likely to churn.")
        else:
            st.write("Customer is likely to churn. It is predicted that they are likely to make a purchase in the next " + str(pred[-1]) + " days, exceeding the 14-day benchmark for potential churn by " + str(pred[-1] - 14) + " days.") 
