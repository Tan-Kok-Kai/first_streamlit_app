# Import neccessary packages
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
# Import Snowflake modules
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T
from snowflake.snowpark import Window
from snowflake.snowpark.functions import col
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
    #"role": "ACCOUNTADMIN",
    #"warehouse": "tasty_ds_wh",
    #"database": "NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE",
    #"schema": "analytics",
}

# Create Snowpark session
session = Session.builder.configs(connection_parameters).create()

# Define the app title and favicon
st.set_page_config(page_title='ICP ASG 3', page_icon="favicon.ico")

# Tabs set-up
tab1, tab2, tab3, tab4, tab5 = st.tabs(['tab1', 'tab2', 'tab3', 'tab4', 'tab5'])

with tab1:
    st.title('Title')
    st.subheader('Sub Title')
    
with tab2:
    st.title('Title')
    st.subheader('Sub Title')
    
with tab3:
    st.title('Title')
    st.subheader('Sub Title')
    
with tab4:
    st.title('Title')
    st.subheader('Sub Title')
    
with tab5:
    import bz2 
    from datetime import datetime, timedelta
    menu_dfs = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.raw_pos.menu")
    truck_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.raw_pos.truck")
    history_df = session.table("FROSTBYTE_POWERBI.ANALYTICS.INVENTORY_MANAGEMENT")
    history_df = history_df.filter(F.col('ORDER_YEAR') == 2022)
    history_df = history_df.filter(F.col('ORDER_MONTH') >= 10)
    truck_df = truck_df.with_column('LAST_DATE', F.iff(F.col("TRUCK_ID") == F.col('TRUCK_ID'), "2022-10-18", '0'))
    truck_df = truck_df.withColumn("DAYS_OPENED", F.datediff("day", F.col("TRUCK_OPENING_DATE"), F.col('LAST_DATE')))
    menu_df = menu_dfs.to_pandas()
    truck_df = truck_df.to_pandas()
    #im = pickle.load(open('inventory_model.sav', 'rb'))
    with bz2.BZ2File('rf.pkl', 'rb') as compressed_file:
        im = pickle.load(compressed_file)
    st.title('Inventory Management')
    st.caption('This inventory management is mainly for truck owners to have an idea on the sales and demand in the next 30 days. \
            First, pick your Truck ID which ranges from 1 to 450. Next, select the food that you are interested in predicting, then\
            select the next number of days you would like to predict.')
    st.subheader('Truck')
    truck_df = truck_df.set_index('TRUCK_ID')

    # Let's put a pick list here so they can pick the fruit they want to include 
    trucks_selected = st.selectbox("Pick a truck:", list(truck_df.index))
    trucks_to_show = truck_df.loc[[trucks_selected]]
    history_df = history_df.filter(F.col('TRUCK_ID') == trucks_selected)
    history_df = history_df.to_pandas()
    #st.dataframe(history_df)
    # Display the table on the page.
    #st.dataframe(trucks_to_show)
    trucks_to_show.reset_index(inplace=True)
    merge = pd.merge(menu_df, trucks_to_show, on=['MENU_TYPE_ID'],how='outer', indicator=True)
    final_scaled = merge[merge['_merge'] == 'both'].drop('_merge', axis = 1)
    st.subheader('Menu')
    menu_df = final_scaled.set_index('MENU_ITEM_NAME')
    # Let's put a pick list here so they can pick the fruit they want to include 
    menu_selected = st.multiselect("Pick some foods:", list(menu_df.index))
    menu_to_show = menu_df.loc[menu_selected]
    #st.dataframe(menu_to_show)
    st.subheader('Prediction')
    final = menu_to_show[['MENU_ITEM_ID', 'TRUCK_ID', 'SALE_PRICE_USD', 'EV_FLAG', 'MENU_TYPE_ID',
                          'ITEM_SUBCATEGORY', 'COST_OF_GOODS_USD', 'ITEM_CATEGORY', 'DAYS_OPENED']]
    unique_menuid = list(final['MENU_ITEM_ID'].unique())
    #st.text(unique_menuid)
    history_df = history_df[history_df['MENU_ITEM_ID'].isin(unique_menuid)]
    #st.dataframe(history_df)
    final['TEMPERATURE_OPTION'] = np.where(final['ITEM_SUBCATEGORY'] == 'Cold Option', 0, np.where(final['ITEM_SUBCATEGORY'] 
                                                                                                  == 'Warm Option', 1, 2))
    final['ITEM_CATEGORY_Main'] = np.where(final['ITEM_CATEGORY'] == 'Main', 1, 0)
    final['ITEM_CATEGORY_Beverage'] = np.where(final['ITEM_CATEGORY'] == 'Beverage', 1, 0)
    final['ITEM_CATEGORY_Dessert'] = np.where(final['ITEM_CATEGORY'] == 'Dessert', 1, 0)
    final['ITEM_CATEGORY_Snack'] = np.where(final['ITEM_CATEGORY'] == 'Snack', 1, 0)
    #st.subheader('Day Slider')
    numdays = st.slider('Predict the next x days', 1, 30)
    #st.write(numdays)
    # '2022-11-01'
    final = pd.concat([final]*numdays, ignore_index=True)
    #st.dataframe(final)
    #st.write(menu_to_show)
    index = 0
    final['ORDER_YEAR'] = 2023
    final['ORDER_MONTH'] = 1
    final['ORDER_DAY'] = 2
    datetime_str = '2022-11-01'
    datetime_object = datetime.strptime(datetime_str, '%Y-%m-%d')
    past_demand = history_df['DEMAND'].sum() / 32 * numdays
    history_df['SALES_GENERATED'] = history_df['DEMAND'] * history_df['UNIT_PRICE']
    past_sales = history_df['SALES_GENERATED'].sum()/32*numdays
    while numdays > 0:
        preddate = datetime_object + timedelta(days=numdays)
        for i in range(len(menu_to_show)):
            final['ORDER_YEAR'][index] = preddate.year
            final['ORDER_MONTH'][index] = preddate.month
            final['ORDER_DAY'][index] = preddate.day
            index += 1
        numdays -= 1
    final['UNIT_PRICE'] = final['SALE_PRICE_USD']
    final_df = final[['MENU_ITEM_ID', 'TRUCK_ID','ORDER_YEAR', 'ORDER_MONTH', 'ORDER_DAY','UNIT_PRICE', 'EV_FLAG', 'DAYS_OPENED',
                      'MENU_TYPE_ID', 'TEMPERATURE_OPTION', 'COST_OF_GOODS_USD', 'ITEM_CATEGORY_Main', 'ITEM_CATEGORY_Beverage'
                      ,'ITEM_CATEGORY_Dessert','ITEM_CATEGORY_Snack']]
    #st.dataframe(final)
    #st.dataframe(final_df)
    if st.button("Predict demand"):
        pred = im.predict(final_df)
        #st.text(pred)
        predlist = []
        counter = -1
        index = 0
        count = 0
        for i in range(len(menu_to_show)):
            counter += 1
            index = counter
            while index < len(pred):
                count += pred[index] 
                index += len(menu_to_show)
            predlist.append(count)
            count = 0
        #st.text(predlist)
        menu = menu_to_show.reset_index()
        str1 = ''
        present_sales = 0
        for i in range(len(menu)):
            str1 = menu['MENU_ITEM_NAME'][i]
            st.subheader(str1)
            st.text('The predicted demand for ' + str1 + ' is ' + str(int(predlist[i])))
            st.text("The sales generated by " + str1 + " will be ${:.2f}".format(predlist[i] * menu['SALE_PRICE_USD'][i]))
            present_sales += predlist[i] * menu['SALE_PRICE_USD'][i]
            #st.text("The average sales generated in this duration is $4000, increase in 200%")
            #st.text("The profit generated will be ...")
        #st.text(past_demand)
        #st.text(past_sales)
        # Sample data (replace with your actual data)
        past = {'Demand': past_demand}
        future = {'Demand': sum(predlist)}

        # Streamlit app
        st.title('Past and Future Comparison')
        st.caption('Past data is calculated by taking the past 1 month of historical data and \
                   averaging to the number of days stated by the slider. Future data is calculated by the sum of predicted values.')
        # Plotting the bar chart
        fig, ax = plt.subplots()
        products = list(past.keys())
        past_values = list(past.values())
        future_values = list(future.values())
        bar_width = 0.35
        indices = np.arange(len(products))
        p1 = ax.bar(indices, past_values, bar_width, label='Past')
        p2 = ax.bar(indices + bar_width, future_values, bar_width, label='Future')

        ax.set_xlabel('Products')
        ax.set_ylabel('Demand')
        ax.set_title('Past vs Future (Demand)')
        ax.set_xticks(indices + bar_width / 2)
        ax.set_xticklabels(products)
        ax.legend()

        # Display the bar chart in Streamlit
        st.pyplot(fig)
        # Sample data (replace with your actual data)
        past = {'Sales Generated': past_sales}
        future = {'Sales Generated': present_sales}

        # Streamlit app
        #st.title('Past and Future Comparison')

        # Plotting the bar chart
        fig, ax = plt.subplots()
        products = list(past.keys())
        past_values = list(past.values())
        future_values = list(future.values())
        bar_width = 0.35
        indices = np.arange(len(products))
        p1 = ax.bar(indices, past_values, bar_width, label='Past')
        p2 = ax.bar(indices + bar_width, future_values, bar_width, label='Future')

        ax.set_xlabel('Products')
        ax.set_ylabel('Sales Generated')
        ax.set_title('Past vs. Future (Sales Generated)')
        ax.set_xticks(indices + bar_width / 2)
        ax.set_xticklabels(products)
        ax.legend()

        # Display the bar chart in Streamlit
        st.pyplot(fig)
