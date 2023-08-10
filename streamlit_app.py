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
    st.title('Demand Forecasting')
    st.caption('This demand forecasting is mainly for truck owners to have an idea on their sales and demand of menu within the next 30 days. \
            It aims to target the high level goal of 25% YoY sales growth over 5 years.\
               The model has learnt the trend of food demand daily/monthly, it will predict the menu with high accuracy and\
               help users understand the trend.\
               Gaining insights into future demand allow truck owners to optimize their menu by focusing on food high in demand\
               , which will attract more customers, increase in profit and ultimately increase in sales!\
               \n\nPlease note that as of August 2021, apps on the free community tier of Streamlit Cloud are limited by RAM, CPU \
               and Disk storage. To prevent crashing of the app, please select the items one at a time!\
               \nReference from: https://blog.streamlit.io/common-app-problems-resource-limits/')
    st.subheader('Truck')
    truck_df = truck_df.sort_values(by='TRUCK_ID').set_index('TRUCK_ID')

    # Let's put a pick list here so they can pick the fruit they want to include 
    #st.caption("As a food truck owner, you should know your truck ID. Pick your Truck ID below which ranges from 1 to 450")
    trucks_selected = st.selectbox("As a food truck owner, you should know your truck ID. Pick your Truck ID below which ranges from 1 to 450", list(truck_df.index))
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
    #st.caption("Select the menu that you would like to predict. By default, the selection will include your \
     #          highest food in demand for the past month!")
    # Let's put a pick list here so they can pick the fruit they want to include 
    topfood = history_df.groupby('MENU_ITEM_ID')['DEMAND'].max().idxmax()
    #st.text(topfood)
    topmenu = final_scaled[final_scaled['MENU_ITEM_ID'] == topfood]
    #st.dataframe(topmenu)
    menu_selected = st.multiselect("Select the menu that you would like to predict. By default, the selection will include your \
               highest food in demand for the past month!", list(menu_df.index), topmenu['MENU_ITEM_NAME'])
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
    #st.caption('Use the slider to predict for the next 1 to 30 days')
    numdays = st.slider('Use the slider to predict for the next 1 to 30 days', 1, 30)
    tdays = numdays
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
            final.loc[index, 'ORDER_YEAR'] = preddate.year
            final.loc[index, 'ORDER_MONTH'] = preddate.month
            final.loc[index, 'ORDER_DAY'] = preddate.day
            index += 1
        numdays -= 1
    final['UNIT_PRICE'] = final['SALE_PRICE_USD']
    final_df = final[['MENU_ITEM_ID', 'TRUCK_ID','ORDER_YEAR', 'ORDER_MONTH', 'ORDER_DAY','UNIT_PRICE', 'EV_FLAG', 'DAYS_OPENED',
                      'MENU_TYPE_ID', 'TEMPERATURE_OPTION', 'COST_OF_GOODS_USD', 'ITEM_CATEGORY_Main', 'ITEM_CATEGORY_Beverage'
                      ,'ITEM_CATEGORY_Dessert','ITEM_CATEGORY_Snack']]
    #st.dataframe(final)
    #st.dataframe(final_df)
    if st.button("Predict demand"):
        taba,  tabd, tabb, tabc = st.tabs(["Predicted demand per food", "Actionable Insights","Past vs Future (Demand)", "Past vs Future (Sales)"])
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
        with taba:
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
        with tabd:
            recipe_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.raw_supply_chain.recipe").to_pandas()
            item_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.raw_supply_chain.item").to_pandas()
            i_df = pd.DataFrame(columns=["MENU_ITEM_ID", "MENU_ITEM_NAME", "DEMAND"])
            for i in range(len(menu)):
                i_df = i_df.append({"MENU_ITEM_ID": menu['MENU_ITEM_ID'][i],"MENU_ITEM_NAME": menu['MENU_ITEM_NAME'][i] ,"DEMAND": predlist[i]}, ignore_index = True)
            #st.dataframe(i_df)
            merged = pd.merge(recipe_df, i_df,on = 'MENU_ITEM_ID',how='outer', indicator=True)
            i_df = merged[merged['_merge'] == 'both'].drop('_merge', axis = 1)
            #st.dataframe(i_df)
            merged2 = pd.merge(i_df, item_df, on = 'ITEM_ID', how = 'outer', indicator = True)
            i_df = merged2[merged2['_merge'] == 'both'].drop('_merge', axis = 1)
            i_df['DEMAND_ITEM'] = i_df['UNIT_QUANTITY'] * i_df['DEMAND']
            i_group = i_df.groupby('ITEM_ID')['DEMAND_ITEM'].sum().reset_index()
            #st.dataframe(i_group)
            #st.dataframe(i_df)
            for i in range(len(i_group)):
                for x in range(len(i_df)):
                    if i_group['ITEM_ID'][i] == i_df['ITEM_ID'][x]:
                        st.subheader(i_df['NAME'][x])
                        st.text('Number of item required: {0:.2f}'.format(i_group['DEMAND_ITEM'][i]))
                        st.text('In units: {0}'.format(i_df['UNIT'][x]))
                        st.text('Cost: ${0:.2f}'.format(i_df["UNIT_PRICE"][i] * i_group['DEMAND_ITEM'][i]))
                        break
                    
            #for i in range(len(menu)):
            #    st.subheader(menu['MENU_ITEM_NAME'][i])
            #    for x in range(len(i_df)):
            #        if i_df['MENU_ITEM_ID'][x] == menu['MENU_ITEM_ID'][i]:
            #            st.text("{0:.2f}: {1}".format(i_df['NAME'][x], i_df['DEMAND_ITEM'][x]))

        with tabb:
            # Streamlit app
            #st.title('Past and Future Comparison')
            #st.caption('Past data is calculated by taking the past 1 month of historical data and \
            #        averaging to the number of days stated by the slider. Future data is calculated by the sum of predicted values.')
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
            st.text("It is predicted to have {0:.2f}% demand increase in the next {1} days".format(
                (sum(predlist) - past_demand) / past_demand * 100, tdays))
        with tabc:
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
            st.text("It is predicted to have {0:.2f}% sales increase in the next {1} days".format(
                (present_sales - past_sales) / past_sales * 100, tdays))
