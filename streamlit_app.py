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

# Tabs set-up
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Predicting Future Sales [Shi Wei]', 'Predicting Customer Spending [Ernest]', 'Predicting Customer Churn [Gwyneth]', 'Guo Fung', 'Demand Forecasting [Kok Kai]'])

with tab1:
    st.title('Predicting Future Sales :money_with_wings:')
    description = """
    Welcome to the 'Predict Future Sales' tab! 
    This dashboard is designed to help Tasty Byte's team analyze and predict future sales, aligning with the company's ambitious goal of achieving 25% YoY growth over the next 5 years.

    With this interactive tool, you can explore valuable insights that will contribute to your strategic decision-making process. 
    Gain a deeper understanding of sales trends, identify growth opportunities, and make data-driven decisions to propel Tasty Byte towards its long-term vision."""
    st.markdown(description)
    
    ## Define function to load the customer's cluster results
    def SW_load_cust_cluster():
        data_cust_cluster = pd.read_csv("./SW_cluster_results.csv") 
        data_cust_cluster = pd.DataFrame(data_cust_cluster)
        return data_cust_cluster
    
    ## Define function to load the uplift prediction model
    def SW_load_uplift_1M():
        data_1m = pd.read_csv("./SW_UpliftPrediction[1M].csv") 
        data_1m = pd.DataFrame(data_1m)
        # Load customer cluster data
        data_cust_cluster = SW_load_cust_cluster()
        data_1m = pd.merge(data_1m, data_cust_cluster, on='CUSTOMER_ID')
        # Return merged data
        return data_1m
    
    def SW_load_uplift_2M():
        data_2m = pd.read_csv("./SW_UpliftPrediction[2M].csv") 
        data_2m = pd.DataFrame(data_2m)
        # Load customer cluster data
        data_cust_cluster = SW_load_cust_cluster()
        data_2m = pd.merge(data_2m, data_cust_cluster, on='CUSTOMER_ID')
        # Return merged data
        return data_2m
    
    def SW_load_uplift_3M():
        data_3m = pd.read_csv("./SW_UpliftPrediction[3M].csv") 
        data_3m = pd.DataFrame(data_3m)
        # Load customer cluster data
        data_cust_cluster = SW_load_cust_cluster()
        data_3m = pd.merge(data_3m, data_cust_cluster, on='CUSTOMER_ID')
        return data_3m
    
    def SW_get_predicted_sales(data):
        predicted_sales = data['PREDICT_SALES'].sum().round(2)
        return predicted_sales
    
    ## Create a DataFrame with city, longitude, and latitude information
    city_coordinates = pd.DataFrame({
        'CITY_FREQUENCY': [10613, 10016, 9261, 9122, 7288],
        'CITY': ['San Mateo', 'New York City', 'Boston', 'Denver', 'Seattle'],
        'LATITUDE': [37.5625, 40.7128, 42.3601, 39.7392, 47.6062],
        'LONGITUDE': [-122.3229, -74.0060, -71.0589, -104.9903, -122.3321]
    })
    
    ## Visualisation 1: Display bar chart 
    # Create dataframe that stores sales values
    nov = SW_get_predicted_sales(SW_load_uplift_1M())
    dec = SW_get_predicted_sales(SW_load_uplift_2M())
    jan = SW_get_predicted_sales(SW_load_uplift_3M())
    data = {
    'Month': ['October 2022', 'November 2022', 'December 2022', 'January 2023'],
    'sales': [4837269.25, nov, dec, jan]}
    sales_df = pd.DataFrame(data)
    
    # Create bar chart
    st.subheader('Sales Trend and Prediction :chart_with_upwards_trend:')
    fig_1 = go.Figure()

    fig_1.add_trace(
        go.Bar(
            x=sales_df['Month'],
            y=sales_df['sales'],
            marker_color='blue',
            text=sales_df['sales'],
            textposition='auto',
        )
    )

    # Customize the layout
    fig_1.update_layout(
        xaxis_title='Month',
        yaxis_title='Sales Amount ($)',
        showlegend=False
    )
    
    # Display the bar chart in the Streamlit tab
    st.plotly_chart(fig_1)
    
    ## Present insights based on the bar chart
    st.subheader('Insights :eyeglasses:')
    st.write("Based on the sales trend and predictions, here are some key observations:")

    # Create a list of insights
    insights = [
        "There is a positive growth trend in sales over the three upcoming months, which aligns with Tasty Byte's growth goal.",
        "The predicted sales for January 2023 shows significant growth and is a positive indicator of progress towards the 25% YoY growth goal.",
    ]

    # Display the insights in a bullet point format
    st.write("✓ " + insights[0])
    st.write("✓ " + insights[1])
    
    ## Visualisation 2: Display pie chart 
    # Create DataFrame
    cluster_results_df = pd.DataFrame(SW_load_cust_cluster())
    
    # Map numeric cluster values to labels
    cluster_results_df['CLUSTER'] = cluster_results_df['CLUSTER'].replace({0: 'Middle Value', 
                                                                           1: 'Low Value', 
                                                                           2: 'High Value'})

    # Count the number of customers in each cluster
    cluster_counts = cluster_results_df['CLUSTER'].value_counts()
    
    # Create a pie chart
    st.subheader('Customer Cluster Distribution :bar_chart:')
    fig_2 = go.Figure()

    fig_2.add_trace(
        go.Pie(
            labels=cluster_counts.index,
            values=cluster_counts.values,
            textinfo='percent+label',
            marker=dict(colors=['gold', 'mediumturquoise', 'darkorange']),
        )
    )

    # Customize the layout
    fig_2.update_layout(
        showlegend=False,
    )

    # Display the pie chart in the Streamlit tab
    st.plotly_chart(fig_2)
    
    ## Predict Future Sales Based On Customer Cluster
    st.subheader('Predict Future Sales Based On Customer Cluster 	:telescope:')

    ## Define user input functions
    # User Input 1: Select Customer Cluster
    def SW_get_cust_cluster():
        # Display the dropdown box
        cluster_selection = ['1 - Low Value (Customers who buy less frequently and generate lower sales)', 
                             '2 - Middle Value (Customers who make average purchases)', 
                             '3 - High Value (Customers who make frequent purchases and generate higher sales.)']
        selected_cluster = st.selectbox(
            ":people_holding_hands: Select Customer Cluster:", cluster_selection)
        if selected_cluster == '1 - Low Value (Customers who buy less frequently and generate lower sales)':
            return 1
        elif selected_cluster == '2 - Middle Value (Customers who make average purchases)':
            return 0
        else:
            return 2
    
    # User Input 2: Select Timeframe
    def SW_get_timeframe():
        # Display the dropdown box
        timeframe_selection = ['1 month', '2 months', '3 months']
        selected_months = st.selectbox(
            ":calendar: Select the range of months for prediction:", timeframe_selection)
        return selected_months
    
    # User Input 3: Select Metric
    def SW_get_selected_metrics():
        # Display checkboxes for key metrics
        st.write(":ballot_box_with_check: Select the metrics to view:")
        show_total_sales = st.checkbox("Total Predicted sales", value=True)
        show_avg_spending = st.checkbox("Average Predicted Spending per Customer", value=True)
        selected_metrics = []
        if show_total_sales:
            selected_metrics.append("0")
        if show_avg_spending:
            selected_metrics.append("1")
        return selected_metrics
    
    ## Define function to get results to display
    def SW_process_data(data, cluster_input):
    # Filter the data based on cluster_input
        filtered_data_cluster = data[data['CLUSTER'] == cluster_input]
        # Replace all values in 'CLUSTER' so we can find the number of customers
        filtered_data_cluster['CLUSTER'] = 1
        filtered_data_city = filtered_data_cluster.groupby('CITY_FREQUENCY').aggregate(sum)

        # Keep only columns that are useful
        filtered_data_city = filtered_data_city[['PREDICT_SALES', 'CLUSTER', 'MONETARY_M3_HO']]

        # Merge city_coordinates with filtered_data_city
        final_df = pd.merge(filtered_data_city, city_coordinates, on='CITY_FREQUENCY')

        # Round the values to two decimal places
        final_df['PREDICT_SALES_Rounded'] = final_df['PREDICT_SALES'].round(2)
        final_df['AVG_SPENDING_Rounded'] = (final_df['PREDICT_SALES'] / final_df['CLUSTER']).round(2)
        final_df['UPLIFT_PERCENTAGE'] = ((final_df['PREDICT_SALES'] - final_df['MONETARY_M3_HO']) / final_df['MONETARY_M3_HO']) * 100

        return final_df
    

    ## Define function to display results in a map based on selected metric
    def SW_display_map(final_df, metrics_input):
        # Create the map figure
        fig = go.Figure()

        # Adjust bubble size based on sales (scaled down for aesthetics)
        fig.add_trace(
            go.Scattermapbox(
                lon=final_df['LONGITUDE'],
                lat=final_df['LATITUDE'],
                text=final_df['CITY'],
                customdata=final_df[['PREDICT_SALES_Rounded', 'AVG_SPENDING_Rounded', 'UPLIFT_PERCENTAGE']],
                hoverinfo='text',  # Display the text when hovering over the markers
                mode='markers',
                marker=dict(
                    size=final_df['PREDICT_SALES_Rounded'] / 500000,  # Adjust marker size based on sales (scaled down for aesthetics)
                    sizemode='area',  # Use area to scale the marker size
                    sizeref=0.003,  # Adjust the scaling factor for marker size
                    sizemin=5,  # Set the minimum size for the markers
                    color=final_df['UPLIFT_PERCENTAGE'],  # Use Uplift Percentage as the marker color
                    colorscale='Viridis',  # Choose a suitable colorscale
                    colorbar=dict(
                        title='Percentage Uplift',
                        thickness=15,
                        len=0.5,
                    ),
                    opacity=0.8
                ),
                showlegend=False  # Set showlegend to False to hide the legend
            )
        )

        # Add custom labels to the map
        labels = []
        for index, row in final_df.iterrows():
            if '0' in metrics_input and '1' in metrics_input:
                label = f"Total Predicted Sales: ${row['PREDICT_SALES_Rounded']:.2f}<br>" + \
                        f"Average Predicted Spending per Customer: ${row['AVG_SPENDING_Rounded']:.2f}<br>" + \
                        f"Percentage Uplift: {row['UPLIFT_PERCENTAGE']:.2f}%"
            elif '0' in metrics_input:
                label = f"Total Predicted Sales: ${row['PREDICT_SALES_Rounded']:.2f}<br>" + \
                        f"Percentage Uplift: {row['UPLIFT_PERCENTAGE']:.2f}%"
            elif '1' in metrics_input:
                label = f"Average Predicted Spending per Customer: ${row['AVG_SPENDING_Rounded']:.2f}<br>" + \
                        f"Percentage Uplift: {row['UPLIFT_PERCENTAGE']:.2f}%"
            else:
                label = f"Percentage Uplift: {row['UPLIFT_PERCENTAGE']:.2f}%"
            labels.append(label)

        fig.add_trace(
            go.Scattermapbox(
                lon=final_df['LONGITUDE'],
                lat=final_df['LATITUDE'],
                mode='text',
                text=labels,
                textfont=dict(size=10, color='white'),
                showlegend=False  # Set showlegend to False to hide the legend
            )
        )

        # Customize the layout
        fig.update_layout(
            title={
                'text': 'Predicted Sales by City<br><span style="font-size:10px; font-weight:normal">*The size of each bubble represents the predicted sales amount.</span>',
                'y': 0.95,  # Adjust the vertical position of the title
                'x': 0.5,  # Center the title horizontally
                'xanchor': 'center',
                'yanchor': 'top',
            },
            mapbox=dict(
                style='carto-positron',
                zoom=2.5,  # Set the initial zoom level
                center=dict(lat=44, lon=-95.7129),  # Set the initial center of the map
            ),
            showlegend=True,  # Show the legend
            legend=dict(
                title='Predicted Sales ($)',  # Legend title
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
            )
        )

        # Display the map in the Streamlit tab
        st.plotly_chart(fig)

    ## Define Function to display results in table
    def SW_display_table(data, metrics_input):
        if metrics_input == ['0']:
            final_df = data[['CITY', 'PREDICT_SALES_Rounded', 'UPLIFT_PERCENTAGE']]
            final_df.rename(columns={'CITY':'City',
                                'PREDICT_SALES_Rounded': 'Predicted Sales ($)',
                                'UPLIFT_PERCENTAGE': 'Uplift Percentage (%)'}, inplace=True)
            
        elif metrics_input == ['1']:
            final_df = data[['CITY', 'AVG_SPENDING_Rounded', 'UPLIFT_PERCENTAGE']]
            final_df.rename(columns={'CITY':'City',
                                'AVG_SPENDING_Rounded': 'Average Sales Per Customer ($)',
                                'UPLIFT_PERCENTAGE': 'Uplift Percentage (%)'}, inplace=True)
            
        else:
            final_df = data[['CITY', 'PREDICT_SALES_Rounded', 'AVG_SPENDING_Rounded', 'UPLIFT_PERCENTAGE']]
            final_df.rename(columns={'CITY':'City',
                                'PREDICT_SALES_Rounded': 'Predicted Sales ($)',
                                'AVG_SPENDING_Rounded': 'Average Sales Per Customer ($)',
                                'UPLIFT_PERCENTAGE': 'Uplift Percentage (%)'}, inplace=True)
            
        st.subheader("Predicted Sales and Average Sales Per Customer with Uplift Percentage for Each City")
        st.write(final_df)
        return
    
    ## Get user inputs
    cluster_input = SW_get_cust_cluster()
    timeframe_input = SW_get_timeframe()
    metrics_input = SW_get_selected_metrics()

    # Display the "Predict" button
    if st.button("Predict"): 
        if (timeframe_input == '1 month'):
            ## Load data based on selected timeframe
            data_1M = SW_load_uplift_1M()
            
            # Filter and process the data based on user input
            final_df = SW_process_data(data_1M, cluster_input)
            
            # Display the map in the Streamlit tab
            SW_display_map(final_df, metrics_input)
            
            # Display table result
            SW_display_table(final_df, metrics_input)
            
            # Display actionable insights (These are customized based on user's input)
            st.subheader("Actionable Insights Based on Selection")
            if cluster_input == 1: #Low Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (low value) over a one-month timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("   **San Mateo** has the highest predicted sales amount of **$1,234,487.37** and a relatively high uplift percentage of **62.86%**. Consider running **targeted marketing campaigns** in this city to capitalize on the high predicted sales and further increase customer engagement. Offer **personalized promotions or incentives** to attract more customers and drive sales even higher.")

                st.write("2. Sales Team Focus:")
                st.write("   **Denver** has a relatively lower uplift percentage of **38.84%** compared to other cities. Consider **directing the sales team's efforts towards this city** to explore opportunities for growth and expansion. The sales team can focus on building relationships with potential customers, understanding their needs, and offering tailored solutions to drive sales in Denver.")

                st.write("3. Pricing Strategy:")
                st.write("   **Boston** exhibits a significant uplift percentage of **111.19%**. Consider implementing **dynamic pricing strategies or limited-time offers** in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")
                
            elif cluster_input == 2: #High Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (high value) over a one-month timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("   Consider running targeted marketing campaigns in **San Mateo**, which shows a significantly higher uplift percentage of **48.64%**. Offer **personalized promotions or incentives** to attract more customers and increase sales in this area.")

                st.write("2. Customer Retention Efforts:")
                st.write("   Focus on customer retention efforts in **Seattle** and **New York City**, where negative uplift percentages (**-2.30% and -2.08%**) indicate a slight decline in predicted sales. **Identify and address reasons for the decline** to retain existing customers and prevent further loss in sales.")

                st.write("3. Pricing Strategy:")
                st.write("   Capitalize on the positive uplift percentage (**15.44%**) observed in **Boston** by implementing **dynamic pricing strategies or limited-time offers**. This can help further boost sales in the city.")

            else: #Middle Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (middle value) over a one-month timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("   Consider running targeted marketing campaigns in **Boston**, which shows a significantly higher uplift percentage of **48.87%**. The company can offer **personalized promotions or incentives** to attract more customers and increase sales in this area.")
                
                st.write("2. Sales Team Focus")
                st.write("   **Denver** has a relatively lower uplift percentage of **10.02%** compared to other cities. Consider **directing the sales team's efforts** towards this city to explore opportunities for growth and expansion. The sales team can focus on building relationships with potential customers, understanding their needs, and offering tailored solutions to drive sales in Denver.")

                st.write("3. Pricing Strategy:")
                st.write("   **Boston** exhibits a significant uplift percentage of **48.87%**. Consider implementing **dynamic pricing strategies or limited-time offers** in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")

        elif (timeframe_input == '2 months'):
            ## Load data based on selected timeframe
            data_2M = SW_load_uplift_2M()
            
            # Filter and process the data based on user input
            final_df = SW_process_data(data_2M, cluster_input)
            
            # Display the map in the Streamlit tab
            SW_display_map(final_df, metrics_input)
            
            # Display table result
            SW_display_table(final_df, metrics_input)
            
            # Display actionable insights (These are customized based on user's input)
            st.subheader("Actionable Insights Based on Selection")
            if cluster_input == 1: #Low Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (low value) over a two-months timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("  **San Mateo** has the highest predicted sales amount of **$1,236,987.90** and a relatively high uplift percentage of **63.19%**. Consider running **targeted marketing campaigns** in this city to capitalize on the high predicted sales and further increase customer engagement. Offer personalized promotions or incentives to attract more customers and drive sales even higher.")

                st.write("2. Sales Team Focus:")
                st.write("   **Denver** has a relatively lower uplift percentage of **39.43%** compared to other cities. Consider **directing the sales team's efforts** towards this city to explore opportunities for growth and expansion. The sales team can focus on building relationships with potential customers, understanding their needs, and offering tailored solutions to drive sales in Denver.")

                st.write("3. Pricing Strategy:")
                st.write("   **Boston** exhibits a significant uplift percentage of **111.69%**. Consider implementing **dynamic pricing strategies or limited-time offers** in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")
                
            elif cluster_input == 2: #High Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (high value) over a two-months timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("   **San Mateo** stands out with the predicted sales amount of **$7,526.23** and the only positive uplift percentage of **2.00%**. Consider running **targeted marketing campaigns** in San Mateo to capitalize on the high predicted sales and further increase customer engagement. Offer personalized promotions or incentives to attract more customers and drive sales even higher.")

                st.write("2. Customer Retention Efforts:")
                st.write("   Focus on customer retention efforts in **Denver**, where negative uplift percentage (**-18.09%**) indicate a slight decline in predicted sales. **Identify and address reasons for the decline** to retain existing customers and prevent further loss in sales.")

                st.write("3. Pricing Strategy:")
                st.write("   Since **San Mateo** exhibits a positive uplift percentage of **2.00%**. Consider implementing **dynamic pricing strategies or limited-time offers** in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")

            else: #Middle Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (middle value) over a two-months timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("   Consider running targeted marketing campaigns in **Boston**, which shows a significantly higher uplift percentage of **49.61%**. The company can offer **personalized promotions or incentives** to attract more customers and increase sales in this area.")
                
                st.write("2. Sales Team Focus")
                st.write("   **Denver** has a relatively lower uplift percentage of **10.56%** compared to other cities. Consider **directing the sales team's efforts** towards this city to explore opportunities for growth and expansion. The sales team can focus on building relationships with potential customers, understanding their needs, and offering tailored solutions to drive sales in Denver.")

                st.write("3. Pricing Strategy:")
                st.write("   **Boston** exhibits a significant uplift percentage of **49.61%**. Consider implementing **dynamic pricing strategies or limited-time offers** in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")
            
        else:
            ## Load data based on selected timeframe
            data_3M = SW_load_uplift_3M()
            
            # Filter and process the data based on user input
            final_df = SW_process_data(data_3M, cluster_input)
            
            # Display the map in the Streamlit tab
            SW_display_map(final_df, metrics_input)
            
            # Display table result
            SW_display_table(final_df, metrics_input)
            
            # Display actionable insights (These are customized based on user's input)
            st.subheader("Actionable Insights Based on Selection")
            if cluster_input == 1: #Low Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (low value) over a three-months timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("  **San Mateo** stands out with the highest predicted sales amount of **$2,984,755.02** and a relatively high uplift percentage of **37.65%**. The company can run **targeted marketing campaigns** in this city to capitalize on the high predicted sales and further increase customer engagement. Offering personalized promotions or incentives can attract more customers and drive sales even higher in San Mateo.")

                st.write("2. Sales Team Focus:")
                st.write("   **Denver** has a relatively lower uplift percentage of **25.58%** compared to other cities. The company should consider **directing the sales team's efforts** towards Denver to explore opportunities for growth and expansion. The sales team can focus on building relationships with potential customers, understanding their needs, and offering tailored solutions to drive sales in Denver.")

                st.write("3. Pricing Strategy:")
                st.write("   **Boston** exhibits a significant uplift percentage of 63.94%. The company should consider implementing **dynamic pricing strategies or limited-time offers** in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")
                
            elif cluster_input == 2: #High Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (high value) over a three-months timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("   **San Mateo** stands out with the predicted sales amount of **$3,116.67** and an impressive uplift percentage of **49.12%**. Consider running **targeted marketing campaigns** in San Mateo to capitalize on the high predicted sales and further increase customer engagement. Offer personalized promotions or incentives to attract more customers and drive sales even higher.")

                st.write("2. Customer Retention Efforts:")
                st.write("   Focus on customer retention efforts in **Denver**, where negative uplift percentage (**-10.18%**) indicate a slight decline in predicted sales. **Identify and address reasons for the decline** to retain existing customers and prevent further loss in sales.")

                st.write("3. Pricing Strategy:")
                st.write("   **Boston** exhibits a significant uplift percentage of **16.14%**. Consider implementing **dynamic pricing strategies or limited-time offers** in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")

            else: #Middle Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (middle value) over a three-months timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("   **Boston** and **New York City** have relatively high predicted sales of **$1,526,225.99** and **$1,202,009.54**, respectively. Consider running **targeted marketing campaigns** in these cities to capitalize on the high predicted sales and further increase customer engagement. Offer personalized promotions or incentives to attract more customers and drive sales even higher.")
                
                st.write("2. Sales Team Focus")
                st.write("   **Denver** has a negative uplift percentage of **0.45%**. Consider **directing the sales team's efforts** towards this city to explore opportunities for growth and expansion. The sales team can focus on building relationships with potential customers, understanding their needs, and offering tailored solutions to drive sales in Denver.")

                st.write("3. Pricing Strategy:")
                st.write("   **New York City** exhibits a significant uplift percentage of **20.89%**. Consider implementing **dynamic pricing strategies or limited-time offers** in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")
        
with tab2:
    st.title('Predicting Customer Spending :moneybag:')

    st.markdown("This tab allows the Tasty Bytes's team to analyse and predict customer spending, to aid them to achieve their goal of 25% YoY growth in sales across 5 years.")
    st.markdown("On this tab, it allows users to make prediction on customer spending in the next month based on their city, recency cluster(R), frequency cluster(F) and monetary cluster(M).")
    st.markdown("It also allows users to manually input values to get the customer spending in the next month and find out what customer group they belong to.")
    st.markdown("With these interactive tools, you will be able to explore and gain valuable insights to push Tasty Bytes to reach the goal!")

    # Get customer details
    customer_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.raw_customer.customer_loyalty")
    us_customer_df = customer_df.filter(F.col("COUNTRY") == "United States")

    # Get order details
    order_header_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.raw_pos.order_header")
    order_header_df = order_header_df.na.drop(subset="CUSTOMER_ID")
    customer_US = us_customer_df.select(F.col("CUSTOMER_ID"))
    order_header_df = order_header_df.join(customer_US, order_header_df.CUSTOMER_ID == customer_US.CUSTOMER_ID, lsuffix = "", rsuffix = "_US")

    # Convert to pandas
    us_customer_df = us_customer_df.to_pandas()
    order_header_df = order_header_df.to_pandas()

    # Define function to load the uplift prediction model
    def load_Uplift_Churn_1M():
        data = pd.read_csv("./uplift/UpliftPrediction[1M].csv") 
        return data
    
    # Load the model
    upliftdata = load_Uplift_Churn_1M()
    upliftdata_copy = load_Uplift_Churn_1M()
    
    # Define function to load the cluster sales
    def load_cluster_sales_1M():
        data = pd.read_csv("./uplift/clusterSales[1M].csv") 
        return data
    def load_city_enc():
        data = pd.read_csv("./uplift/city_enc.csv") 
        city_dict = data.set_index('CITY').T.to_dict('dict')
        return city_dict

    # User inpput
    def get_city():
        city_selection = st.multiselect("Select City* :cityscape:", options = load_city_enc())
        city_list = list(city_selection)
        return city_list
    def get_city_int(city_input):
        # Load city mapping 
        city_dict = load_city_enc()
        city_int = [] # Store selected city frequency 
        for each in city_input:
            city_int.append(city_dict[each]['CITY_FREQUENCY'])
        return city_int
    
    # Bar chart 
    order_year = order_header_df[['ORDER_ID', 'ORDER_TS', 'ORDER_AMOUNT']]

    # Extract the year from the 'ORDER_TS' column
    order_year['year'] = order_year['ORDER_TS'].dt.year

    # Group the data by 'year' and calculate the sum of 'ORDER_AMOUNT'
    sum_by_year = order_year.groupby('year')['ORDER_AMOUNT'].sum()

    # Convert the result into a DataFrame
    bar_df = sum_by_year.reset_index()
    # Set 'year' as the index
    bar_df.set_index('year', inplace=True)

    # Display bar chart
    st.subheader("Bar Chart of Sales across Year")
    st.bar_chart(bar_df)
    st.subheader("Insights :chart_with_upwards_trend:")
    st.markdown("From the bar chart, we are able to observe:")
    st.markdown("● Yearly sales for Tasty Bytes has been increasing from 2019 to 2022,  indicating that Tasty Bytes is growing steadily over the years.")
    st.markdown("● Across the years, the sales has been increasing by more than 25% compared to the previous year, with the most increase being from 2019 to 2022 at a increase of 618%.")
    st.markdown("● However, the sales increase from 2021 to 2022 has not been that great at only 47%.")
    st.markdown("Will it be able to achieve the goal of increasing the sales by 25% YoY across 5 years?")
    st.markdown("Use the predictors below to find out! :money_mouth_face:") 

    # Sub tab for user to navigate 
    tab1, tab2 = st.tabs(["Based on RFM", "Based on Specific Value"])

    # Based on RFM
    with tab1:
        st.subheader('Based on RFM')

        # Select City
        city_input = get_city() 

        # RFM input
        col1, col2, col3 = st.columns(3)

        # Setting the input
        cluster_mapping_r = {0: 'Old Customers', 1: 'Recent Customers'}
        upliftdata['CUST_REC_CLUSTER'] = upliftdata['CUST_REC_CLUSTER'].map(cluster_mapping_r)

        cluster_mapping_f = {0: 'Moderately Frequent Customers', 1: 'Infrequent Customers', 2: 'Frequent Customers'}
        upliftdata['CUST_FREQ_CLUSTER'] = upliftdata['CUST_FREQ_CLUSTER'].map(cluster_mapping_f)

        cluster_mapping_m = {0: 'Low Spending Customers', 1: 'High Spending Customers'}
        upliftdata['CUST_MONETARY_CLUSTER'] = upliftdata['CUST_MONETARY_CLUSTER'].map(cluster_mapping_m)

        # Recency
        col1.subheader("Recency Cluster")
        rCluster =  col1.selectbox(label="Recency", options = upliftdata['CUST_REC_CLUSTER'].unique())

        # Frequency
        col2.subheader("Frequency Cluster")
        fCluster =  col2.selectbox(label="Frequency", options = upliftdata['CUST_FREQ_CLUSTER'].unique())

        # Monetary
        col3.subheader("Monetary Cluster")
        mCluster =  col3.selectbox(label="Monetary", options = upliftdata['CUST_MONETARY_CLUSTER'].unique())

        # Filter csv data
        def filterdata(data):
            # Predicted total sales
            predictedsales = data['PREDICT_SALES'].sum()

            # Actual sales
            actualsales = data['MONETARY_M3_HO'].sum()

            # Uplift
            uplift = predictedsales - actualsales

            # Percent uplift
            percentuplift = ((predictedsales - actualsales) / actualsales) * 100

            st.write("In the next month, the selected group of customer will generate **${:0,.2f}**.".format(predictedsales))
            if (predictedsales > actualsales):
                st.write("This is an increase by ${:0,.2f}".format(uplift) + " which is an increase of {:.2f}%. :smile:".format(percentuplift))

            else:
                st.write("Which is an decrease of ${:0,.2f}".format(uplift) + " which is a decrease of {:.2f}%. :pensive:".format(percentuplift))

        # Convert dataframe to csv 
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        # Predict uplift button
        if st.button('Predict Uplift'):
            # City input
            city_int = get_city_int(city_input)

            if (len(city_int) == 0):
                st.write("**Please select at least one city.**")
            else:
                # Filtering data
                filtered_data = upliftdata[(upliftdata['CITY_FREQUENCY'].isin(city_int)) & (upliftdata['CUST_REC_CLUSTER'] == rCluster) & (upliftdata['CUST_FREQ_CLUSTER'] == fCluster) & (upliftdata['CUST_MONETARY_CLUSTER'] == mCluster)]

                filterdata(filtered_data)

        # Display recommendations checkbox
        if st.checkbox('Display Recommendations'):
            st.subheader("Details of Customer that did not contribute to the uplift")

            # City input
            city_int = get_city_int(city_input)

            if (len(city_int) == 0):
                st.write("**Please select at least one city.**")
            else:
                # Filtering data
                filtered_data = upliftdata[(upliftdata['CITY_FREQUENCY'].isin(city_int)) & (upliftdata['CUST_REC_CLUSTER'] == rCluster) & (upliftdata['CUST_FREQ_CLUSTER'] == fCluster) & (upliftdata['CUST_MONETARY_CLUSTER'] == mCluster)]

                lower = filtered_data[filtered_data['MONETARY_M3_HO'] > filtered_data['PREDICT_SALES']]

                lower = lower.merge(us_customer_df, on = "CUSTOMER_ID", how = "left")

                lower = lower[["CUSTOMER_ID", "FIRST_NAME", "LAST_NAME", "E_MAIL", "PHONE_NUMBER"]]

                lower = lower.sort_values(by=['CUSTOMER_ID'])
                st.write(lower) 

                # Convert dataframe to csv
                csv = convert_df(lower)

                # download csv button
                st.download_button(
                label="Download details of customer that did not contribute to the uplift",
                data=csv,
                file_name='customer_details.csv',
                mime='text/csv',
            )
                # recommendations
                st.subheader("Recommendations")
                st.write("Here are some recommendations that could be done to help generate more sales:")
                st.write("**1. Personalised Communication:**")
                st.write("By downloading the csv file on the contact information of customers that did not contribute to the uplift, we are able to use it to send personalised messages or emails to these customers. By tailoring the communication based on their preferences, past interactions, and purchase history, the company can make the message more relevant and appealing. This would be able to engage customers more and boost the sales.")
                
                if (rCluster == "Old Customers" and mCluster == "Low Spending Customers"):
                    st.write("**2. Customer Reactivation Campaign:**")
                    st.write("As this group of customers belong to the Old Customer cluster, it means that they have not been buying from us for a period of time. Therefore, we can start a customer reactivation campaign to entice them to return to our food. We could send them a special discount code as an incentive to make a purchase. This way, it provides a reason for the customer to return and thus help increase our sales, achieving our goal.")
                    st.write("**3. Cross-selling:**")
                    st.write("For this group of customers, they tend to spend lesser. To increase their spending, we could do cross-selling. For every customer purchase, we could suggest adding another item that is complementary to it. One example is if a customer orders a burger, we can suggest adding a side of fries and a drink for a discounted combo price.")
                elif (rCluster == "Old Customers" and mCluster == "High Spending Customers"):
                    st.write("**2. Customer Reactivation Campaign:**")
                    st.write("As this group of customers belong to the Old Customer cluster, it means that they have not been buying from us for a period of time. Therefore, we can start a customer reactivation campaign to entice them to return to our food. We could send them a special discount code as an incentive to make a purchase. This way, it provides a reason for the customer to return and thus help increase our sales, achieving our goal.")
                else:
                    st.write("")
                
                if (rCluster == "Recent Customers" and fCluster != "Frequent Customers" and mCluster == "Low Spending Customers"):
                    st.write("**2. Loyalty Program:**")
                    st.write("As this group of customers belong to the Recent Customer cluster, it means that they have recently bought from us. To keep them engaged, we could launch a loyalty program. By launching a loyalty program, it can incentivise customers to remain engaged with Tasty Bytes. By offering points to them, where they can accumulate and claim a rewards, it will make customers want to purchase more oftenly, thus generating more sales and helping us reach our goal.")
                    st.write("**3. Cross-selling:**")
                    st.write("For this group of customers, they tend to spend lesser. To increase their spending, we could do cross-selling. For every customer purchase, we could suggest adding another item that is complementary to it. One example is if a customer orders a burger, we can suggest adding a side of fries and a drink for a discounted combo price.")
                elif (rCluster == "Recent Customers" and fCluster != "Frequent Customers" and mCluster == "High Spending Customers"):
                    st.write("**2. Loyalty Program:**")
                    st.write("As this group of customers belong to the Recent Customer cluster, it means that they have recently bought from us. To keep them engaged, we could launch a loyalty program. By launching a loyalty program, it can incentivise customers to remain engaged with Tasty Bytes. By offering points to them, where they can accumulate and claim a rewards, it will make customers want to purchase more oftenly, thus generating more sales and helping us reach our goal.")
                elif (rCluster == "Recent Customers" and fCluster == "Frequent Customers" and mCluster == "Low Spending Customers"):
                    st.write("**2. Cross-selling:**")
                    st.write("For this group of customers, they tend to spend lesser. To increase their spending, we could do cross-selling. For every customer purchase, we could suggest adding another item that is complementary to it. One example is if a customer orders a burger, we can suggest adding a side of fries and a drink for a discounted combo price.")
                else:
                    st.write("")

    # Based on Specific Value
    with tab2:
        st.subheader("Based on Specific Value")

        # Load the model
        with open('./uplift/Uplift_1M.pkl', 'rb') as file:
            uplift_1M = pickle.load(file)
        
        # Define function to load the cluster sales
        def load_cluster_sales_1M():
            data = pd.read_csv("./uplift/clusterSales[1M].csv") 
            return data
        
        # User input
        def get_city2():
            city_selection2 = st.selectbox("Select City* :cityscape:", options = load_city_enc())
            city_list2 = [city_selection2]
            return city_list2
        def get_city_int2(city_input2):
            # Load city mapping 
            city_dict = load_city_enc()
            city_int2 = [] # Store selected city frequency 
            for each in city_input2:
                city_int2.append(city_dict[each]['CITY_FREQUENCY'])
            return city_int2

        # Select City
        city_input2 = get_city2() 

        # Years with us
        timeframe_value =  list(range(0, 5))
        years_pickle = st.select_slider("Years as Member",options=timeframe_value)
        st.write('You selected:', years_pickle)

        # Recency
        recency_value = list(range(0, 151))
        recency_pickle = st.select_slider("Days since last purchase", options=recency_value)
        st.write('You selected:', recency_pickle)

        # Frequency
        frequency_value = list(range(5, 51))
        frequency_pickle = st.select_slider("Number of orders yearly", options=frequency_value)
        st.write('You selected:', frequency_pickle)

        # Monetary
        monetary_value = list(range(150, 1501))
        monetary_pickle = st.select_slider("Total amount spent yearly", options=monetary_value)
        st.write('You selected:', monetary_pickle)

        # Total Spent
        totalspent_pickle = monetary_pickle * years_pickle

        # Dataframe
        data = {
            'TOTAL_SPENT': [totalspent_pickle],
            'YEARS_WITH_US': [years_pickle],
            'MONETARY_VALUE': [monetary_pickle],
            'CUSTOMER_FREQUENCY': [frequency_pickle],
            'TOTAL_ORDER': [0],
            'RECENCY_DAYS': [recency_pickle],
            'CITY_FREQUENCY': get_city_int2(city_input2),
            'ORDER_TOTAL_S1': [0],
            'ORDER_TOTAL_S2': [0],
            'ORDER_TOTAL_S3': [0],
            'CHANGE_FROM_S1_TO_S2': [0],
            'CHANGE_FROM_S2_TO_S3': [0]
        }

        final = pd.DataFrame(data)

        if (recency_pickle <21):
            final['CUST_REC_CLUSTER'] = 1
        else:
            final['CUST_REC_CLUSTER'] = 0

        if (frequency_pickle <= 16):
            final['CUST_FREQ_CLUSTER'] = 1
        elif (frequency_pickle > 16 and frequency_pickle <21):
            final['CUST_FREQ_CLUSTER'] = 0
        else:
            final['CUST_FREQ_CLUSTER'] = 2
        
        if (monetary_pickle <= 680):
            final['CUST_MONETARY_CLUSTER'] = 0
        else:
            final['CUST_MONETARY_CLUSTER'] = 1

        final['OVERALL_SCORE'] = final['CUST_REC_CLUSTER'] + final['CUST_FREQ_CLUSTER'] + final['CUST_MONETARY_CLUSTER']

        if st.button('Predict Customer Spending'):
            if (len(get_city_int2(city_input2)) == 0):
                st.write("**Please select a city.**")
            else:
                # Get data based on customer cluster
                filter_value_r = int(final.iloc[0]['CUST_REC_CLUSTER'])
                filter_value_f = int(final.iloc[0]['CUST_FREQ_CLUSTER'])
                filter_value_m = int(final.iloc[0]['CUST_MONETARY_CLUSTER'])

                filtered_data_2 = upliftdata_copy[(upliftdata_copy['CUST_REC_CLUSTER'] == filter_value_r) & (upliftdata_copy['CUST_FREQ_CLUSTER'] == filter_value_f) &
                                                (upliftdata_copy['CUST_MONETARY_CLUSTER'] == filter_value_m) & (upliftdata_copy['CITY_FREQUENCY'].isin(get_city_int2(city_input2)))]
                

                filter_value_order = filtered_data_2["TOTAL_ORDER"].mean()
                filter_value_s1 = filtered_data_2["ORDER_TOTAL_S1"].mean()
                filter_value_s2 = filtered_data_2["ORDER_TOTAL_S2"].mean()
                filter_value_s3 = filtered_data_2["ORDER_TOTAL_S3"].mean()

                final["TOTAL_ORDER"] = filter_value_order
                final["ORDER_TOTAL_S1"] = filter_value_s1
                final["ORDER_TOTAL_S2"] = filter_value_s2
                final["ORDER_TOTAL_S3"] = filter_value_s3
                final["CHANGE_FROM_S1_TO_S2"] = (filter_value_s2 - filter_value_s1)
                final["CHANGE_FROM_S2_TO_S3"] = (filter_value_s3 - filter_value_s2)

                cluster_sales = load_cluster_sales_1M()
                pred = uplift_1M.predict_proba(final)
                pred = pd.DataFrame(pred)
                final["PREDICT_SALES_ST"] = (pred[0] * cluster_sales['Cluster0MeanSales'][0]) + (pred[1] * cluster_sales['Cluster1MeanSales'][0])
                amountSpent = final["PREDICT_SALES_ST"]
                avgSpent = upliftdata_copy['PREDICT_SALES'].mean()

                if (amountSpent[0] > avgSpent):
                    positiveDiff = amountSpent[0] - avgSpent
                    st.write("This specific customer would be spending **${:0,.2f}** in the next month,".format(amountSpent[0]))
                    st.write(" which is **${:0,.2f}** higher than the average customer spending. :thumbsup:".format(positiveDiff))
                else:
                    negativeDiff = avgSpent - amountSpent[0]
                    st.write("This specific customer would be spending **${:0,.2f}** in the next month,".format(amountSpent[0]))
                    st.write(" which is **${:0,.2f}** lower than the average customer spending. :thumbsdown:".format(negativeDiff))
                
                if (final["CUST_REC_CLUSTER"][0] == 1):
                    st.write("● This customer belongs to the Recent Customers cluster, which means that they are customers that recently visited Tasty Bytes.")
                else:
                    st.write("● This customer belongs to the Old Customers cluster, which means that they are customers that have not recently visited Tasty Bytes.")

                if (final["CUST_FREQ_CLUSTER"][0] == 1):
                    st.write("● This customer belongs to the Moderately Frequent Customers cluster, which means that they are customers that visited Tasty Bytes somewhat frequently.")
                elif (final["CUST_FREQ_CLUSTER"][0] == 0):
                    st.write("● This customer belongs to the Infrequent Customers cluster, which means that they are customers that visited Tasty Bytes infrequently.")
                else:
                    st.write("● This customer belongs to the Frequent Customers cluster, which means that they are customers that visited Tasty Bytes frequently.")

                if (final["CUST_MONETARY_CLUSTER"][0] == 1):
                    st.write("● This customer belongs to the High Spending Customers cluster, which means that they consistently spend a significant amount of money in a year.")
                else:
                    st.write("● This customer belongs to the Low Spending Customers cluster, which means that they consistently spend a lesser amount of money in a year.")
    
with tab3:
    st.title('Predicting Customer Churn :worried:')
    # introduction of web tab
    st.write("In this web tab, Tasty Bytes management will have the ability to obtain details about churn customers across various customer segments. They can **leverage the insights and advice provided to take proactive measures** aimed at retaining these customers in order for Tasty Bytes to **reach their goal** of increasing Net Promoter Score (NPS) from 3 to 40 by year end 2023. By **effectively addressing churn**, this will ensure that customers are engaged and shows strong loyalty and satisfaction towards Tasty Bytes, signifying that the NPS score is **poised to increase**.")
    st.write("Additionally, the management will also be able to **experiment** with customer's details to **predict** whether they will be likely to churn or not.")
    st.write("Customers are likely to churn when their predicted days to next purchase is **more than 14 days**.")
    st.header('Details of Churn Customers :face_with_monocle:')

    # loading of dataset 
    def load_next_purchase_cust_seg():
        data = pd.read_csv("./churn/NextPurchaseCustSeg2.csv")
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
    with open('./churn/NextPurchase2.pkl', 'rb') as file:
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
    
with tab4:
   
    # --- Title & Description for Tab--- 
    st.markdown("###" +' :arrow_up_small: Uplift Analysis for Churn/Non-Churn Customers')
    description = '''
    Using this tab, you can predict the uplift in revenue for both churn and non-churn customers in the United States (US), which **plays a crucial role in helping Tasty Byte achieve its goal of attaining 25% year-over-year growth over a period of 5 years**. 
    \nThe model employed for these predictions is an **AdaBoost Classifier**, which was **trained on historical data spanning from 1st January, 2019, to 1st November, 2022**.

    \nBelow, you will find a histogram  displaying the distribution for "Days to next Purchase." 
    '''
    st.markdown(description)
    
    #-----Functions for loading of files-----#  
    with open('./uplift/Uplift_1M.pkl', 'rb') as file:
        uplift_1M = pickle.load(file)
    with open('./uplift/Uplift_2W.pkl', 'rb') as file:
        uplift_2W = pickle.load(file)
    with open('./uplift/Uplift_3M.pkl', 'rb') as file:
        uplift_3M = pickle.load(file)
    # caching computations that return data
    @st.cache_data
    # Define function to load the uplift prediction model
    def load_Uplift_Churn_2W():
        data = pd.read_csv("./uplift/UpliftPrediction[2W].csv") 
        df = pd.DataFrame(data)
        return df
    @st.cache_data
    def load_Uplift_Churn_1M():
        data = pd.read_csv("./uplift/UpliftPrediction[1M].csv") 
        df = pd.DataFrame(data)
        return df
    @st.cache_data
    def load_Uplift_Churn_3M():
        data = pd.read_csv("./uplift/UpliftPrediction[3M].csv") 
        df = pd.DataFrame(data)
        return df
    
    @st.cache_data
    # Define function to load the cluster sales
    def load_cluster_sales_2W():
        data = pd.read_csv("./uplift/clusterSales[2W].csv")
        return data
    @st.cache_data
    def load_cluster_sales_1M():
        data = pd.read_csv("./uplift/clusterSales[1M].csv") 
        return data
    @st.cache_data
    def load_cluster_sales_3M():
        data = pd.read_csv("./uplift/clusterSales[3M].csv") 
        return data
    @st.cache_data
    def load_next_purchase():
        data = pd.read_csv("./uplift/NextPurchase2.csv")
        return data
    @st.cache_data
    def load_city_enc():
        data = pd.read_csv("./uplift/city_enc.csv") 
        city_dict = data.set_index('CITY').T.to_dict('dict')
        return city_dict
    #=================================================================================================#
    # Define the user input functions 
    def get_churn_days(_cNum):
        data = load_next_purchase() 
        churn_days = _cNum.slider("Determine days for churning", math.floor(data["TARGET"].min()) , 30,  14) # Default value of 14
        return churn_days
    def get_timeframe():
        timeframe_selection = ['2 Weeks', '1 Month', '3 Months']
        timeframe = st.selectbox('Select a timeframe :date:', timeframe_selection)
        return timeframe
    def get_city():
        city_selection = st.multiselect("Select City :cityscape:", options = load_city_enc())
        city_list = list(city_selection)
        return city_list
    def get_city_int(city_input):
        # Load city mapping 
        city_dict = load_city_enc()
        city_int = [] # Store selected city frequency 
        for each in city_input:
            city_int.append(city_dict[each]['CITY_FREQUENCY'])
        return city_int
    def get_years_with_us(cNum):
        # Slider
        timeframe_label =  [1,2,3,4]
        years_with_us_range = cNum.slider("Years as Member",timeframe_label[0],timeframe_label[-1], (timeframe_label[0],timeframe_label[-1]))
        return years_with_us_range
    #=================================================================================================#
    # Functions for analytics
    
    # Function to filter
    def filter_higher(next_purchase_data, uplift_predictions_time, churn_days_higher, years_with_us_range, city_int):
        custGroup = next_purchase_data[ (next_purchase_data["TARGET"] > churn_days_higher)] # Churn days higher 
        filtered_data_higher = uplift_predictions_time[ (uplift_predictions_time['YEARS_WITH_US'] >= years_with_us_range[0]) & (uplift_predictions_time['YEARS_WITH_US'] <= years_with_us_range[1]) 
                                            & (uplift_predictions_time['CITY_FREQUENCY'].isin(city_int)) ]
    
        filtered_data_higher = filtered_data_higher[filtered_data_higher["CUSTOMER_ID"].isin(custGroup["CUSTOMER_ID"])]
        filtered_data_higher = filtered_data_higher.drop(columns=['CUSTOMER_ID','MONETARY_M3_HO','LTVCluster','PREDICTED_PROBA_0','PREDICTED_PROBA_1','PREDICT_SALES','INDEX'])
        #st.write(filtered_data_higher)
        return filtered_data_higher 
    
    def filter_lower(next_purchase_data, uplift_predictions_time, churn_days_lower, years_with_us_range, city_int):
        custGroup = next_purchase_data[ (next_purchase_data["TARGET"] <= churn_days_lower)] # Churn days falling below
        filtered_data_lower = uplift_predictions_time[ (uplift_predictions_time['YEARS_WITH_US'] >= years_with_us_range[0]) & (uplift_predictions_time['YEARS_WITH_US'] <= years_with_us_range[1]) 
                                            & (uplift_predictions_time['CITY_FREQUENCY'].isin(city_int)) ]

        filtered_data_lower = filtered_data_lower[filtered_data_lower["CUSTOMER_ID"].isin(custGroup["CUSTOMER_ID"])]
        filtered_data_lower = filtered_data_lower.drop(columns=['CUSTOMER_ID','MONETARY_M3_HO','LTVCluster','PREDICTED_PROBA_0','PREDICTED_PROBA_1','PREDICT_SALES','INDEX'])
        #st.write(filtered_data_lower)
        return filtered_data_lower  
        
    # Function to find uplift
    def find_uplift(pred_df,cluster_sales, uplift_predictions_duration):
        
        pred_df["PREDICT_SALES_ST"] = (pred_df[0] * cluster_sales['Cluster0MeanSales'][0]) + (pred_df[1] * cluster_sales['Cluster1MeanSales'][0])
        
        # Merge pred_df and filtered data
        result_df = uplift_predictions_duration.merge(pred_df, left_index=True, right_index=True, how='right')
        #st.write(result_df)
        
        # Total sales
        totalSales = result_df['PREDICT_SALES_ST'].sum()
        # Total uplift
        totalUplift = result_df['PREDICT_SALES_ST'].sum() - result_df['MONETARY_M3_HO'].sum()
        # Calculation for change in sales
        percentUplift = ((result_df['PREDICT_SALES_ST'].sum()- result_df['MONETARY_M3_HO'].sum())/ result_df['MONETARY_M3_HO'].sum()) * 100
        
        return result_df, totalSales, totalUplift, percentUplift
    def recomendation_text(churn_uplift, nonChurn_Uplift):
        st.markdown("#### "+ "Practical Insights :mag:")
        if (churn_uplift > nonChurn_Uplift):
                recommendation = '''Churned customers have yielded a **greater uplift when compared to their non-churning counterparts**. Churned customers leads to a potential loss in sales, and the analytics displayed above confirm that they are likely to **generate a higher revenue uplift than customers who have not churned**. Moreover, retaining existing customers proves to be a more cost-effective strategy than acquiring new ones, as churned customers already **possess a certain level of familiarity with Tasty Bytes**.
                Churn customers represents potential lost in sales, and from the analytics shown above, we acknowledged that they are **likely to generate a higher uplift in sales as compared to customers who have not churn**.
                
                \nConsequently, **re-engaging with them holds the promise of securing purchases and contributing to Tasty Bytes' sales over the long term**, aligning with Tasty Bytes' objective of **achieving a 25% year-over-year growth over the next five years**.
                \nHence, the marketing team can **download the CSV file containing customer details of churned customers**. Armed with this information, the team can execute **personalized marketing strategies targeting churned customers**, cultivating enduring relationships that will help Tasty Bytes in achieving its goal.
                '''
                st.write(recommendation)
        else:
            recommendation = '''Non-churning customers have yielded a **greater uplift when compared to customers that have churned**. Non-churning customers are engaged and actively contributing to Tasty Bytes's sales. 
            This group of customers have **established their loyalty by contributing consistently to the sales stream**, this allows Tasty Bytes sales to be more predictable, allowing for **better financial planning and forecasting**.
            
            \nAs a strong base of non churning customers provides stable growth towards the business therefore, the marketing team can **download the CSV file containing customer details of non-churning customers**. Armed with this information, the team can execute **personalized marketing strategies targeting churned customers**, cultivating enduring relationships that will help Tasty Bytes in achieving its goal.
            '''
            st.write(recommendation)
        
    # Function to display model predictions in visulisations, seperated by tabs 
    def display_prediction_analysis(nonChurnTotalSales, nonChurn_totalUplift, nonChurn_percentUplift, nonChurn_df, churnTotalSales, churn_totalUplift, churn_percentUplift, churn_df):
        
        # Grouping results into different aspects
        totalSales = [nonChurnTotalSales, churnTotalSales]
        totalUplift = [nonChurn_totalUplift, churn_totalUplift]
        percentUplift = [nonChurn_percentUplift, churn_percentUplift]
        
        # Display group of customers
        index = ['Non-Churn','Churn']
        
        # Storing results as dataframes
        df_totalSales = pd.DataFrame({'Total Sales': totalSales}, index=index)
        df_totalUplift = pd.DataFrame({'Total Uplift': totalUplift}, index=index)
        df_percentUplift = pd.DataFrame({'Percentage Uplift': percentUplift}, index=index) 
        
        # Sub tab for user to navigate 
        tab1, tab2, tab3 = st.tabs(["Total Sales :dollar:", "Total Uplift :mag: ", "Percentage Uplift :mag:"])

        # Total Sales Tab
        with tab1:
            st.write("#### "+ "Analysis of Total Sales for Churn & Non-Churn Customers :heavy_dollar_sign:")
            # Bar chart for total sales analysis 
            totalSalesGraph = go.Figure()

            totalSalesGraph.add_trace(
                go.Bar(
                    x = index,
                    y = df_totalSales['Total Sales'],
                    marker_color='#003f5c',
                    text = df_totalSales['Total Sales'],
                    textposition='inside',
                    texttemplate='$%{y:,.0f}'
                )
            )

            # Customize the layout
            totalSalesGraph.update_layout(
                xaxis_title='Type of Customer',
                yaxis_title='Amount ($)',
                font=dict(
                size=12,
                color='#000000') ,
                showlegend=False
            )
            # Plotting
            st.plotly_chart(totalSalesGraph)
            # Display insights 
            nonChurnSalesMetric, churnSalesMetric = st.columns(2)
            nonChurnSalesMetric.metric(label="Sales Generated for Non-Churn Customers", value="${:,.2f}".format(nonChurnTotalSales))
            churnSalesMetric.metric(label="Sales Generated for Non-Churn Customers:", value="${:,.2f}".format(churnTotalSales))

        # Total Uplift Tab
        with tab2:
            st.write("#### "+ "Analysis of Total Uplift for Churn & Non-Churn Customers :chart_with_upwards_trend: ")
            totalUpliftGraph = go.Figure()

            totalUpliftGraph.add_trace(
                go.Bar(
                    x = index,
                    y = df_totalUplift['Total Uplift'],
                    marker_color='#003f5c',
                    text = df_totalUplift['Total Uplift'],
                    textposition='inside',
                    texttemplate='$%{y:,.0f}'
                )
            )

            # Customize the layout
            totalUpliftGraph.update_layout(
                xaxis_title='Type of Customer',
                yaxis_title='Amount ($)',
                font=dict(
                size=12,
                color='#000000') ,
                showlegend=False
            )
            # Plotting
            st.plotly_chart(totalUpliftGraph)
            # Display insights
            
            # Check for Negative uplift for non churn
            nonChurnUpliftMetric, churnUpliftMetric = st.columns(2)
            if (nonChurn_totalUplift >= 0):
                nonChurnUpliftMetric.metric(label="Uplift for Non-Churn Customers :large_green_circle:", value="${:,.2f}".format(nonChurn_totalUplift))
            else:
                nonChurnUpliftMetric.metric(label="Uplift for Non-Churn Customers :small_red_triangle_down:", value="${:,.2f}".format(nonChurn_totalUplift))
            # Chcek for Negative uplift for churn
            if (churn_totalUplift >= 0):
                churnUpliftMetric.metric(label="Uplift for Churn Customers :large_green_circle:", value="${:,.2f}".format(churn_totalUplift))
            else:
                churnUpliftMetric.metric(label="Uplift for Churn Customers :small_red_triangle_down:", value="${:,.2f}".format(churn_totalUplift))
            
            st.markdown("Displayed above, the uplift generated by **non-churn customers amounts to {:,.2f} stemming from the participation of {} US customers**. While the uplift attributed to **churn customers stands at {:,.2f} with the involvement of {} US customers.**".format(nonChurn_totalUplift, len(nonChurn_df.index), churn_totalUplift, len(churn_df.index)))
            
            # Recomendation and Call-To-Action
            recomendation_text(churn_totalUplift,nonChurn_totalUplift)
            downloadCustomerDetails(nonChurn_df, churn_df,key1="tabTotal1", key2="tabTotal2")
            
            

        # Percentage (%) Tab 
        with tab3:
            st.write("#### "+ "Analysis of Percentage Uplift (%) for Churn & Non-Churn Customers")
            percentUpliftGraph = go.Figure()

            percentUpliftGraph.add_trace(
                go.Bar(
                    x = index,
                    y = df_percentUplift['Percentage Uplift'],
                    marker_color='#003f5c',
                    text = df_percentUplift['Percentage Uplift'],
                    textposition='inside',
                    texttemplate='%{y:,.2f} %'
                )
            )

            # Customize the layout
            percentUpliftGraph.update_layout(
                xaxis_title='Type of Customer',
                yaxis_title='Percentage (%)',
                font=dict(
                size=12,
                color='#000000') ,
                showlegend=False
            )
            # Plotting
            st.plotly_chart(percentUpliftGraph)
            # Display insights
            nonChurnPercentMetric, churnPercentMetric = st.columns(2)
            # Check for Negative uplift for non churn
            if (nonChurn_percentUplift >= 0):
                nonChurnPercentMetric.metric(label="Percentage Uplift Non-Churn Customers :large_green_circle:", value="{:,.2f} %".format(nonChurn_percentUplift))
            else:
                nonChurnPercentMetric.metric(label="Percentage Uplift for Non-Churn Customers :small_red_triangle_down:", value="{:,.2f} %".format(nonChurn_percentUplift))
            # Check for Negative uplift for churn
            if (churn_percentUplift >= 0):
                churnPercentMetric.metric(label="Percentage Uplift for Churn Customers :large_green_circle:", value="{:,.2f} %".format(churn_percentUplift))
            else:
                churnPercentMetric.metric(label="Percentage Uplift for Churn Customers :small_red_triangle_down:", value="{:,.2f} %".format(churn_percentUplift))  
            
            st.markdown("Displayed above, the uplift generated by **non-churn customers amounts to {:,.2f}% uplift in sales stemming from the participation of {} US customers**. While the uplift attributed to **churn customers stands at {:,.2f}% uplift increase in sales with the involvement of {} US customers.**".format(nonChurn_percentUplift, len(nonChurn_df.index), churn_percentUplift, len(churn_df.index)))
            
            # Recomendation and Call-To-Action
            recomendation_text(churn_percentUplift,nonChurn_percentUplift)
            downloadCustomerDetails(nonChurn_df, churn_df,key1="tabPercent1", key2="tabPercent2")
            
            
    # Download filtered customer details     
    def downloadCustomerDetails(nonChurn_df, churn_df, key1, key2):   
        st.markdown("""---""")
        # Description of Callable Action
        st.markdown("###" +' :clipboard: Download Customer Details')
        callToActionDescription = '''
        Analysis on the predictions for churn and non-churn customer groups are displayed above, now you can download customer details that belongs to the respectively group by simply clicking a button :three_button_mouse:. 
        '''
        st.markdown(callToActionDescription)
        
        customerDetails_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.RAW_CUSTOMER.CUSTOMER_LOYALTY").to_pandas()
        
        download_nonChurn , download_churn = st.columns(2) # Position Left Right
        download_nonChurn.download_button(
            label="Non-Churn Customer Details as CSV",
            data= customerDetails_df[customerDetails_df["CUSTOMER_ID"].isin(nonChurn_df["CUSTOMER_ID"])].to_csv().encode('utf-8'),
            file_name='nonChurn_customer_details.csv',
            mime='text/csv',
            key=key1
        )
        download_churn.download_button(
            label="Churn Customers Details as CSV",
            data = customerDetails_df[customerDetails_df["CUSTOMER_ID"].isin(churn_df["CUSTOMER_ID"])].to_csv().encode('utf-8'),
            file_name='churn_customer_details.csv',
            mime='text/csv',
            key=key2
        )
    
    #=================================================================================================#
    # Tab Content
    
    # Load dataset
    days_to_next_purchase = load_next_purchase()

    # Draw the histogram
    histogram_days_to_next_purchase = go.Figure()

    histogram_days_to_next_purchase.add_trace(
        go.Histogram(
            name='Histogram of Days to next Purchase',
            x= days_to_next_purchase['TARGET'], marker_color='#003f5c', histnorm='density'
        )
    )
    # Customize histogram
    histogram_days_to_next_purchase.update_layout(
        title={
            'text' : 'Distribution of Days to Next Purchase',
            'x':0.5,
            'xanchor': 'center',
            "font" :dict(size=20)
        },
        xaxis_title='Days',
        yaxis_title='No. of Customers',
        showlegend=False
    )
    # Display the bar chart in the Streamlit tab
    st.plotly_chart(histogram_days_to_next_purchase)
    
    histogramAnalysis = '''
    Based on the histogram provided, it can be deduced that the distribution for days until the next purchase is skewed to the right. 
                The majority of US customers are anticipated to make a visit to Tasty Bytes within the upcoming 35 days.
    \nBy analyzing this plot, you can determine the number of days it takes for customers to churn, how many years they have been members, and then choose a specific timeframe for making predictions.'''
                
    st.markdown(histogramAnalysis)
    
    # Streamlit Tab Design
    # Define the user input fields
    st.write("###"+ " Predict Uplift")
    user_input_position_1 , user_input_position_2 = st.columns(2) # Position Left Right
    churn_days = get_churn_days(user_input_position_1) # Select customer base
    years_with_us_range = get_years_with_us(user_input_position_2) # Select duration of membership 
    timeframe_input = get_timeframe() # Select Timeframe Model 
    city_input = get_city() # Select City 
    
    
    if st.button('Predict Uplift', key='Uplift Analysis'):
        # 2 Week Model timeframe 
        if (timeframe_input == '2 Weeks'):
            # Load the 2W Uplift Model
            uplift_predictions_2W = load_Uplift_Churn_2W()
            
            # Load the next purchase date 
            next_purchase_data = load_next_purchase()
            # Get city int  
            city_int = get_city_int(city_input)
            if (len(city_int) == 0):
                st.write("To proceed, please choose at least one city.")
            else:
                cluster_sales = load_cluster_sales_2W()
                # --- Uplift Analysis for Non-Churn Customers --- 
                # Filter for customer that did not churn (lower than churn days)
                filtered_lower = filter_lower(next_purchase_data, uplift_predictions_2W, churn_days, years_with_us_range, city_int)
                pred = uplift_2W.predict_proba(filtered_lower)
                pred_df_lower = pd.DataFrame(pred, index=filtered_lower.index)
                
                nonChurn_df, nonChurnTotalSales, nonChurn_totalUplift, nonChurn_percentUplift = find_uplift(pred_df_lower, cluster_sales, uplift_predictions_2W)

                # --- Uplift Analysis for Churn Customers ---
                # Filter for customer that churn (higher than churn days)
                filtered_higher = filter_higher(next_purchase_data, uplift_predictions_2W, churn_days, years_with_us_range, city_int)
                
                # Make a prediction, write as dataframe
                pred = uplift_2W.predict_proba(filtered_higher)
                pred_df_higher = pd.DataFrame(pred, index=filtered_higher.index)

                churn_df, churnTotalSales, churn_totalUplift, churn_percentUplift = find_uplift(pred_df_higher, cluster_sales, uplift_predictions_2W)
                
                # Compare Churn and Non Churn 
                display_prediction_analysis(nonChurnTotalSales, nonChurn_totalUplift, nonChurn_percentUplift, nonChurn_df, churnTotalSales, churn_totalUplift, churn_percentUplift, churn_df) 
                
        # 1 Month Model timeframe    
        elif (timeframe_input == '1 Month'):
            # Load the 1M Uplift Model
            uplift_predictions_1M = load_Uplift_Churn_1M()
            
            # Load the next purchase date 
            next_purchase_data = load_next_purchase()
            # Get city int  
            city_int = get_city_int(city_input)
            if (len(city_int) == 0):
                st.write("To proceed, please choose at least one city.")
            else:
                cluster_sales = load_cluster_sales_1M()
                # --- Uplift Analysis for Non-Churn Customers --- 
                # Filter for customer that did not churn (lower than churn days)
                filtered_lower = filter_lower(next_purchase_data, uplift_predictions_1M, churn_days, years_with_us_range, city_int)
                pred = uplift_1M.predict_proba(filtered_lower)
                pred_df_lower = pd.DataFrame(pred, index=filtered_lower.index)
    
                nonChurn_df, nonChurnTotalSales, nonChurn_totalUplift, nonChurn_percentUplift = find_uplift(pred_df_lower, cluster_sales, uplift_predictions_1M)
                
                # --- Uplift Analysis for Churn Customers ---
                # Filter for customer that churn (higher than churn days)
                filtered_higher = filter_higher(next_purchase_data, uplift_predictions_1M, churn_days, years_with_us_range, city_int)
                
                # Make a prediction, write as dataframe
                pred = uplift_1M.predict_proba(filtered_higher)
                pred_df_higher = pd.DataFrame(pred, index=filtered_higher.index)
    
                churn_df, churnTotalSales, churn_totalUplift, churn_percentUplift = find_uplift(pred_df_higher, cluster_sales, uplift_predictions_1M)
                
                # Compare Churn and Non Churn 
                display_prediction_analysis(nonChurnTotalSales, nonChurn_totalUplift, nonChurn_percentUplift, nonChurn_df, churnTotalSales, churn_totalUplift, churn_percentUplift, churn_df)
                
        # 3 Month Model timeframe 
        elif (timeframe_input == '3 Months'):
            # Load the 3M Uplift Model
            uplift_predictions_3M = load_Uplift_Churn_3M()
            
            # Load the next purchase date 
            next_purchase_data = load_next_purchase()
            # Get city int  
            city_int = get_city_int(city_input)
            if (len(city_int) == 0):
                st.write("To proceed, please choose at least one city.")
            else:
                cluster_sales = load_cluster_sales_3M()
                # --- Uplift Analysis for Non-Churn Customers --- 
                # Filter for customer that did not churn (lower than churn days)
                filtered_lower = filter_lower(next_purchase_data, uplift_predictions_3M, churn_days, years_with_us_range, city_int)
                pred = uplift_3M.predict_proba(filtered_lower)
                pred_df_lower = pd.DataFrame(pred, index=filtered_lower.index)                
    
                nonChurn_df, nonChurnTotalSales, nonChurn_totalUplift, nonChurn_percentUplift = find_uplift(pred_df_lower, cluster_sales, uplift_predictions_3M)
                
                # --- Uplift Analysis for Churn Customers ---
                # Filter for customer that churn (higher than churn days)
                filtered_higher = filter_higher(next_purchase_data, uplift_predictions_3M, churn_days, years_with_us_range, city_int)
                
                # Make a prediction, write as dataframe
                pred = uplift_3M.predict_proba(filtered_higher)
                pred_df_higher = pd.DataFrame(pred, index=filtered_higher.index)
    
                churn_df, churnTotalSales, churn_totalUplift, churn_percentUplift = find_uplift(pred_df_higher, cluster_sales, uplift_predictions_3M)
                
                 # Compare Churn and Non Churn 
                display_prediction_analysis(nonChurnTotalSales, nonChurn_totalUplift, nonChurn_percentUplift, nonChurn_df, churnTotalSales, churn_totalUplift, churn_percentUplift, churn_df)
    
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
               Gaining insights into future demand allow truck owners to optimize their menu by focusing on food high in demand\
               , which will attract more customers, increase in profit and ultimately increase in sales!')
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
