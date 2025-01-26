# Customer Churn Analysis in Telecom Industry

---
jupyter:
  kaggle:
    accelerator: none
    dataSources:
    - datasetId: 4968559
      sourceId: 8360350
      sourceType: datasetVersion
    dockerImageVersionId: 30732
    isGpuEnabled: false
    isInternetEnabled: false
    language: python
    sourceType: notebook
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.8
  nbformat: 4
  nbformat_minor: 4
---

::: {.cell .markdown}
# Customer Churn Analysis in Telecom Industry
:::

::: {.cell .markdown}
### Introduction:

In the highly competitive telecommunications industry, understanding the
factors driving customer churn is crucial for maintaining a robust
customer base and ensuring long-term profitability. This analysis delves
into various aspects of customer data to identify key indicators of
churn and suggest actionable strategies for enhancing customer
retention. By examining relationships between customer attributes and
churn, we aim to uncover patterns and insights that can inform more
effective business decisions.
:::

::: {.cell .markdown}
1.  **Title Page**:
    -   Title: Customer Churn Analysis in Telecom Industry
    -   Subtitle: Insights from Customer Data Analysis
    -   Date: \[Insert Date\]
    -   Author: Swastik Tripathi `<hr>`{=html}
:::

::: {.cell .code execution_count="1" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" execution="{\"iopub.execute_input\":\"2024-06-07T06:14:01.961049Z\",\"iopub.status.busy\":\"2024-06-07T06:14:01.960571Z\",\"iopub.status.idle\":\"2024-06-07T06:14:04.823231Z\",\"shell.execute_reply\":\"2024-06-07T06:14:04.821884Z\",\"shell.execute_reply.started\":\"2024-06-07T06:14:01.961013Z\"}"}
``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
:::

::: {.cell .markdown}
### Loading dataset
:::

::: {.cell .code execution_count="2" execution="{\"iopub.execute_input\":\"2024-06-07T06:14:04.826482Z\",\"iopub.status.busy\":\"2024-06-07T06:14:04.825836Z\",\"iopub.status.idle\":\"2024-06-07T06:14:04.944455Z\",\"shell.execute_reply\":\"2024-06-07T06:14:04.943316Z\",\"shell.execute_reply.started\":\"2024-06-07T06:14:04.826431Z\"}"}
``` python
data = pd.read_csv('telco.csv')
```
:::

::: {.cell .code execution_count="3" execution="{\"iopub.execute_input\":\"2024-06-07T06:14:06.271119Z\",\"iopub.status.busy\":\"2024-06-07T06:14:06.270557Z\",\"iopub.status.idle\":\"2024-06-07T06:14:06.277709Z\",\"shell.execute_reply\":\"2024-06-07T06:14:06.276062Z\",\"shell.execute_reply.started\":\"2024-06-07T06:14:06.271074Z\"}"}
``` python
pd.set_option('display.max_columns', None)
```
:::

::: {.cell .code execution_count="4" execution="{\"iopub.execute_input\":\"2024-06-07T06:14:13.605723Z\",\"iopub.status.busy\":\"2024-06-07T06:14:13.605277Z\",\"iopub.status.idle\":\"2024-06-07T06:14:13.667122Z\",\"shell.execute_reply\":\"2024-06-07T06:14:13.665969Z\",\"shell.execute_reply.started\":\"2024-06-07T06:14:13.605691Z\"}"}
``` python
data.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7043 entries, 0 to 7042
    Data columns (total 50 columns):
     #   Column                             Non-Null Count  Dtype  
    ---  ------                             --------------  -----  
     0   Customer ID                        7043 non-null   object 
     1   Gender                             7043 non-null   object 
     2   Age                                7043 non-null   int64  
     3   Under 30                           7043 non-null   object 
     4   Senior Citizen                     7043 non-null   object 
     5   Married                            7043 non-null   object 
     6   Dependents                         7043 non-null   object 
     7   Number of Dependents               7043 non-null   int64  
     8   Country                            7043 non-null   object 
     9   State                              7043 non-null   object 
     10  City                               7043 non-null   object 
     11  Zip Code                           7043 non-null   int64  
     12  Latitude                           7043 non-null   float64
     13  Longitude                          7043 non-null   float64
     14  Population                         7043 non-null   int64  
     15  Quarter                            7043 non-null   object 
     16  Referred a Friend                  7043 non-null   object 
     17  Number of Referrals                7043 non-null   int64  
     18  Tenure in Months                   7043 non-null   int64  
     19  Offer                              3166 non-null   object 
     20  Phone Service                      7043 non-null   object 
     21  Avg Monthly Long Distance Charges  7043 non-null   float64
     22  Multiple Lines                     7043 non-null   object 
     23  Internet Service                   7043 non-null   object 
     24  Internet Type                      5517 non-null   object 
     25  Avg Monthly GB Download            7043 non-null   int64  
     26  Online Security                    7043 non-null   object 
     27  Online Backup                      7043 non-null   object 
     28  Device Protection Plan             7043 non-null   object 
     29  Premium Tech Support               7043 non-null   object 
     30  Streaming TV                       7043 non-null   object 
     31  Streaming Movies                   7043 non-null   object 
     32  Streaming Music                    7043 non-null   object 
     33  Unlimited Data                     7043 non-null   object 
     34  Contract                           7043 non-null   object 
     35  Paperless Billing                  7043 non-null   object 
     36  Payment Method                     7043 non-null   object 
     37  Monthly Charge                     7043 non-null   float64
     38  Total Charges                      7043 non-null   float64
     39  Total Refunds                      7043 non-null   float64
     40  Total Extra Data Charges           7043 non-null   int64  
     41  Total Long Distance Charges        7043 non-null   float64
     42  Total Revenue                      7043 non-null   float64
     43  Satisfaction Score                 7043 non-null   int64  
     44  Customer Status                    7043 non-null   object 
     45  Churn Label                        7043 non-null   object 
     46  Churn Score                        7043 non-null   int64  
     47  CLTV                               7043 non-null   int64  
     48  Churn Category                     1869 non-null   object 
     49  Churn Reason                       1869 non-null   object 
    dtypes: float64(8), int64(11), object(31)
    memory usage: 2.7+ MB
:::
:::

::: {.cell .code execution_count="5" execution="{\"iopub.execute_input\":\"2024-06-07T06:14:42.944422Z\",\"iopub.status.busy\":\"2024-06-07T06:14:42.943984Z\",\"iopub.status.idle\":\"2024-06-07T06:14:43.038801Z\",\"shell.execute_reply\":\"2024-06-07T06:14:43.037673Z\",\"shell.execute_reply.started\":\"2024-06-07T06:14:42.944388Z\"}"}
``` python
data.describe()
```

::: {.output .execute_result execution_count="5"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Number of Dependents</th>
      <th>Zip Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Population</th>
      <th>Number of Referrals</th>
      <th>Tenure in Months</th>
      <th>Avg Monthly Long Distance Charges</th>
      <th>Avg Monthly GB Download</th>
      <th>Monthly Charge</th>
      <th>Total Charges</th>
      <th>Total Refunds</th>
      <th>Total Extra Data Charges</th>
      <th>Total Long Distance Charges</th>
      <th>Total Revenue</th>
      <th>Satisfaction Score</th>
      <th>Churn Score</th>
      <th>CLTV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
      <td>7043.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>46.509726</td>
      <td>0.468692</td>
      <td>93486.070567</td>
      <td>36.197455</td>
      <td>-119.756684</td>
      <td>22139.603294</td>
      <td>1.951867</td>
      <td>32.386767</td>
      <td>22.958954</td>
      <td>20.515405</td>
      <td>64.761692</td>
      <td>2280.381264</td>
      <td>1.962182</td>
      <td>6.860713</td>
      <td>749.099262</td>
      <td>3034.379056</td>
      <td>3.244924</td>
      <td>58.505040</td>
      <td>4400.295755</td>
    </tr>
    <tr>
      <th>std</th>
      <td>16.750352</td>
      <td>0.962802</td>
      <td>1856.767505</td>
      <td>2.468929</td>
      <td>2.154425</td>
      <td>21152.392837</td>
      <td>3.001199</td>
      <td>24.542061</td>
      <td>15.448113</td>
      <td>20.418940</td>
      <td>30.090047</td>
      <td>2266.220462</td>
      <td>7.902614</td>
      <td>25.104978</td>
      <td>846.660055</td>
      <td>2865.204542</td>
      <td>1.201657</td>
      <td>21.170031</td>
      <td>1183.057152</td>
    </tr>
    <tr>
      <th>min</th>
      <td>19.000000</td>
      <td>0.000000</td>
      <td>90001.000000</td>
      <td>32.555828</td>
      <td>-124.301372</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>18.250000</td>
      <td>18.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21.360000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>2003.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.000000</td>
      <td>0.000000</td>
      <td>92101.000000</td>
      <td>33.990646</td>
      <td>-121.788090</td>
      <td>2344.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>9.210000</td>
      <td>3.000000</td>
      <td>35.500000</td>
      <td>400.150000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>70.545000</td>
      <td>605.610000</td>
      <td>3.000000</td>
      <td>40.000000</td>
      <td>3469.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>46.000000</td>
      <td>0.000000</td>
      <td>93518.000000</td>
      <td>36.205465</td>
      <td>-119.595293</td>
      <td>17554.000000</td>
      <td>0.000000</td>
      <td>29.000000</td>
      <td>22.890000</td>
      <td>17.000000</td>
      <td>70.350000</td>
      <td>1394.550000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>401.440000</td>
      <td>2108.640000</td>
      <td>3.000000</td>
      <td>61.000000</td>
      <td>4527.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>60.000000</td>
      <td>0.000000</td>
      <td>95329.000000</td>
      <td>38.161321</td>
      <td>-117.969795</td>
      <td>36125.000000</td>
      <td>3.000000</td>
      <td>55.000000</td>
      <td>36.395000</td>
      <td>27.000000</td>
      <td>89.850000</td>
      <td>3786.600000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1191.100000</td>
      <td>4801.145000</td>
      <td>4.000000</td>
      <td>75.500000</td>
      <td>5380.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.000000</td>
      <td>9.000000</td>
      <td>96150.000000</td>
      <td>41.962127</td>
      <td>-114.192901</td>
      <td>105285.000000</td>
      <td>11.000000</td>
      <td>72.000000</td>
      <td>49.990000</td>
      <td>85.000000</td>
      <td>118.750000</td>
      <td>8684.800000</td>
      <td>49.790000</td>
      <td>150.000000</td>
      <td>3564.720000</td>
      <td>11979.340000</td>
      <td>5.000000</td>
      <td>96.000000</td>
      <td>6500.000000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="6" execution="{\"iopub.execute_input\":\"2024-06-07T06:15:37.436192Z\",\"iopub.status.busy\":\"2024-06-07T06:15:37.435740Z\",\"iopub.status.idle\":\"2024-06-07T06:15:37.491442Z\",\"shell.execute_reply\":\"2024-06-07T06:15:37.490149Z\",\"shell.execute_reply.started\":\"2024-06-07T06:15:37.436153Z\"}"}
``` python
data.sample(5)
```

::: {.output .execute_result execution_count="6"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Under 30</th>
      <th>Senior Citizen</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Number of Dependents</th>
      <th>Country</th>
      <th>State</th>
      <th>City</th>
      <th>Zip Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Population</th>
      <th>Quarter</th>
      <th>Referred a Friend</th>
      <th>Number of Referrals</th>
      <th>Tenure in Months</th>
      <th>Offer</th>
      <th>Phone Service</th>
      <th>Avg Monthly Long Distance Charges</th>
      <th>Multiple Lines</th>
      <th>Internet Service</th>
      <th>Internet Type</th>
      <th>Avg Monthly GB Download</th>
      <th>Online Security</th>
      <th>Online Backup</th>
      <th>Device Protection Plan</th>
      <th>Premium Tech Support</th>
      <th>Streaming TV</th>
      <th>Streaming Movies</th>
      <th>Streaming Music</th>
      <th>Unlimited Data</th>
      <th>Contract</th>
      <th>Paperless Billing</th>
      <th>Payment Method</th>
      <th>Monthly Charge</th>
      <th>Total Charges</th>
      <th>Total Refunds</th>
      <th>Total Extra Data Charges</th>
      <th>Total Long Distance Charges</th>
      <th>Total Revenue</th>
      <th>Satisfaction Score</th>
      <th>Customer Status</th>
      <th>Churn Label</th>
      <th>Churn Score</th>
      <th>CLTV</th>
      <th>Churn Category</th>
      <th>Churn Reason</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4359</th>
      <td>6988-CJEYV</td>
      <td>Male</td>
      <td>56</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>United States</td>
      <td>California</td>
      <td>Crows Landing</td>
      <td>95313</td>
      <td>37.435664</td>
      <td>-121.049056</td>
      <td>1508</td>
      <td>Q3</td>
      <td>No</td>
      <td>0</td>
      <td>49</td>
      <td>Offer B</td>
      <td>Yes</td>
      <td>45.46</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber Optic</td>
      <td>27</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-Month</td>
      <td>No</td>
      <td>Bank Withdrawal</td>
      <td>98.7</td>
      <td>4920.55</td>
      <td>0.0</td>
      <td>0</td>
      <td>2227.54</td>
      <td>7148.09</td>
      <td>3</td>
      <td>Stayed</td>
      <td>No</td>
      <td>79</td>
      <td>6148</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>141</th>
      <td>1587-FKLZB</td>
      <td>Male</td>
      <td>79</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>0</td>
      <td>United States</td>
      <td>California</td>
      <td>Santa Barbara</td>
      <td>93101</td>
      <td>34.419203</td>
      <td>-119.710008</td>
      <td>31727</td>
      <td>Q3</td>
      <td>Yes</td>
      <td>1</td>
      <td>66</td>
      <td>Offer A</td>
      <td>Yes</td>
      <td>28.86</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber Optic</td>
      <td>25</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Month-to-Month</td>
      <td>Yes</td>
      <td>Credit Card</td>
      <td>99.5</td>
      <td>6822.15</td>
      <td>0.0</td>
      <td>0</td>
      <td>1904.76</td>
      <td>8726.91</td>
      <td>1</td>
      <td>Churned</td>
      <td>Yes</td>
      <td>96</td>
      <td>4479</td>
      <td>Competitor</td>
      <td>Competitor had better devices</td>
    </tr>
    <tr>
      <th>730</th>
      <td>4526-ZJJTM</td>
      <td>Female</td>
      <td>72</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>0</td>
      <td>United States</td>
      <td>California</td>
      <td>San Mateo</td>
      <td>94403</td>
      <td>37.538309</td>
      <td>-122.305109</td>
      <td>37926</td>
      <td>Q3</td>
      <td>Yes</td>
      <td>1</td>
      <td>25</td>
      <td>NaN</td>
      <td>Yes</td>
      <td>19.44</td>
      <td>No</td>
      <td>Yes</td>
      <td>Fiber Optic</td>
      <td>4</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two Year</td>
      <td>No</td>
      <td>Bank Withdrawal</td>
      <td>88.4</td>
      <td>2191.15</td>
      <td>0.0</td>
      <td>0</td>
      <td>486.00</td>
      <td>2677.15</td>
      <td>3</td>
      <td>Stayed</td>
      <td>No</td>
      <td>51</td>
      <td>4849</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>398</th>
      <td>5383-MMTWC</td>
      <td>Female</td>
      <td>75</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>0</td>
      <td>United States</td>
      <td>California</td>
      <td>Oakley</td>
      <td>94561</td>
      <td>37.999406</td>
      <td>-121.686241</td>
      <td>27607</td>
      <td>Q3</td>
      <td>Yes</td>
      <td>1</td>
      <td>8</td>
      <td>NaN</td>
      <td>Yes</td>
      <td>46.05</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber Optic</td>
      <td>22</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Month-to-Month</td>
      <td>Yes</td>
      <td>Bank Withdrawal</td>
      <td>84.0</td>
      <td>613.40</td>
      <td>0.0</td>
      <td>0</td>
      <td>368.40</td>
      <td>981.80</td>
      <td>3</td>
      <td>Churned</td>
      <td>Yes</td>
      <td>91</td>
      <td>4819</td>
      <td>Dissatisfaction</td>
      <td>Limited range of services</td>
    </tr>
    <tr>
      <th>1306</th>
      <td>2207-NHRJK</td>
      <td>Male</td>
      <td>58</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>United States</td>
      <td>California</td>
      <td>Carmel</td>
      <td>93923</td>
      <td>36.460611</td>
      <td>-121.852507</td>
      <td>13121</td>
      <td>Q3</td>
      <td>No</td>
      <td>0</td>
      <td>1</td>
      <td>Offer E</td>
      <td>Yes</td>
      <td>1.28</td>
      <td>No</td>
      <td>Yes</td>
      <td>DSL</td>
      <td>27</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Month-to-Month</td>
      <td>No</td>
      <td>Bank Withdrawal</td>
      <td>50.8</td>
      <td>50.80</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.28</td>
      <td>52.08</td>
      <td>1</td>
      <td>Churned</td>
      <td>Yes</td>
      <td>85</td>
      <td>3718</td>
      <td>Competitor</td>
      <td>Competitor had better devices</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
### Cleaing dataset
:::

::: {.cell .code execution_count="7" execution="{\"iopub.execute_input\":\"2024-06-07T06:17:34.021766Z\",\"iopub.status.busy\":\"2024-06-07T06:17:34.021206Z\",\"iopub.status.idle\":\"2024-06-07T06:17:34.058030Z\",\"shell.execute_reply\":\"2024-06-07T06:17:34.056676Z\",\"shell.execute_reply.started\":\"2024-06-07T06:17:34.021718Z\"}"}
``` python
data.isnull().sum()
```

::: {.output .execute_result execution_count="7"}
    Customer ID                             0
    Gender                                  0
    Age                                     0
    Under 30                                0
    Senior Citizen                          0
    Married                                 0
    Dependents                              0
    Number of Dependents                    0
    Country                                 0
    State                                   0
    City                                    0
    Zip Code                                0
    Latitude                                0
    Longitude                               0
    Population                              0
    Quarter                                 0
    Referred a Friend                       0
    Number of Referrals                     0
    Tenure in Months                        0
    Offer                                3877
    Phone Service                           0
    Avg Monthly Long Distance Charges       0
    Multiple Lines                          0
    Internet Service                        0
    Internet Type                        1526
    Avg Monthly GB Download                 0
    Online Security                         0
    Online Backup                           0
    Device Protection Plan                  0
    Premium Tech Support                    0
    Streaming TV                            0
    Streaming Movies                        0
    Streaming Music                         0
    Unlimited Data                          0
    Contract                                0
    Paperless Billing                       0
    Payment Method                          0
    Monthly Charge                          0
    Total Charges                           0
    Total Refunds                           0
    Total Extra Data Charges                0
    Total Long Distance Charges             0
    Total Revenue                           0
    Satisfaction Score                      0
    Customer Status                         0
    Churn Label                             0
    Churn Score                             0
    CLTV                                    0
    Churn Category                       5174
    Churn Reason                         5174
    dtype: int64
:::
:::

::: {.cell .code execution_count="8" execution="{\"iopub.execute_input\":\"2024-06-07T06:20:58.690789Z\",\"iopub.status.busy\":\"2024-06-07T06:20:58.690339Z\",\"iopub.status.idle\":\"2024-06-07T06:20:58.699701Z\",\"shell.execute_reply\":\"2024-06-07T06:20:58.698250Z\",\"shell.execute_reply.started\":\"2024-06-07T06:20:58.690751Z\"}"}
``` python
data['Offer'] = data['Offer'].fillna('No offer')
```
:::

::: {.cell .code execution_count="9" execution="{\"iopub.execute_input\":\"2024-06-07T06:24:50.171079Z\",\"iopub.status.busy\":\"2024-06-07T06:24:50.170643Z\",\"iopub.status.idle\":\"2024-06-07T06:24:50.185055Z\",\"shell.execute_reply\":\"2024-06-07T06:24:50.183566Z\",\"shell.execute_reply.started\":\"2024-06-07T06:24:50.171044Z\"}"}
``` python
(data[data['Internet Type'].isnull()]['Internet Service'] == 'Yes').sum()
```

::: {.output .execute_result execution_count="9"}
    0
:::
:::

::: {.cell .code execution_count="10" execution="{\"iopub.execute_input\":\"2024-06-07T06:26:04.291302Z\",\"iopub.status.busy\":\"2024-06-07T06:26:04.290729Z\",\"iopub.status.idle\":\"2024-06-07T06:26:04.301494Z\",\"shell.execute_reply\":\"2024-06-07T06:26:04.299991Z\",\"shell.execute_reply.started\":\"2024-06-07T06:26:04.291252Z\"}"}
``` python
data['Internet Service'].unique()
```

::: {.output .execute_result execution_count="10"}
    array(['Yes', 'No'], dtype=object)
:::
:::

::: {.cell .markdown}
means that the column \'Internet Type\' is only Null when \'Internet
Service\' is \'No\'. it means whenever \'Internet Type\' is Null, we can
change it to \'No Internet\'
:::

::: {.cell .code execution_count="11" execution="{\"iopub.execute_input\":\"2024-06-07T06:40:45.686236Z\",\"iopub.status.busy\":\"2024-06-07T06:40:45.685779Z\",\"iopub.status.idle\":\"2024-06-07T06:40:45.695106Z\",\"shell.execute_reply\":\"2024-06-07T06:40:45.693834Z\",\"shell.execute_reply.started\":\"2024-06-07T06:40:45.686194Z\"}"}
``` python
data['Internet Type'] = data['Internet Type'].fillna('No Internet')
```
:::

::: {.cell .code execution_count="12" execution="{\"iopub.execute_input\":\"2024-06-07T06:34:43.162059Z\",\"iopub.status.busy\":\"2024-06-07T06:34:43.161624Z\",\"iopub.status.idle\":\"2024-06-07T06:34:43.178254Z\",\"shell.execute_reply\":\"2024-06-07T06:34:43.176976Z\",\"shell.execute_reply.started\":\"2024-06-07T06:34:43.162023Z\"}"}
``` python
(data[data['Churn Category'].isnull()]['Churn Label'] == 'Yes').sum()
```

::: {.output .execute_result execution_count="12"}
    0
:::
:::

::: {.cell .code execution_count="13" execution="{\"iopub.execute_input\":\"2024-06-07T06:36:09.603155Z\",\"iopub.status.busy\":\"2024-06-07T06:36:09.602063Z\",\"iopub.status.idle\":\"2024-06-07T06:36:09.623726Z\",\"shell.execute_reply\":\"2024-06-07T06:36:09.622045Z\",\"shell.execute_reply.started\":\"2024-06-07T06:36:09.603096Z\"}"}
``` python
(data[data['Churn Reason'].isnull()]['Churn Label'] == 'Yes').sum()
```

::: {.output .execute_result execution_count="13"}
    0
:::
:::

::: {.cell .markdown}
means \'Churn Category\' and \'Churn reason\' is only Null if \'Customer
Label\' indicates that the customer hasn\'t churned
:::

::: {.cell .code execution_count="14" execution="{\"iopub.execute_input\":\"2024-06-07T06:37:26.441497Z\",\"iopub.status.busy\":\"2024-06-07T06:37:26.441087Z\",\"iopub.status.idle\":\"2024-06-07T06:37:26.449933Z\",\"shell.execute_reply\":\"2024-06-07T06:37:26.448438Z\",\"shell.execute_reply.started\":\"2024-06-07T06:37:26.441465Z\"}"}
``` python
data['Churn Category'] = data['Churn Category'].fillna('Not Churned')
```
:::

::: {.cell .code execution_count="15" execution="{\"iopub.execute_input\":\"2024-06-07T06:37:50.326831Z\",\"iopub.status.busy\":\"2024-06-07T06:37:50.326310Z\",\"iopub.status.idle\":\"2024-06-07T06:37:50.334909Z\",\"shell.execute_reply\":\"2024-06-07T06:37:50.333513Z\",\"shell.execute_reply.started\":\"2024-06-07T06:37:50.326792Z\"}"}
``` python
data['Churn Reason'] = data['Churn Reason'].fillna('Not Churned')
```
:::

::: {.cell .code execution_count="16" execution="{\"iopub.execute_input\":\"2024-06-07T06:39:34.735617Z\",\"iopub.status.busy\":\"2024-06-07T06:39:34.735145Z\",\"iopub.status.idle\":\"2024-06-07T06:39:34.773626Z\",\"shell.execute_reply\":\"2024-06-07T06:39:34.772297Z\",\"shell.execute_reply.started\":\"2024-06-07T06:39:34.735561Z\"}"}
``` python
data.isnull().sum()
```

::: {.output .execute_result execution_count="16"}
    Customer ID                          0
    Gender                               0
    Age                                  0
    Under 30                             0
    Senior Citizen                       0
    Married                              0
    Dependents                           0
    Number of Dependents                 0
    Country                              0
    State                                0
    City                                 0
    Zip Code                             0
    Latitude                             0
    Longitude                            0
    Population                           0
    Quarter                              0
    Referred a Friend                    0
    Number of Referrals                  0
    Tenure in Months                     0
    Offer                                0
    Phone Service                        0
    Avg Monthly Long Distance Charges    0
    Multiple Lines                       0
    Internet Service                     0
    Internet Type                        0
    Avg Monthly GB Download              0
    Online Security                      0
    Online Backup                        0
    Device Protection Plan               0
    Premium Tech Support                 0
    Streaming TV                         0
    Streaming Movies                     0
    Streaming Music                      0
    Unlimited Data                       0
    Contract                             0
    Paperless Billing                    0
    Payment Method                       0
    Monthly Charge                       0
    Total Charges                        0
    Total Refunds                        0
    Total Extra Data Charges             0
    Total Long Distance Charges          0
    Total Revenue                        0
    Satisfaction Score                   0
    Customer Status                      0
    Churn Label                          0
    Churn Score                          0
    CLTV                                 0
    Churn Category                       0
    Churn Reason                         0
    dtype: int64
:::
:::

::: {.cell .code execution_count="17" execution="{\"iopub.execute_input\":\"2024-06-07T06:49:51.931390Z\",\"iopub.status.busy\":\"2024-06-07T06:49:51.930936Z\",\"iopub.status.idle\":\"2024-06-07T06:49:51.941845Z\",\"shell.execute_reply\":\"2024-06-07T06:49:51.940321Z\",\"shell.execute_reply.started\":\"2024-06-07T06:49:51.931357Z\"}"}
``` python
data = data.drop('Under 30', axis=1)
```
:::

::: {.cell .markdown}
since \'Under 30\' column can be derived from \'Age\' column, it becomes
redundant
:::

::: {.cell .code execution_count="18"}
``` python
data[data['Senior Citizen'] == 'No'].sort_values('Age', ascending=False).head(1)
```

::: {.output .execute_result execution_count="18"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Senior Citizen</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Number of Dependents</th>
      <th>Country</th>
      <th>State</th>
      <th>City</th>
      <th>Zip Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Population</th>
      <th>Quarter</th>
      <th>Referred a Friend</th>
      <th>Number of Referrals</th>
      <th>Tenure in Months</th>
      <th>Offer</th>
      <th>Phone Service</th>
      <th>Avg Monthly Long Distance Charges</th>
      <th>Multiple Lines</th>
      <th>Internet Service</th>
      <th>Internet Type</th>
      <th>Avg Monthly GB Download</th>
      <th>Online Security</th>
      <th>Online Backup</th>
      <th>Device Protection Plan</th>
      <th>Premium Tech Support</th>
      <th>Streaming TV</th>
      <th>Streaming Movies</th>
      <th>Streaming Music</th>
      <th>Unlimited Data</th>
      <th>Contract</th>
      <th>Paperless Billing</th>
      <th>Payment Method</th>
      <th>Monthly Charge</th>
      <th>Total Charges</th>
      <th>Total Refunds</th>
      <th>Total Extra Data Charges</th>
      <th>Total Long Distance Charges</th>
      <th>Total Revenue</th>
      <th>Satisfaction Score</th>
      <th>Customer Status</th>
      <th>Churn Label</th>
      <th>Churn Score</th>
      <th>CLTV</th>
      <th>Churn Category</th>
      <th>Churn Reason</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5850</th>
      <td>8294-UIMBA</td>
      <td>Female</td>
      <td>64</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>United States</td>
      <td>California</td>
      <td>Northridge</td>
      <td>91325</td>
      <td>34.236683</td>
      <td>-118.517588</td>
      <td>32307</td>
      <td>Q3</td>
      <td>No</td>
      <td>0</td>
      <td>30</td>
      <td>No offer</td>
      <td>Yes</td>
      <td>2.67</td>
      <td>No</td>
      <td>Yes</td>
      <td>Fiber Optic</td>
      <td>27</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One Year</td>
      <td>Yes</td>
      <td>Bank Withdrawal</td>
      <td>94.4</td>
      <td>2638.1</td>
      <td>0.0</td>
      <td>0</td>
      <td>80.1</td>
      <td>2718.2</td>
      <td>5</td>
      <td>Stayed</td>
      <td>No</td>
      <td>68</td>
      <td>4584</td>
      <td>Not Churned</td>
      <td>Not Churned</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
this means that people above 64 year old are considered senior citizen.
since we can determine senior citizen status from age attribute, we can
drop this column.
:::

::: {.cell .code execution_count="19"}
``` python
data.drop(['Senior Citizen'], axis=1, inplace=True)
```
:::

::: {.cell .markdown}
## convert to categorical values
:::

::: {.cell .code execution_count="20" execution="{\"iopub.execute_input\":\"2024-06-07T06:53:15.887920Z\",\"iopub.status.busy\":\"2024-06-07T06:53:15.887472Z\",\"iopub.status.idle\":\"2024-06-07T06:53:15.894886Z\",\"shell.execute_reply\":\"2024-06-07T06:53:15.893545Z\",\"shell.execute_reply.started\":\"2024-06-07T06:53:15.887885Z\"}"}
``` python
data['Gender'] = data['Gender'].astype('category')
data['Married'] = data['Married'].astype('category')
data['Dependents'] = data['Dependents'].astype('category')
data['State'] = data['State'].astype('category')
data['Quarter'] = data['Quarter'].astype('category')
data['Referred a Friend'] = data['Referred a Friend'].astype('category')
data['Phone Service'] = data['Phone Service'].astype('category')
data['Multiple Lines'] = data['Multiple Lines'].astype('category')
data['Internet Service'] = data['Internet Service'].astype('category')
data['Online Security'] = data['Online Security'].astype('category')
data['Online Backup'] = data['Online Backup'].astype('category')
data['Device Protection Plan'] = data['Device Protection Plan'].astype('category')
data['Premium Tech Support'] = data['Premium Tech Support'].astype('category')
data['Streaming TV'] = data['Streaming TV'].astype('category')
data['Streaming Movies'] = data['Streaming Movies'].astype('category')
data['Streaming Music'] = data['Streaming Music'].astype('category')
data['Unlimited Data'] = data['Unlimited Data'].astype('category')
data['Paperless Billing'] = data['Paperless Billing'].astype('category')
data['Payment Method'] = data['Payment Method'].astype('category')
data['Customer Status'] = data['Customer Status'].astype('category')
data['Churn Label'] = data['Churn Label'].astype('category')
```
:::

::: {.cell .code execution_count="21" execution="{\"iopub.execute_input\":\"2024-06-07T06:53:17.609211Z\",\"iopub.status.busy\":\"2024-06-07T06:53:17.608758Z\",\"iopub.status.idle\":\"2024-06-07T06:53:17.654683Z\",\"shell.execute_reply\":\"2024-06-07T06:53:17.653437Z\",\"shell.execute_reply.started\":\"2024-06-07T06:53:17.609175Z\"}"}
``` python
data.sample()
```

::: {.output .execute_result execution_count="21"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Number of Dependents</th>
      <th>Country</th>
      <th>State</th>
      <th>City</th>
      <th>Zip Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Population</th>
      <th>Quarter</th>
      <th>Referred a Friend</th>
      <th>Number of Referrals</th>
      <th>Tenure in Months</th>
      <th>Offer</th>
      <th>Phone Service</th>
      <th>Avg Monthly Long Distance Charges</th>
      <th>Multiple Lines</th>
      <th>Internet Service</th>
      <th>Internet Type</th>
      <th>Avg Monthly GB Download</th>
      <th>Online Security</th>
      <th>Online Backup</th>
      <th>Device Protection Plan</th>
      <th>Premium Tech Support</th>
      <th>Streaming TV</th>
      <th>Streaming Movies</th>
      <th>Streaming Music</th>
      <th>Unlimited Data</th>
      <th>Contract</th>
      <th>Paperless Billing</th>
      <th>Payment Method</th>
      <th>Monthly Charge</th>
      <th>Total Charges</th>
      <th>Total Refunds</th>
      <th>Total Extra Data Charges</th>
      <th>Total Long Distance Charges</th>
      <th>Total Revenue</th>
      <th>Satisfaction Score</th>
      <th>Customer Status</th>
      <th>Churn Label</th>
      <th>Churn Score</th>
      <th>CLTV</th>
      <th>Churn Category</th>
      <th>Churn Reason</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1626</th>
      <td>9492-TOKRI</td>
      <td>Female</td>
      <td>55</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>United States</td>
      <td>California</td>
      <td>Dinuba</td>
      <td>93618</td>
      <td>36.523619</td>
      <td>-119.386868</td>
      <td>24206</td>
      <td>Q3</td>
      <td>No</td>
      <td>0</td>
      <td>18</td>
      <td>No offer</td>
      <td>Yes</td>
      <td>13.65</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber Optic</td>
      <td>5</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Month-to-Month</td>
      <td>Yes</td>
      <td>Credit Card</td>
      <td>90.0</td>
      <td>1527.35</td>
      <td>0.0</td>
      <td>0</td>
      <td>245.7</td>
      <td>1773.05</td>
      <td>1</td>
      <td>Churned</td>
      <td>Yes</td>
      <td>90</td>
      <td>2708</td>
      <td>Competitor</td>
      <td>Competitor had better devices</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="22"}
``` python
categorical_columns = ['Gender', 'Married', 'Dependents',
                       'Referred a Friend', 'Phone Service', 'Multiple Lines',
                       'Internet Service', 'Online Security', 'Online Backup', 
                       'Device Protection Plan', 'Premium Tech Support', 
                       'Streaming TV', 'Streaming Movies', 'Streaming Music', 
                       'Unlimited Data', 'Paperless Billing', 
                       'Payment Method', 'Customer Status', 'Churn Label']

dummies_gender = pd.get_dummies(data['Gender'], prefix='Gender', drop_first=True)
dummies_married = pd.get_dummies(data['Married'], prefix='Married', drop_first=True)
dummies_dependents = pd.get_dummies(data['Dependents'], prefix='Dependents', drop_first=True)
dummies_referred_friend = pd.get_dummies(data['Referred a Friend'], prefix='Referred a Friend', drop_first=True)
dummies_phone_service = pd.get_dummies(data['Phone Service'], prefix='Phone Service', drop_first=True)
dummies_multiple_lines = pd.get_dummies(data['Multiple Lines'], prefix='Multiple Lines', drop_first=True)
dummies_internet_service = pd.get_dummies(data['Internet Service'], prefix='Internet Service', drop_first=True)
dummies_online_security = pd.get_dummies(data['Online Security'], prefix='Online Security', drop_first=True)
dummies_online_backup = pd.get_dummies(data['Online Backup'], prefix='Online Backup', drop_first=True)
dummies_device_protection_plan = pd.get_dummies(data['Device Protection Plan'], prefix='Device Protection Plan', drop_first=True)
dummies_premium_tech_support = pd.get_dummies(data['Premium Tech Support'], prefix='Premium Tech Support', drop_first=True)
dummies_streaming_tv = pd.get_dummies(data['Streaming TV'], prefix='Streaming TV', drop_first=True)
dummies_streaming_movies = pd.get_dummies(data['Streaming Movies'], prefix='Streaming Movies', drop_first=True)
dummies_streaming_music = pd.get_dummies(data['Streaming Music'], prefix='Streaming Music', drop_first=True)
dummies_unlimited_data = pd.get_dummies(data['Unlimited Data'], prefix='Unlimited Data', drop_first=True)
dummies_paperless_billing = pd.get_dummies(data['Paperless Billing'], prefix='Paperless Billing', drop_first=True)
dummies_payment_method = pd.get_dummies(data['Payment Method'], prefix='Payment Method', drop_first=True)
dummies_customer_status = pd.get_dummies(data['Customer Status'], prefix='Customer Status')
dummies_churn_label = pd.get_dummies(data['Churn Label'], prefix='Churn Label', drop_first=True)

data = pd.concat([data,
                  dummies_gender,
                  dummies_married,
                  dummies_dependents,
                  dummies_referred_friend,
                  dummies_phone_service,
                  dummies_multiple_lines,
                  dummies_internet_service,
                  dummies_online_security,
                  dummies_online_backup,
                  dummies_device_protection_plan,
                  dummies_premium_tech_support,
                  dummies_streaming_tv,
                  dummies_streaming_movies,
                  dummies_streaming_music,
                  dummies_unlimited_data,
                  dummies_paperless_billing,
                  dummies_payment_method,
                  dummies_customer_status,
                  dummies_churn_label,], axis=1)

data.drop(categorical_columns, axis=1, inplace=True)
```
:::

::: {.cell .code execution_count="23"}
``` python
data.sample(5)
```

::: {.output .execute_result execution_count="23"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer ID</th>
      <th>Age</th>
      <th>Number of Dependents</th>
      <th>Country</th>
      <th>State</th>
      <th>City</th>
      <th>Zip Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Population</th>
      <th>Quarter</th>
      <th>Number of Referrals</th>
      <th>Tenure in Months</th>
      <th>Offer</th>
      <th>Avg Monthly Long Distance Charges</th>
      <th>Internet Type</th>
      <th>Avg Monthly GB Download</th>
      <th>Contract</th>
      <th>Monthly Charge</th>
      <th>Total Charges</th>
      <th>Total Refunds</th>
      <th>Total Extra Data Charges</th>
      <th>Total Long Distance Charges</th>
      <th>Total Revenue</th>
      <th>Satisfaction Score</th>
      <th>Churn Score</th>
      <th>CLTV</th>
      <th>Churn Category</th>
      <th>Churn Reason</th>
      <th>Gender_Male</th>
      <th>Married_Yes</th>
      <th>Dependents_Yes</th>
      <th>Referred a Friend_Yes</th>
      <th>Phone Service_Yes</th>
      <th>Multiple Lines_Yes</th>
      <th>Internet Service_Yes</th>
      <th>Online Security_Yes</th>
      <th>Online Backup_Yes</th>
      <th>Device Protection Plan_Yes</th>
      <th>Premium Tech Support_Yes</th>
      <th>Streaming TV_Yes</th>
      <th>Streaming Movies_Yes</th>
      <th>Streaming Music_Yes</th>
      <th>Unlimited Data_Yes</th>
      <th>Paperless Billing_Yes</th>
      <th>Payment Method_Credit Card</th>
      <th>Payment Method_Mailed Check</th>
      <th>Customer Status_Churned</th>
      <th>Customer Status_Joined</th>
      <th>Customer Status_Stayed</th>
      <th>Churn Label_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6758</th>
      <td>0083-PIVIK</td>
      <td>25</td>
      <td>0</td>
      <td>United States</td>
      <td>California</td>
      <td>Tulelake</td>
      <td>96134</td>
      <td>41.813521</td>
      <td>-121.492666</td>
      <td>2595</td>
      <td>Q3</td>
      <td>0</td>
      <td>64</td>
      <td>No offer</td>
      <td>5.49</td>
      <td>DSL</td>
      <td>69</td>
      <td>One Year</td>
      <td>81.25</td>
      <td>5567.55</td>
      <td>0.0</td>
      <td>40</td>
      <td>351.36</td>
      <td>5958.91</td>
      <td>3</td>
      <td>67</td>
      <td>4761</td>
      <td>Not Churned</td>
      <td>Not Churned</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>93</th>
      <td>3254-YRILK</td>
      <td>74</td>
      <td>0</td>
      <td>United States</td>
      <td>California</td>
      <td>Dobbins</td>
      <td>95935</td>
      <td>39.381174</td>
      <td>-121.211910</td>
      <td>614</td>
      <td>Q3</td>
      <td>0</td>
      <td>19</td>
      <td>Offer D</td>
      <td>39.97</td>
      <td>Fiber Optic</td>
      <td>9</td>
      <td>Month-to-Month</td>
      <td>88.20</td>
      <td>1775.80</td>
      <td>0.0</td>
      <td>0</td>
      <td>759.43</td>
      <td>2535.23</td>
      <td>1</td>
      <td>69</td>
      <td>5683</td>
      <td>Competitor</td>
      <td>Competitor had better devices</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1920</th>
      <td>8945-GRKHX</td>
      <td>41</td>
      <td>0</td>
      <td>United States</td>
      <td>California</td>
      <td>Santa Barbara</td>
      <td>93110</td>
      <td>34.437945</td>
      <td>-119.771910</td>
      <td>15757</td>
      <td>Q3</td>
      <td>0</td>
      <td>1</td>
      <td>Offer E</td>
      <td>46.09</td>
      <td>Fiber Optic</td>
      <td>10</td>
      <td>Month-to-Month</td>
      <td>78.65</td>
      <td>78.65</td>
      <td>0.0</td>
      <td>0</td>
      <td>46.09</td>
      <td>124.74</td>
      <td>1</td>
      <td>81</td>
      <td>2728</td>
      <td>Competitor</td>
      <td>Competitor had better devices</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2124</th>
      <td>1502-XFCVR</td>
      <td>57</td>
      <td>0</td>
      <td>United States</td>
      <td>California</td>
      <td>Los Angeles</td>
      <td>90038</td>
      <td>34.088017</td>
      <td>-118.327168</td>
      <td>32562</td>
      <td>Q3</td>
      <td>0</td>
      <td>58</td>
      <td>Offer B</td>
      <td>14.99</td>
      <td>Fiber Optic</td>
      <td>12</td>
      <td>One Year</td>
      <td>106.45</td>
      <td>6145.85</td>
      <td>0.0</td>
      <td>0</td>
      <td>869.42</td>
      <td>7015.27</td>
      <td>1</td>
      <td>82</td>
      <td>5902</td>
      <td>Competitor</td>
      <td>Competitor offered higher download speeds</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3055</th>
      <td>3174-AKMAS</td>
      <td>61</td>
      <td>0</td>
      <td>United States</td>
      <td>California</td>
      <td>Hume</td>
      <td>93628</td>
      <td>36.807595</td>
      <td>-118.901544</td>
      <td>93</td>
      <td>Q3</td>
      <td>1</td>
      <td>46</td>
      <td>No offer</td>
      <td>37.02</td>
      <td>DSL</td>
      <td>6</td>
      <td>Two Year</td>
      <td>64.20</td>
      <td>3009.50</td>
      <td>0.0</td>
      <td>0</td>
      <td>1702.92</td>
      <td>4712.42</td>
      <td>4</td>
      <td>41</td>
      <td>4699</td>
      <td>Not Churned</td>
      <td>Not Churned</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
## Visualizing key features
:::

::: {.cell .code execution_count="24"}
``` python
churnChart = sns.countplot(x='Churn Label_Yes', data=data)
churnChart.set_xticklabels(['Not Churned', 'Churned'])
churnChart.set_title('Customer Churn Distribution')
churnChart.set_xlabel('Customer Churned')
plt.show()
```

::: {.output .stream .stderr}
    C:\Users\SWASTIK\AppData\Local\Temp\ipykernel_14356\2403011004.py:2: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
      churnChart.set_xticklabels(['Not Churned', 'Churned'])
:::

::: {.output .display_data}
![](vertopal_664b24753eb5447396a6449d256d5785/f64ce46e5267d069314f4bb0b569ab30b321d657.png)
:::
:::

::: {.cell .markdown}
**Findings:** The chart shows that about one-fourth of the customers
have churned.`<br>`{=html} **Key Insight:** The churn rate is
significant, with 25% of customers leaving the service. This indicates a
potential issue with customer retention.
:::

::: {.cell .code execution_count="25"}
``` python
plt.figure(figsize=(15,6))
ageChart = sns.countplot(x='Age', data=data, hue='Churn Label_Yes')
ageChart.set_title('Age Distribution')
plt.legend(title='Legend', labels=['Not Churned', 'Churned'])
plt.show()
```

::: {.output .display_data}
![](vertopal_664b24753eb5447396a6449d256d5785/a3c9d13d6a169c4994808716b52aa5edcc7294d1.png)
:::
:::

::: {.cell .code execution_count="26"}
``` python
data['Age Group'] = pd.cut(data['Age'], bins=[0, 40, 65, 100], labels=['Under 40', '40-64', '65+'])
plt.figure(figsize=(8, 6))
sns.countplot(x='Age Group', hue='Churn Label_Yes', data=data)
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Age Group vs Churn')
plt.legend(title='Legend', labels=['Not Churned', 'Churned'])
plt.show()
data.drop(['Age Group'], axis=1, inplace=True)
```

::: {.output .display_data}
![](vertopal_664b24753eb5447396a6449d256d5785/b63bb3bf5a8f0cbb910ac756af7eb685337910f6.png)
:::
:::

::: {.cell .markdown}
**Findings:** Most customers are aged between 19 to 64. Across all age
groups, the number of non-churning customers is about three times the
number of churned customers. However, churn rate increases with
age.`<br>`{=html} **Key Insight:** Older customers tend to churn more
frequently. This could imply that services might not be meeting the
needs of older demographics effectively.
:::

::: {.cell .code execution_count="27"}
``` python
genderChart = sns.countplot(x='Gender_Male', hue='Churn Label_Yes', data=data)
genderChart.set_title('Gender wise churn distribution')
genderChart.set_xlabel('Gender')
genderChart.set_xticklabels(['Female', 'Male'])
genderChart.legend(title='Legend', labels=['Not Churned', 'Churned'])
plt.show()
```

::: {.output .stream .stderr}
    C:\Users\SWASTIK\AppData\Local\Temp\ipykernel_14356\483836313.py:4: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
      genderChart.set_xticklabels(['Female', 'Male'])
:::

::: {.output .display_data}
![](vertopal_664b24753eb5447396a6449d256d5785/fd68d2f4428fb3338fa16db883dd50f08d1ae917.png)
:::
:::

::: {.cell .markdown}
**Findings:** The distribution is almost equal between male and female
customers. In both cases, the majority are non-churners.`<br>`{=html}
**Key Insight:** Gender does not significantly influence churn rates as
both males and females exhibit similar behaviors regarding churn.
:::

::: {.cell .code execution_count="28"}
``` python
pairChart = sns.pairplot(data, hue='Churn Label_Yes', vars=['Tenure in Months', 'Monthly Charge', 'Total Charges', 'Total Refunds'])
pairChart.fig.suptitle('Tenure in Months vs Monthly Charge vs Total Charges vs Total Refunds', y=1.02)
plt.show()
```

::: {.output .display_data}
![](vertopal_664b24753eb5447396a6449d256d5785/6d14c238ece39820089b1ce69087a35613a800cb.png)
:::
:::

::: {.cell .markdown}
**Findings**:

1.  **Tenure in Months vs. Monthly Charge**:
    -   Higher churn is observed among customers with shorter tenures.
    -   Longer-tenured customers show lower churn rates.
2.  **Tenure in Months vs. Total Charges**:
    -   Positive correlation between tenure and total charges.
    -   Higher churn for customers with shorter tenures and lower total
        charges.
3.  **Tenure in Months vs. Total Refunds**:
    -   No strong relationship between tenure and total refunds for
        churned customers.
    -   Slight concentration of churned customers at low refund values.
4.  **Monthly Charge vs. Total Charges**:
    -   Positive correlation between monthly charges and total charges.
    -   Churned customers more prevalent at higher monthly charges.
5.  **Monthly Charge vs. Total Refunds**:
    -   No clear pattern between monthly charges and total refunds for
        churned customers.
6.  **Total Charges vs. Total Refunds**:
    -   No strong correlation between total charges and total refunds.
    -   Slight concentration of churned customers with lower total
        refunds.
7.  **Univariate Distributions**:
    -   **Tenure in Months**: Higher churn density at lower tenures.
    -   **Monthly Charge**: Higher churn at elevated monthly charges.
    -   **Total Charges**: Slightly higher churn density at lower total
        charges.
    -   **Total Refunds**: Most customers have low refunds, with churn
        spread across.

**Key Insights**:

1.  **Short Tenure**: Strong indicator of churn. Early engagement is
    crucial.
2.  **High Monthly Charges**: Higher churn suggests the need for
    reviewing pricing strategies.
3.  **Total Refunds**: Not a significant churn factor but still
    important for customer trust.
:::

::: {.cell .code execution_count="29"}
``` python
sns.countplot(data=data, x='Internet Type', hue='Churn Label_Yes')
plt.title('Internet Type Distribution')
plt.show()
```

::: {.output .display_data}
![](vertopal_664b24753eb5447396a6449d256d5785/d006b5140635017694e93e3ca83a73226409675f.png)
:::
:::

::: {.cell .markdown}
**Findings:** Most users have fiber optic internet, but these users are
most likely to churn. DSL users tend to churn less.`<br>`{=html} **Key
Insight:** Fiber optic users are more likely to churn, suggesting
possible issues with the service or competition from other providers.
:::

::: {.cell .code execution_count="30"}
``` python
fig, axes = plt.subplots(1,3, figsize=(12,4))

sns.countplot(ax=axes[0], data=data, x='Phone Service_Yes', hue='Churn Label_Yes')
axes[0].set_title('Phone Service Distribution')
sns.countplot(ax=axes[1], data=data, x='Internet Service_Yes', hue='Churn Label_Yes')
axes[1].set_title('Internet Service Distribution')
sns.countplot(ax=axes[2], data=data, x='Contract', hue='Churn Label_Yes')
axes[2].set_title('Contract Type Distribution')
plt.show()
```

::: {.output .display_data}
![](vertopal_664b24753eb5447396a6449d256d5785/2af36080afd97bd0612c7cdb4374a4d0c56decdb.png)
:::
:::

::: {.cell .markdown}
**Phone Service:**`<br>`{=html}  **Findings:** Most customers have phone
service. The churn rate is higher among those who do not have phone
service.`<br>`{=html} **Internet Service:**`<br>`{=html}  **Findings:**
Similar to phone service, most customers have internet service, and the
churn rate is higher among those who do not have internet
service.`<br>`{=html} **Contract Type:**`<br>`{=html}  **Findings:**
Customers with month-to-month contracts are more likely to churn
compared to those with one or two-year contracts.`<br>`{=html} **Key
Insight:** Longer contracts tend to reduce churn, indicating a need for
strategies to encourage long-term commitments.
:::

::: {.cell .code execution_count="31"}
``` python
fig, axes = plt.subplots(1,2, figsize=(8,4))
sns.scatterplot(ax=axes[0], x='Tenure in Months', y='Satisfaction Score', data=data, hue='Churn Label_Yes')
sns.countplot(ax=axes[1], x='Referred a Friend_Yes', data=data, hue='Churn Label_Yes')
plt.show()
```

::: {.output .display_data}
![](vertopal_664b24753eb5447396a6449d256d5785/545dd5a2cde688c5b74af69755c042612b66821b.png)
:::
:::

::: {.cell .markdown}
**Findings:** There is no clear trend between tenure and satisfaction
score.`<br>`{=html} **Key Insight:** Satisfaction scores do not
necessarily improve with tenure, indicating that other factors might be
influencing customer satisfaction.`<br>`{=html} **Findings:** Customers
who have referred friends are less likely to churn.`<br>`{=html} **Key
Insight:** Referrals can be a strong indicator of customer satisfaction
and loyalty.
:::

::: {.cell .code execution_count="32"}
``` python
sns.countplot(x='Churn Category', data=data, order=data['Churn Category'].value_counts().index)
plt.show()
```

::: {.output .display_data}
![](vertopal_664b24753eb5447396a6449d256d5785/44b425e503a7e0be04977eb60dd24aace39b9ac8.png)
:::
:::

::: {.cell .markdown}
**Findings:** Various reasons for churn are visualized, with some
categories showing higher churn rates.`<br>`{=html} **Key Insight:**
Understanding specific churn categories can help in addressing targeted
issues.
:::

::: {.cell .code execution_count="33"}
``` python
categoryReasonCrosstab = pd.crosstab(data[data['Churn Category'] != 'Not Churned']['Churn Category'], data['Churn Reason'])
categoryReasonCrosstab.plot(kind='bar', stacked=True, figsize=(12,8), colormap='tab20')
plt.xlabel('Churn Category')
plt.ylabel('Count')
plt.title('Churn Reasons by Churn Category')
plt.legend(title='Churn Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()
```

::: {.output .display_data}
![](vertopal_664b24753eb5447396a6449d256d5785/9f400a789726f9d7b9d267ac5f5d1ae75eac8f4e.png)
:::
:::

::: {.cell .markdown}
**Findings:** A significant number of customers cited competition as a
primary reason for churning. Many customers left because competitors
provided superior devices and more attractive offers. Some customers
churned because competitors offered better data packages. Another major
reason for churn is related to customer support. Customers reported
dissatisfaction with the attitude of support staff as a major factor in
their decision to leave.`<br>`{=html} **Key Insight:** The high churn
rate due to competitors indicates that the company needs to enhance its
device offerings, create more attractive offers, and provide better data
packages to stay competitive. The significant number of customers citing
poor support staff attitude suggests that improving customer service
training and monitoring could reduce churn rates.
:::

::: {.cell .code execution_count="34"}
``` python
sns.countplot(data=data, x='Premium Tech Support_Yes', hue='Churn Label_Yes')

plt.show()
```

::: {.output .display_data}
![](vertopal_664b24753eb5447396a6449d256d5785/a0b33f0f6cf50ee83189b94a74b691197d7b68a3.png)
:::
:::

::: {.cell .markdown}
**Findings:** Customer churn is more prevalent among those without
premium tech support. The churn rate is considerably lower among those
with tech support.`<br>`{=html} **Key Insight:** Offering premium tech
support can significantly reduce churn rates.
:::

::: {.cell .markdown}
```{=html}
<hr>
```
:::

::: {.cell .markdown}
### Summary:

This analysis revealed that customer churn is predominantly influenced
by factors such as tenure length, monthly charges, and customer support
experiences. Short-tenured customers and those with higher monthly
charges are more prone to churn. Additionally, competitors offering
better devices, more data, and superior customer service were
significant reasons for customer churn.
:::

::: {.cell .markdown}
### Recommendations:

1.  **Early Engagement**:
    -   Enhance early-stage customer support and engagement efforts to
        reduce churn among new customers.
2.  **Pricing Strategy**:
    -   Review and potentially revise pricing strategies to ensure
        high-cost plans deliver sufficient value.
3.  **Customer Support**:
    -   Invest in training customer support personnel to improve their
        interaction with customers, addressing the dissatisfaction
        leading to churn.
4.  **Competitor Analysis**:
    -   Continuously monitor competitors\' offerings and adapt to meet
        or exceed them in terms of value, technology, and customer
        service.
:::

::: {.cell .markdown}
### Conclusion:

Understanding the reasons behind customer churn allows businesses to
implement targeted strategies to improve retention. This analysis
provides a comprehensive view of the factors influencing churn in the
telecommunications industry. By focusing on early engagement, pricing
strategies, and customer support, companies can significantly reduce
churn rates and enhance customer satisfaction, ultimately leading to
sustained business growth.
:::
