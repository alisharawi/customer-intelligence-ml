# **Steps and notes for customer intelligence ML project**



### **## Phase 1 :**

1- print the data (read/info/describe/head)

2- see which columns/data we dont need and we drop them 

i dropped the data where customer id is NA and removed the cancellations also dropped the duplicates (around 200 thousand no id transactions and around 19000 cancelled transactions and around 6000 duplicated transactions were removed)

3- made a new column which we need (total price) for better info

4- wanted to see the top customers 

5- made a chart for monthly sales to see the sales over the year

6- made a chart to see the country sales for the data

7- made a chart showing top 10 products sales

8- made a new column called customer dates it shows every how many days each customer buys on average (made a histogram about it) and got about every 2-3 weeks



### **## Phase 2 :**

file name: src/data\_preprocessing \& src/feature\_engineering

1- i made the pipeline that is reusable and that we can use it when we get new data and it makes a table that gives analysis about the data 

2- made a file that has a function where it cleans the data (removes cancelled \& NULL customer ID transactions and duplicates)

3- added a total price column

4- made a file that splits the data into 2 parts (before last 30days and the last 30 days to make predictions)

5- made a new DF that has the features we want grouped by customer id so every customer and his info in one row 



### **## Phase 3:**

filename : src/train.py

1- choose the features 

&#x09;\[ 'Recency', 'Frequency', 'Monetary\_Value', 

&#x20;       'Total\_Purchased\_Quantity', 'Unique\_Products\_Purchased', 'Average\_Order\_Value']

&#x09;recency: days since last purchase

&#x09;frequency: historical purchasing frequency

&#x09;monetary value: historical customer spending



2- identified X and y

3- made the train\_test\_split 20% test data

4- initialized RFC\_model

5- fitted the model

6- made predictions

7- analyzed results

8- restructured all of the file to make 2 functions (train\_test\_split/precession\_at\_k)

9- compared all models against each other and choose the best 

10- used precision score / confusion matrix / recall / f1 / roc auc / average precision to choose the best out of them all 

11- choose logistic regression as the best among them all based on : 

&#x09;good precision / best recall / best f1 score / best roc-auc / best pr-auc / and second on precision@100



### **## Phase 4:**

file name: src/clustering.py

1- loaded the data using the pipelines i already did

2- created a DF that only has RFM features \['Recency', 'Frequency', 'Monetary\_Value'] 

3- handled the outliers using log equation that wont ignores whole buyers

4- scaled the features so they have the same weight when predicting and grouping/clustering

5- made sure to make 4 different clustering groups for customers

6- made a new column for the DF called cluster to assign for each customer its group

7- made a summary that groups the data based on cluster group and to identify the champion buyers (group 2) which had the average of 5000 dollars within 12 days

8- plotted the summary data we made to a visual data



### **## Phase 5:**

file name: src/save\_model.py \& api/main.py \& api/test\_main.py

1- made the pickle file so it uses always logistic regression and loads the model faster

2- mad the api/main.py file which initiates the interactive web server that anyone can use to predict the next purchase of any customer by entering specific data of the customer

&#x09;to boot the api and test it : ***\*\*  uvicorn api.main:app --reload  \*\****

3- made a file that tests the health and the prediction of the web server so it always makes sure that the model/api is working as it should 

&#x09;to run the test on the api and the model: ***\*\* pytest api/test\_main.py -v \*\****



### **## Phase 6:**

file name: dashboard/app.py

to run the web app: ***\*\* streamlit run dashboard/app.py \*\****

made the web application that has many sections:

1- business analytics on the data we have provided 

2- customer segmentation: shows a plot of our types of buyers that we have and their RFM

3- prediction simulator: the actual model where we can input data so it predicts for us if the customer will buy in the next 30 days or not

4- model evaluation: it plots which features does the model (logistic regression) rely on when predicting if the customer will buy or not











&#x09;







