#!/usr/bin/env python
# coding: utf-8

# # Importing the packages

# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
import plotly.express as px 
import plotly.graph_objects as go 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor


# In[3]:


data = pd. read_csv (r'//Users/amirmac/Desktop/Capstone_dataset.csv')
data.head()


# In[4]:


data.shape


# In[5]:


data.info() 


# In[6]:


data.columns


# In[7]:


(data.isna().sum() / len(data)) * 100


# In[8]:


data.describe()


# In[9]:


import pandas as pd

# Assuming you have a DataFrame named df with a 'hash' column containing hexadecimal values

# Define a function to convert a hash rate value from hex to EH/s
def convert_to_ehs(hex_value):
    hash_rate_hps = int(hex_value, 16)
    hash_rate_ehs = hash_rate_hps / (10**18)  # Convert to exahashes per second (EH/s)
    return hash_rate_ehs

# Apply the conversion function to the entire 'hash' column and store results in a new column
data['hash_rate_ehs'] = data['hash'].apply(convert_to_ehs)

# Print the updated DataFrame with the converted hash rates
for index, row in data.iterrows():
    hash_rate_ehs = row['hash_rate_ehs']
    print(f"Hash Rate: {hash_rate_ehs:.6f} EH/s")


# In[10]:


data.head()


# # Exploratory Data Analysis

# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named df with 'hash_rate_ehs' column
hash_rate_ehs = data['hash_rate_ehs']

# Distribution plot using Seaborn
plt.figure(figsize=(8, 6))
sns.histplot(hash_rate_ehs, bins=30, kde=True)
plt.title('Distribution of Hash Rate (EH/s)')
plt.xlabel('Hash Rate (EH/s)')
plt.ylabel('Frequency')
plt.show()


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))

# Create the countplot
ax = sns.countplot(data=data, x='pool_name', palette='afmhot')

# Display counts on top of the bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='baseline', fontsize=10, fontweight='bold')

plt.title('Count of Pool Names')
plt.show()


# In[13]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named df with 'pool_name' and 'hash_rate_ehs' columns

# Set a custom Seaborn style
sns.set_style("whitegrid")

# Create a bar plot using Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='pool_name', y='hash_rate_ehs', data=data, ci=None, palette="RdBu_r")
plt.title('Hash Rate (EH/s) by Pool Name')
plt.xlabel('Pool Name')
plt.ylabel('Hash Rate (EH/s)')
plt.xticks(rotation=45, ha='right')  # Rotate and align x-axis labels
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named df with 'pool_name' and 'reward_fees' columns

# Calculate the average reward fees for each pool
avg_reward_fees_by_pool = data.groupby('pool_name')['reward_fees'].mean()

# Convert reward fees to millions
avg_reward_fees_by_pool_millions = avg_reward_fees_by_pool / 1e6

# Get unique pool names and assign colors
unique_pools = data['pool_name'].unique()
color_map = plt.cm.get_cmap('tab20', len(unique_pools))

# Create a bar chart with separate colors for each pool
plt.figure(figsize=(12, 6))
bars = plt.bar(avg_reward_fees_by_pool_millions.index, avg_reward_fees_by_pool_millions.values, color=[color_map(i) for i in range(len(unique_pools))])

# Remove grid lines
plt.grid(False)

# Add data labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{round(yval, 2)}M', ha='center', va='bottom', color='black')

plt.title('Average Reward Fees by Pool Name')
plt.xlabel('Pool Name')
plt.ylabel('Average Reward Fees (Millions)')
plt.xticks(rotation=45, ha='right')  # Rotate and align x-axis labels
plt.tight_layout()  # Adjust layout to prevent label cutoff

plt.show()



# In[15]:


data.head()


# In[16]:


column_names_to_drop = ['difficulty','reward_block','bits', 'difficulty_double', 'is_orphan', 'is_sw_block', 'hashrate EH/s']
data.drop(columns=column_names_to_drop, inplace=True)


# In[17]:


plt.figure(figsize=(15,8))
sns.heatmap(data.corr(), annot=True, cmap='Oranges')


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have a DataFrame named df with 'pool_name' and 'sigops' columns

# Calculate the average sigops for each pool
avg_sigops_by_pool = data.groupby('pool_name')['sigops'].mean()

# Get unique pool names and assign colors
unique_pools = data['pool_name'].unique()
color_map = plt.cm.get_cmap('tab20', len(unique_pools))

# Create an array for the x-axis positions
x = np.arange(len(unique_pools))

# Create a figure and axis
plt.figure(figsize=(12, 6))
ax = plt.gca()

# Plot the average sigops for each pool with thicker bars
bars = ax.bar(x, avg_sigops_by_pool.values, width=0.6, color=[color_map(i) for i in range(len(unique_pools))])

# Remove grid lines
ax.grid(False)

# Add data labels on top of bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{round(yval, 2)}', ha='center', va='bottom', color='black')

ax.set_title('Average Sigops by Pool Name')
ax.set_xlabel('Pool Name')
ax.set_ylabel('Average Sigops')
ax.set_xticks(x)
ax.set_xticklabels(avg_sigops_by_pool.index, rotation=45, ha='right')
ax.set_ylim(0, max(avg_sigops_by_pool.values) * 1.1)  # Adjust y-axis limit for better visualization
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', which='both', length=0)  # Hide tick marks on x-axis
plt.tight_layout()

plt.show()


# # Model Training

# In[19]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Select relevant columns for the regression
selected_columns = ["height", "timestamp", "size","reward_fees","pool_difficulty","tx_count","reward_fees"]
X = data[selected_columns]
y = data["hash_rate_ehs"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict using the test set
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


# # Predictive Modeling


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)
linear_r2 = r2_score(y_test, linear_pred)
print(f"Linear Regression R-squared: {linear_r2:.4f}")

# Initialize and train the Decision Tree Regressor model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
tree_r2 = r2_score(y_test, tree_pred)
print(f"Decision Tree Regression R-squared: {tree_r2:.4f}")

# Initialize and train the Random Forest Regressor model
forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(X_train, y_train)
forest_pred = forest_model.predict(X_test)
forest_r2 = r2_score(y_test, forest_pred)
print(f"Random Forest Regression R-squared: {forest_r2:.4f}")
# # Linear Regression

# In[21]:


import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# Define selected columns and target variable
selected_columns = ["height", "timestamp", "size", "reward_fees","pool_difficulty","tx_count","reward_fees"]
target_variable = "hash_rate_ehs"

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data[selected_columns])
test_data_scaled = scaler.transform(test_data[selected_columns])

# Add a constant term to the features
X_train = sm.add_constant(train_data_scaled)

# Perform linear regression using StatsModels
model = sm.OLS(train_data[target_variable], X_train)
results = model.fit()

# Create a summary table for coefficients
coef_summary = pd.DataFrame({
    'Coefficient': results.params,
    'Standard Error': results.bse,
    't-value': results.tvalues,
    'p-value': results.pvalues,
})

# Display the coefficient summary table
print("Linear Regression Coefficients Summary:")
print(coef_summary)

# Predict on the test data
X_test = sm.add_constant(test_data_scaled)
linear_reg_predictions = results.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(test_data[target_variable], linear_reg_predictions)

# Create a summary table for predictions
pred_summary = pd.DataFrame({
    'Predicted': linear_reg_predictions,
    'Actual': test_data[target_variable],
})

# Display the prediction summary table
print("\nLinear Regression Predictions Summary:")
print(pred_summary)
print("\nMean Squared Error:", mse)


# In[22]:


import statsmodels.api as sm
import pandas as pd


selected_columns = ["height", "timestamp", "size", "reward_fees","pool_difficulty","tx_count","reward_fees"]
target_variable = "hash_rate_ehs"

X = data[selected_columns]
y = data[target_variable]

# Add constant term to the independent variables
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())



# In[23]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

linear_reg_model = LinearRegression()

# Fit the linear regression model to training data
linear_reg_model.fit(train_data_scaled, train_data[target_variable])

# Predict target values using the linear regression model
predicted_train = linear_reg_model.predict(train_data_scaled)
predicted_test = linear_reg_model.predict(test_data_scaled)

# Create scatter plot for training data
plt.figure(figsize=(10, 6))
plt.scatter(train_data[target_variable], predicted_train, color='blue', label='Training Data')
plt.xlabel('Actual ' + target_variable)
plt.ylabel('Predicted ' + target_variable)
plt.title('Linear Regression - Training Data')
plt.legend()
plt.grid(True)
plt.show()

# Create scatter plot for testing data
plt.figure(figsize=(10, 6))
plt.scatter(test_data[target_variable], predicted_test, color='green', label='Testing Data')
plt.xlabel('Actual ' + target_variable)
plt.ylabel('Predicted ' + target_variable)
plt.title('Linear Regression - Testing Data')
plt.legend()
plt.grid(True)
plt.show()


# # Decision Tree

# In[24]:


train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Initialize the Decision Tree Regressor model with limited maximum depth
max_depth = 1  # You can adjust this value to limit the depth
tree_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)

# Fit the model to the training data
tree_model.fit(train_data[selected_columns], train_data[target_variable])

# Predict target values using the decision tree model
predicted_train = tree_model.predict(train_data[selected_columns])
predicted_test = tree_model.predict(test_data[selected_columns])

# Calculate the Mean Squared Error
mse_train = mean_squared_error(train_data[target_variable], predicted_train)
mse_test = mean_squared_error(test_data[target_variable], predicted_test)

print("Decision Tree Regression Mean Squared Error (Train):", mse_train)
print("Decision Tree Regression Mean Squared Error (Test):", mse_test)

# Visualize the reduced-depth decision tree (requires Graphviz)
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 15))
plot_tree(tree_model, feature_names=selected_columns, filled=True, rounded=True)
plt.show()



# In[25]:


dt_reg_model = DecisionTreeRegressor()
dt_reg_model.fit(train_data_scaled, train_data[target_variable])
dt_reg_score = dt_reg_model.score(test_data_scaled, test_data[target_variable])

dt_reg_predictions = dt_reg_model.predict(test_data_scaled)


# Create a summary table
dt_summary_table = pd.DataFrame({
    'Predicted': dt_reg_predictions,
    'Actual': test_data[target_variable],
})

# Display the summary table
print("Decision Tree Regression Summary:")
print(dt_summary_table)
print("\nMean Squared Error:", mse)



# # Random Forest

# In[26]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Initialize the RandomForestRegressor model
rf_reg_model = RandomForestRegressor()

# Fit the model to the training data
rf_reg_model.fit(train_data[selected_columns], train_data[target_variable])

# Visualize feature importances
input_features = selected_columns
feature_importances = rf_reg_model.feature_importances_
sorted_idx = np.argsort(feature_importances)

plt.barh(range(len(input_features)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(input_features)), [input_features[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()


# In[27]:


rf_reg_predictions = rf_reg_model.predict(test_data_scaled)

mse = mean_squared_error(test_data[target_variable], rf_reg_predictions)

# Create a summary table for predictions
rf_summary_table = pd.DataFrame({
    'Predicted': rf_reg_predictions,
    'Actual': test_data[target_variable],
})

# Display the summary table
print("RandomForest Regression Predictions Summary:")
print(rf_summary_table)
print("\nMean Squared Error:", mse)


# # Feature Importance

# In[28]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Select columns and target variable
selected_columns = ["height", "timestamp", "size", "reward_fees","pool_difficulty","tx_count","reward_fees"]
target_variable = "hash_rate_ehs"

# Models
tree_model = DecisionTreeRegressor()
forest_model = RandomForestRegressor()

# Fit models
tree_model.fit(train_data[selected_columns], train_data[target_variable])
forest_model.fit(train_data[selected_columns], train_data[target_variable])

# Feature importance
tree_feature_importance = tree_model.feature_importances_
forest_feature_importance = forest_model.feature_importances_

# Create a DataFrame to show feature importance
feature_importance_df = pd.DataFrame({
    'Feature': selected_columns,
    'Decision Tree Importance': tree_feature_importance,
    'Random Forest Importance': forest_feature_importance
})

# Sort by importance in Random Forest
feature_importance_df = feature_importance_df.sort_values(by='Random Forest Importance', ascending=False)

print("Feature Importance:")
print(feature_importance_df)


# # Model Comparison

# In[30]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Select columns and target variable
selected_columns = ["height", "timestamp", "size", "reward_fees","pool_difficulty","tx_count","reward_fees"]
target_variable = "hash_rate_ehs"

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(train_data[selected_columns], train_data[target_variable])
linear_predictions = linear_model.predict(test_data[selected_columns])
linear_mse = mean_squared_error(test_data[target_variable], linear_predictions)
linear_mae = mean_absolute_error(test_data[target_variable], linear_predictions)
linear_r2 = r2_score(test_data[target_variable], linear_predictions)

# Decision Tree Regression
tree_model = DecisionTreeRegressor()
tree_model.fit(train_data[selected_columns], train_data[target_variable])
tree_predictions = tree_model.predict(test_data[selected_columns])
tree_mse = mean_squared_error(test_data[target_variable], tree_predictions)
tree_mae = mean_absolute_error(test_data[target_variable], tree_predictions)
tree_r2 = r2_score(test_data[target_variable], tree_predictions)

# Random Forest Regression
forest_model = RandomForestRegressor()
forest_model.fit(train_data[selected_columns], train_data[target_variable])
forest_predictions = forest_model.predict(test_data[selected_columns])
forest_mse = mean_squared_error(test_data[target_variable], forest_predictions)
forest_mae = mean_absolute_error(test_data[target_variable], forest_predictions)
forest_r2 = r2_score(test_data[target_variable], forest_predictions)

# Print MSE, MAE, and R2 for each model
print("Linear Regression:")
print("  MSE:", linear_mse)
print("  MAE:", linear_mae)
print("  R-squared:", linear_r2)
print("\nDecision Tree Regression:")
print("  MSE:", tree_mse)
print("  MAE:", tree_mae)
print("  R-squared:", tree_r2)
print("\nRandom Forest Regression:")
print("  MSE:", forest_mse)
print("  MAE:", forest_mae)
print("  R-squared:", forest_r2)

# Comparison based on MSE
mse_results = {
    "Linear Regression": linear_mse,
    "Decision Tree Regression": tree_mse,
    "Random Forest Regression": forest_mse
}

best_model_mse = min(mse_results, key=mse_results.get)
print("\nModel with the lowest MSE:", best_model_mse)

# Comparison based on R2
r2_results = {
    "Linear Regression": linear_r2,
    "Decision Tree Regression": tree_r2,
    "Random Forest Regression": forest_r2
}

best_model_r2 = max(r2_results, key=r2_results.get)
print("Model with the highest R-squared:", best_model_r2)

