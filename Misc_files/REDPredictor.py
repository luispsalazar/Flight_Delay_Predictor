import pandas as pd
import pickle

# Load data
clean = pd.read_csv('REDdelay_clean.csv')
clean.head()


# Define features set
X = clean.copy()
X = X.drop("DEP_DEL15", axis=1)
X.head()


# Define target
y = clean["DEP_DEL15"].values


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Splitting into Train and Test sets (70% - 30% respectively)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 20)


# In[6]:


# Creating StandardScaler instance
scaler = StandardScaler()


# In[7]:


# Fitting Standard Scaler
X_scaler = scaler.fit(X_train)


# In[8]:


# Scaling data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# ## Choose best learning rate

# In[9]:


from sklearn.ensemble import GradientBoostingClassifier


# In[10]:


# Create a classifier object
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    classifier = GradientBoostingClassifier(n_estimators = 20,
                                            learning_rate = learning_rate,
                                            max_features = 5,
                                            max_depth = 3,
                                            random_state = 20)

    # Fit the model
    classifier.fit(X_train_scaled, y_train)
    print("Learning rate: ", learning_rate)

    # Score the model
    print("Accuracy score (training): {0:.3f}".format(
        classifier.score(
            X_train_scaled,
            y_train)))
    print("Accuracy score (validation): {0:.3f}".format(
        classifier.score(
            X_test_scaled,
            y_test)))
    print()


# ## Creating Gradient Boosting Classifier using best learning rate = 0.75

# In[11]:


# Choose a learning rate and create classifier
classifier = GradientBoostingClassifier(n_estimators = 20,
                                        learning_rate = 0.75,
                                        max_features = 5,
                                        max_depth = 3,
                                        random_state = 20)


# In[12]:


# Fit the model
classifier.fit(X_train_scaled, y_train)


# In[13]:


# Make Prediction
predictions = classifier.predict(X_test_scaled)
pd.DataFrame({"Prediction": predictions, "Actual": y_test}).sample(20)


# ## Model evaluation

# In[14]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[15]:


# Calculating the accuracy score
acc_score = accuracy_score(y_test, predictions)
print(f"Accuracy Score : {acc_score}")


# In[16]:


# Generate the confusion matrix
cm = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(
    cm, index = ["Actual 0", "Actual 1"],
    columns = ["Predicted 0", "Predicted 1"]
)

# Displaying results
display(cm_df)


# In[17]:


# Generate classification report
print("Classification Report")
print(classification_report(y_test, predictions))


# ## Random Oversampling

# In[18]:


from imblearn.over_sampling import RandomOverSampler
from collections import Counter


# In[19]:


ros = RandomOverSampler(random_state = 20)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

Counter(y_resampled)


# In[20]:


# Splitting into Train and Test sets (70% - 30% respectively)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
                                                    test_size = 0.3, random_state = 20)


# In[21]:


# Fitting Standard Scaler
X_scaler = scaler.fit(X_train)


# In[22]:


# Scaling data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# In[23]:


# Create a classifier object
# learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25]
# for learning_rate in learning_rates:
#     classifier = GradientBoostingClassifier(n_estimators = 30,
#                                             learning_rate = learning_rate,
#                                             max_features = 15,
#                                             max_depth = 5,
#                                             random_state = 20)

#     # Fit the model
#     classifier.fit(X_train_scaled, y_train)
#     print("Learning rate: ", learning_rate)

#     # Score the model
#     print("Accuracy score (training): {0:.3f}".format(
#         classifier.score(
#             X_train_scaled,
#             y_train)))
#     print("Accuracy score (validation): {0:.3f}".format(
#         classifier.score(
#             X_test_scaled,
#             y_test)))
#     print()


# In[24]:


# Choose a learning rate and create classifier
# classifier = GradientBoostingClassifier(n_estimators = 30,
#                                         learning_rate = 1,
#                                         max_features = 15,
#                                         max_depth = 5,
#                                         random_state = 20)


# In[25]:


# Fit the model
classifier.fit(X_train_scaled, y_train)


# In[26]:


# Make Prediction
# predictions = classifier.predict(X_test_scaled)
# pd.DataFrame({"Prediction": predictions, "Actual": y_test}).sample(20)


# In[27]:


# Calculating the accuracy score
# acc_score = accuracy_score(y_test, predictions)
# print(f"Accuracy Score : {acc_score}")


# In[28]:


# Generate the confusion matrix
# cm = confusion_matrix(y_test, predictions)
# cm_df = pd.DataFrame(
#     cm, index = ["Actual 0", "Actual 1"],
#     columns = ["Predicted 0", "Predicted 1"]
# )

# Displaying results
# display(cm_df)


# In[29]:


# Generate classification report
# print("Classification Report")
# print(classification_report(y_test, predictions))


# ## Saving the model

# In[30]:


# Save the model to disk
filename = 'gb_model.sav'
#pickle.dump(classifier, open(filename, 'wb'))


# In[31]:


# Load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))


# ## Making predictions with the loaded model

# In[32]:


# Make Prediction
predictions = loaded_model.predict(X_test_scaled)
pd.DataFrame({"Prediction": predictions, "Actual": y_test}).sample(20)


# In[ ]:




