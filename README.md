# SPOTLIGHT PROJECT
> [!NOTE]
> Not all of the code is included in this project due to confidentiality

## Background

I created this project for an web analytics startup called Lantern (previously Lazy Lantern). They currently have a product that monitors a company’s website data and then sends them an alert whenever they detect an anomaly in browsing behavior, and wanted to be able to provide their customers with more insight into the root cause of these alerts. Due to the wide variety of Lantern customers, the primary challenge was to create something that could generalize to any company with any anomaly.


## Example Use Case
Sarah is the founder of a clothing startup. She gets an anomaly alert from Lantern that says her company is selling twice their usual number of socks. This is obviously a good thing, but she wants to know if it’s because of something the company did.

Her analyst uses a tool like Google Analytics to manually check each of their hundreds of user groups one at a time. Several hours later he returns with good news. The company is running an ad campaign about socks and it’s apparently going very well! Sarah is happy, but already has five more alerts for him to look into.

Even large companies don’t have the bandwidth to track down the cause of each metric anomaly. The goal of the project is to automate the analyst’s task of identifying which user groups are causing changes in web traffic.


## Approach
### Data Structure
![image](https://github.com/user-attachments/assets/3ddb4287-0f40-488b-b5b1-562f2e1f4d24)

Web data is defined by events over time that performed by users. There can be hundreds and thousands of users creating millions of events at any given time
A Mongo database contained all of the raw events and another SQL database contained details about the anomalies detected by Lantern’s algorithm. By combining the event and anomaly databases, I tagged the events and users connected to each anomaly. 

In this socks ad example, I used the anomaly information to tag events labeled “buy socks”, but only when they fall within the anomaly window.

Because I am using pre-detected anomalies to label the data, I can use that to determine which user features predict these labels. By setting up the data this way, the problem can be approached with supervised learning.

I used the anomaly events labeled as the targets and all other events as non-targets.

![image](https://github.com/user-attachments/assets/48a1be4d-fe00-4adb-99e7-e576359499d7)
![image](https://github.com/user-attachments/assets/d43416f1-312a-41ed-8040-76994e9b84b1)

Events are represented here as colored boxes but they are actually represented as nested dictionaries of behavioral and demographic features.

![image](https://github.com/user-attachments/assets/e7e39ce3-8eb5-4128-85c5-d57b9cb1101d)

My goal was to find which of these features predict the targets (in this case the “sock purchase” events during the anomaly period). To do that, I needed to get the data into a more usable format. This data transformation was actually one of the most challenging aspects of this project because even within a single company, there is a great deal of variety in the data structure. I ended up creating a way to transform any feature out of this format and into a dataframe column without needing to know its exact placement [[link](https://github.com/sierrasosa/spotlight-project/blob/master/data_cleaning.py)].


### Model - Random Forest Classifier
After cleaning and EDA I discovered my data had the following attributes:
1. A class imbalance of about 10:1 against anomaly events
2. Hightly correlated features
3. Non-linear relationships

I chose a random forest classifier because it fits these criteria and can perform well with minimal training/tuning.

I trained a separate model for each anomaly. For each one, I used a grid search optimized for recall to tune hyperparameters [[link](https://github.com/sierrasosa/spotlight-project/blob/master/rf_gridsearch.py)]. 

Below is the the confusion matrix for one of the models (left) and the ROC distribution (right) for a sample of 50 models (average 0.73).

![image](https://github.com/user-attachments/assets/855762a7-aa7d-487d-b7ed-1684c4a0da36)

### Feature Importance Measurement - Recursive Feature Elimination + SHAP
So far the model has not given us information we don't already know. However, we need it so that we can use feature importance measurement to figure out which features the model is relying on to predict the anomaly events.

Having correlated features can artificially decrease importance scores (for example: the features “viewed the socks ad” and “entered site through paid advertising” are highly correlated). Because I fit a model to each anomaly, I needed a way to automatically remove correlated features without human intervention. For this I used recursive feature elimination. It works by continually removing the lowest ranking features until eventually the less important columns are removed and the remaining columns can be scored appropriately.

I then used SHAP values to interpret directionality. For example, in this graph, we see that high positive values for users that saw the socks ad campaign increases the likelihood that the user is part of the anomaly.

![image](https://github.com/user-attachments/assets/81006345-3b81-4e4f-b241-80aecea8b31d)

The highest score(s) determines the root cause. If the model was a poor fit or no features stood out as important, then the root cause was deemed unknown.

With this method I was able to provide the root cause for over 80% of surfaced anomalies.
