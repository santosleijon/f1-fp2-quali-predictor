import math
from random import randint
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing, tree
from sklearn.datasets import load_iris
from sklearn.metrics import mean_absolute_error

def load_encoded_data_set(file: str) -> pd.DataFrame:
  data_set = pd.read_csv(file)
  data_set = data_set.drop(columns=['Year', 'Race'])
  categorical_columns = data_set.select_dtypes(['object']).columns

  label_encoded_name_mapping = {}
  label_encoder = preprocessing.LabelEncoder()
  for col in data_set[categorical_columns]:
      data_set[col]= label_encoder.fit_transform(data_set[col])
      label_encoded_name_mapping[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

  for category_name in label_encoded_name_mapping:
    print("---------------------------------------")
    print(category_name)
    print("---------------------------------------")
    for index, category_value in enumerate(label_encoded_name_mapping[category_name]):
       print(f"{index} = {category_value}")
    print("---------------------------------------")
    print("")

  return data_set

encoded_data_set = load_encoded_data_set("data_set_2021-2022.csv")
data_set_rows_count = encoded_data_set.shape[0]

percentage_training_data = 0.8
percentage_test_data = 1 - percentage_training_data

training_set_rows_count = math.floor(data_set_rows_count * percentage_training_data)
traning_set = encoded_data_set.head(training_set_rows_count)
test_set_rows_count = math.floor(data_set_rows_count * percentage_test_data)
test_set = encoded_data_set.tail(test_set_rows_count)

training_set_features_data = traning_set.drop(columns=['QualifyingPosition']).to_numpy()
training_set_target_data = traning_set.loc[:,"QualifyingPosition"].to_numpy()

clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=10)
clf = clf.fit(training_set_features_data, training_set_target_data)

test_set_features_data = test_set.drop(columns=['QualifyingPosition']).to_numpy()
test_set_actual_target_data = test_set.loc[:,"QualifyingPosition"].to_numpy()

test_set_predicted_target_data = clf.predict(test_set_features_data)


# Evalutation of resulting model
mean_error = mean_absolute_error(test_set_actual_target_data, test_set_predicted_target_data).round(4)
print(f'Mean absolute error: {mean_error}')

tree.export_graphviz(clf,
                     out_file="decision_tree.dot",
                     feature_names=["Track","Team","Driver","LapNumber","Compound","TyreLife","TrackStatus","PracticeLapTimeDelta"],
                     class_names=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"],
                     filled=True,
                     rounded=True,  
                     special_characters=True)  
