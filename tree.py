import numpy as np
from sklearn.tree import DecisionTreeRegressor

# paste your get_features_targets function here
def get_features_targets(data):
  features = np.zeros(shape=(len(data), 4))
  features[:, 0] = data['u'] - data['g']
  features[:, 1] = data['g'] - data['r']
  features[:, 2] = data['r'] - data['i']
  features[:, 3] = data['i'] - data['z']
  targets = data['redshift']
  
  return features, targets

# paste your median_diff function here
def median_diff(predicted, actual):
  med_diff = np.median(np.abs(predicted - actual))
  
  return med_diff

# write a function that splits the data into training and testing subsets
def split_data(features, targets):
  split_f = features.shape[0]//2
  train_f = features[:split_f]
  test_f = features[split_f:]
  
  split_t = features.shape[0]//2
  train_t = targets[:split_t]
  test_t = targets[split_t:]
  
  return train_f, test_f, train_t, test_t

# trains the model and returns the prediction accuracy with median_diff
def validate_model(model, features, targets):
  # split the data into training and testing features and predictions
  train_features, test_features, train_targets, test_targets = split_data(features, targets)
  
  # train the model
  model.fit(train_features, train_targets)

  # get the predicted_redshifts
  predictions = model.predict(test_features)
  
  # use median_diff function to calculate the accuracy
  return median_diff(test_targets, predictions)


if __name__ == "__main__":
  data = np.load('sdss_galaxy_colors.npy')
  features, targets = get_features_targets(data)

  # initialize model
  dtr = DecisionTreeRegressor()

  # validate the model and print the med_diff
  diff = validate_model(dtr, features, targets)
  print('Median difference: {:f}'.format(diff))
