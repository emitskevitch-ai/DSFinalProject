import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import csv
import math as m
import numpy as np
import statistics as s
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
#see layers of fire geo data
import pyogrio

class KNN():
    """K-Nearest Neighbors classifier implemented from scratch."""
    def __init__(self, k):
        """Initialize the KNN model."""
        self.k = k

    def fit(self, X, y):
        """Store training data and labels."""
        #stores data points
        self.X_train = X
        #stores labels
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        """
        Desc: 
        Computes the Euclidean distance between two data points.
        Args: 
        x1 and x2, two data points.
        Returns: 
        a single float representing the Euclidean distance between x1 and x2.
        """
        #returns distance of the straight line in between the two points
        return float(m.dist(x1, x2))


    def compute_distances(self, x):
        """
        Desc: 
        Computes the distance from a single input point to every point in the training set.
        Args: 
        x, a single data point.
        Returns: 
        A list of (distance, label) tuples, one for each training sample. The distance is
        the Euclidean distance from x to the training point, and the label is the 
        corresponding training label.
        """
        #to store tuples
        distances = []
        #go through all training points
        for i in range(len(self.X_train)):
            #find distance between from x and training point i
            dist = self.euclidean_distance(x, self.X_train[i])
            #pairs distance with i-th label
            distances.append((dist, self.y_train[i]))
        #return distance, label list of tuples
        return distances

    def get_kneighbors(self, distances):
        """
        Desc: 
        Sorts the list of (distance, label) tuples by distance in ascending order and 
        return only the K closest entries.
        Args: 
        distance, a list of (distance, label) tuples as returned by compute_distances
        method.
        Returns: 
        A list of the K nearest (distance, label) tuples, sorted from closest to farthest.
        """
        #sorts the distances in ascending order and returns self.k amount of entries
        return sorted(distances)[:self.k]

    def classification(self, k_nearest):
        """
        Desc: 
        Takes the K nearest neighbors and returns the most frequently occurring class
        label among them (majority vote).
        Args: 
        k_nearest, which is a list of (distance, label) tuples representing the K closest
        training samples.
        Returns: 
        a single label, which is the class predicted by majority vote among the K
        neighbors.
        """
        #dictionary for the frequency of labels
        frequency = {}
        # loop through all neighbors
        for neighbor in k_nearest:
            # extract label either 0 or 1
            label = neighbor[1]
            #if label exists in dictionary add 1
            if label in frequency:
                frequency[label] += 1
            # if it dosent exist make a new one
            else:
                frequency[label] = 1
            #this will eventually initalize 1 and 0 in dictionary and then just add
        return max(frequency, key=frequency.get)
        

    def predict_single(self, x):
        """
        Desc: 
        Coordinates the full prediction pipeline for a single data point by calling
        compute_distances, get_kneighbors, and classification methods in sequence.
        Args: 
        x, a single data point
        Returns: 
        A single predicted class label for the input point.
        """
        #save data to cd
        cd = self.compute_distances(x)
        #save data to gk
        gk = self.get_kneighbors(cd)
        #use gk and call classification
        return self.classification(gk)

    def predict(self, X):
        """
        Desc: 
        Generates predictions for the given data by calling predict_single on each data
        point.
        Args: 
        X, a 2D list where each sublist represents a data point (row).
        Returns: 
        A list of predicted class labels, one per input sample, in the same order as the
        input.
        """
        # create list to append results too
        class_labels = []
        # for each data point in X
        for i in X:
            # append prediction to class labels
            class_labels.append(self.predict_single(i))
        return class_labels

def preprocessing(X_tr,X_val,X_te):
    """
    Desc:
    Takes the given data and scales it using the built-in function for z-score
    Args:
    X_tr - X training set
    X_val - X value set
    X_te - X test set
    Returns:
    scaled training, value, and test
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr) # fit only on train
    X_val_scaled = scaler.transform(X_val)
    X_te_scaled = scaler.transform(X_te)
    return X_tr_scaled, X_val_scaled, X_te_scaled

def train_model(k, X_tr, y_tr):
    """
    Desc:
    trains KNN model
    Args:
    k - number of nearest neighbors
    X_tr - X training set
    y_tr - Y training set
    Returns:
    returns trained knn model
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_tr,y_tr)
    return knn

def make_predictions(model,X_val):
    """
    Desc:
    makes predictions based on the validation data
    Args:
    model - model to make predicitons from
    X_val - x validation data
    Returns:
    prediction label list
    """
    prediction_val = model.predict(X_val) #validation is our practice question, we are getting the results
    return prediction_val


def evaluate(prediction_val, y_val):
    """
    Desc:
    Using the validation predictions, this function evaluates the accuracy of the model by returning the f1-score
    Args:
    prediction_val - prediction score
    y_val - x validation data
    Returns:
    f1 score
    """
    # Calculate F1 Score
    f1 = f1_score(y_val, prediction_val, pos_label=1)
    return f1


def find_best_k(X_tr, y_tr, X_val, y_val, max_k):
    """
    Desc:
    finds the best k based on the previous functions defined and using the validation data to choose k, NOT test data.
    Args:
    X_tr - X training set
    y_tr - y training set
    X_val - x validation set
    y_val - y validation set
    max_k - max number of nearest neighbors
    Returns:
    best amount of nearest neigbors
    """
    best_k = 1
    best_score = 0
    for k in range(1, max_k + 1):
        model = train_model(k, X_tr, y_tr)
        preds = make_predictions(model, X_val)
        score = evaluate(preds, y_val)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def select_features(X_tr, X_val, X_te, selected_cols):
    """
    Selects specific columns from datasets
    Args:
    X_tr - x training set
    X_val - x val set
    X_te -  x test set
    selected_cols - list of column names
    Returns:
    filtered datasets
    """
    X_tr_new = X_tr[selected_cols]
    X_val_new = X_val[selected_cols]
    X_te_new = X_te[selected_cols]
    return X_tr_new, X_val_new, X_te_new

def main():
    #import data set with pandas
    water = pd.read_csv("water_quality_allinfo_master.csv")
    #use geopandas to add geometry to dataframe using the longitude and latitude crs="EPSG:4326" communicates the lat/long format
    
    water_gdf = gpd.GeoDataFrame(water, geometry=gpd.points_from_xy(water['TargetLongitude'], water['TargetLatitude']), crs="EPSG:4326")

    #import data set with pandas
    fire = pd.read_csv("facility_fire_protection.csv")
    #use geopandas to add geometry to dataframe using the longitude and latitude crs="EPSG:4326" communicates the lat/long format
    #hold up! I dont think this is telling us what we want to know. This spreadsheet is the location of the fire station locations, not the 
    #wildfire instances. But then I don't think this is used again, so may not be an issue. 
    fire_gdf = gpd.GeoDataFrame(fire, geometry=gpd.points_from_xy(fire['Longitude'], fire['Latitude']), crs="EPSG:4326")

    gdb_fires_path = "C:/Users/Val/CS2500/fire24_1.gdb" 
    
    # Load and reproject fire perimeters
    fires = gpd.read_file(gdb_fires_path, engine="pyogrio", layer="firep24_1")
    fires = fires.to_crs("EPSG:4326")

    # Spatial join - water stations inside fire perimeters
    joined = gpd.sjoin(water_gdf, fires[['FIRE_NAME', 'ALARM_DATE', 'CONT_DATE', 'GIS_ACRES', 'geometry']], how='left', predicate='within')

    print(f"Total water readings: {len(joined)}")
    print(f"Readings near a fire: {joined['FIRE_NAME'].notna().sum()}")
    print(joined)
    '''
    output = pd.DataFrame({
    "ROW_ID": range(len(joined)),
    "Water Stations in Fire Perimeters": joined
    })
    '''
    joined.to_csv("Water_Location_predictions.csv", index=False)
    
'''
    water = pd.read_csv("swamp_data_dashboard.csv")
    fire = pd.read_csv("facility_fire_protection.csv")
    
    print("=== WATER COLUMNS ===")
    print(water.columns.tolist())
    print("\n=== WATER SHAPE ===")
    print(water.shape)
    print("\n=== WATER HEAD ===")
    print(water.head(2))

    print("\n=== FIRE COLUMNS ===")
    print(fire.columns.tolist())
    print("\n=== FIRE SHAPE ===")
    print(fire.shape)
    print("\n=== FIRE HEAD ===")
    print(fire.head(2))

    # see available layers
    print(pyogrio.list_layers(gdb_fires_path))

    # load the fire perimeters layer
    fires = gpd.read_file(gdb_fires_path, engine="pyogrio", layer="firep24_1")

    print("\n=== FIRE PERIMETERS COLUMNS ===")
    print(fires.columns.tolist())
    print("\n=== SHAPE ===")
    print(fires.shape)
    print("\n=== HEAD ===")
    print(fires.head(2))
    print("\n=== DATE RANGE ===")
    print(fires['ALARM_DATE'].min(), "to", fires['ALARM_DATE'].max())
    '''


if __name__ == "__main__":
    main()
