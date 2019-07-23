import pandas as pd
import geojson as gj
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def poi_in_cell(poi_coords, coords):
    return not ((poi_coords[0] > coords[1][0]) or (poi_coords[1] > coords[1][1]) or
                (poi_coords[0] < coords[3][0]) or (poi_coords[1] < coords[3][1]))

#[1] Define cell list with point of interest
def cell_list():
    poi = pd.read_csv('inputs/milano_pois/pois_milano_tripadvisor.csv')
    poi.sort_values('reviews', ascending=False).head()
    with open('inputs/milano-grid.geojson') as gf:
        grid = gj.load(gf)
    cell_position = pd.DataFrame([([cell["properties"]["cellId"]] + cell["geometry"]["coordinates"][0][0])
                                  for cell in grid['features']], columns=['cellId', 'lat', 'lon']).set_index('cellId')
    cell_position.head()

    poi_to_cell_dict = {}
    cell_reviews = defaultdict(int)
    for index, p in poi.iterrows():
        if index % 50 == 0:
            print index, p[1]
        for cell in grid['features']:
            if poi_in_cell([p[3], p[2]], cell['geometry']['coordinates'][0]):
                poi_to_cell_dict[p[0]] = cell['id']
                cell_reviews[cell['id']] += p[1]
    cell_reviews = pd.Series(cell_reviews)
    npoi = pd.Series(Counter(poi_to_cell_dict.values()))
    return npoi

#[2] Clustering
def kMeans(npoi):
    elbowMethod(npoi)
    df = pd.DataFrame({'square_id': npoi.index, 'poi': npoi.values})
    X = df.to_numpy()
    plt.scatter(X[:, 0], X[:, 1], s=50);
    plt.show()
    k = 5
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    y_kmeans = kmeans.fit_predict(df[['poi']])
    clust_labels = kmeans.predict(df[['poi']])
    cent = kmeans.cluster_centers_
    kmeans = pd.DataFrame(clust_labels)
    df.insert((df.shape[1]), 'kmeans', kmeans)
    print(df)
    ax1 = plt.axes()
    y_axis = ax1.axes.get_yaxis()
    y_axis.set_visible(False)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.show()
    return df

#[3] Elbow Method
def elbowMethod(npoi):
    Sum_of_squared_distances = []
    df = pd.DataFrame({'square_id': npoi.index, 'poi': npoi.values})
    X = df.to_numpy()
    plt.scatter(X[:, 0], X[:, 1], s=50);
    K = range(1, 15)
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        y_kmeans = kmeans.fit_predict(df[['poi']])
        clust_labels = kmeans.predict(df[['poi']])
        cent = kmeans.cluster_centers_
        Sum_of_squared_distances.append(kmeans.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Somma dei quadrati delle distanze')
    plt.title('Elbow Method per il k ottimo')
    plt.show()


#[4] Filter data by cluster id
def filter(n):
    df = kMeans(cell_list())
    if n == 0:
        veryLowPoi = df.query("kmeans == 0")
        return veryLowPoi['square_id'].tolist()
    if n == 1:
        lowPoi = df.query("kmeans == 1")
        return lowPoi['square_id'].tolist()
    if n == 2:
        mediumPoi = df.query("kmeans == 2")
        return mediumPoi['square_id'].tolist()
    if n == 3:
        highPoi = df.query("kmeans == 3")
        return highPoi['square_id'].tolist()
    if n == 4:
        veryHighPoi = df.query("kmeans == 4")
        return veryHighPoi['square_id'].tolist()

