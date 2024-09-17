#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from patsy import dmatrices
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree



#%%
mcdonalds_df = pd.read_csv(r'C:\Users\sauma\Documents\Mcdonalds\McDonalds Case Study\mcdonalds.csv')
print(mcdonalds_df)
#%%
# Assuming you've already imported the mcdonalds dataset into a Pandas DataFrame called mcdonalds_df
MD_x = mcdonalds_df.iloc[:, 0:11]

# Convert "Yes" to 1 and "No" to 0
MD_x = (MD_x == "Yes").astype(int)

# Calculate column means and round to 2 decimal places
means = MD_x.mean(axis=0).round(2)

print(means)
#%%

#%%
# Assuming you've already created the MD_x matrix
pca = PCA()
MD_pca = pca.fit_transform(MD_x)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
print("Cumulative Explained Variance Ratio:")
print(pca.explained_variance_ratio_.cumsum())


# Assuming you've already created the MD_x matrix
pca = PCA()
MD_pca = pca.fit_transform(MD_x)

print("Standard deviations:")
print(pca.singular_values_)

print("Rotation (n x k) = (11 x 11):")
print(pd.DataFrame(pca.components_.T, index=['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting'], 
                    columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11']).round(2))



# Assuming you've already created the MD_x matrix
pca = PCA()
MD_pca = pca.fit_transform(MD_x)

plt.scatter(MD_pca[:, 0], MD_pca[:, 1], c='grey')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projected Data')
plt.show()
#%%
#import numpy as np
#from sklearn.cluster import KMeans
#from sklearn.metrics import pairwise_distances_argmin_min

# Setting the seed for reproducibility
np.random.seed(1234)

# Example data, replace this with your actual data
# MD.x should be a 2D numpy array where rows are samples and columns are features
MD_x = np.random.rand(100, 5)  # replace with your actual data

# Function to perform KMeans clustering and choose the best number of clusters
def step_kmeans(X, min_clusters, max_clusters, nrep=10):
    best_labels = None
    best_inertia = None
    best_n_clusters = None

    for n_clusters in range(min_clusters, max_clusters + 1):
        inertia_sum = 0

        for _ in range(nrep):
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=np.random.randint(0, 10000))
            kmeans.fit(X)
            inertia_sum += kmeans.inertia_

        avg_inertia = inertia_sum / nrep
        if best_inertia is None or avg_inertia < best_inertia:
            best_inertia = avg_inertia
            best_labels = kmeans.labels_
            best_n_clusters = n_clusters

    return best_labels, best_n_clusters

# Running the clustering process
labels, n_clusters = step_kmeans(MD_x, 2, 8, nrep=10)

# Displaying the chosen number of clusters and labels
print(f"Optimal number of clusters: {n_clusters}")
print(f"Cluster labels: {labels}")

# Optionally, you can reassign cluster labels to a new set of cluster labels
# This is a simplified example; relabeling may need custom logic based on your specific requirements
unique_labels = np.unique(labels)
new_labels = np.arange(len(unique_labels))
relabeled = np.array([new_labels[np.where(unique_labels == label)[0][0]] for label in labels])

print(f"Relabeled cluster labels: {relabeled}")

# %%
np.random.seed(1234)

# %%
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans

# Function to perform KMeans clustering and compute inertia for each number of clusters
def step_kmeans(X, min_clusters, max_clusters, nrep=10):
    best_inertia = []
    best_labels = []
    best_n_clusters = None

    for n_clusters in range(min_clusters, max_clusters + 1):
        inertia_sum = 0
        for _ in range(nrep):
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=np.random.randint(0, 10000))
            kmeans.fit(X)
            inertia_sum += kmeans.inertia_

        avg_inertia = inertia_sum / nrep
        best_inertia.append(avg_inertia)
        best_labels = kmeans.labels_
        best_n_clusters = n_clusters

    return best_labels, best_n_clusters, best_inertia

# Example data, replace this with your actual data
MD_x = np.random.rand(100, 5)  # replace with your actual data

# Running the clustering process
labels, n_clusters, inertia = step_kmeans(MD_x, 2, 8, nrep=10)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(range(2, 9), inertia, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Segments (Clusters)')
plt.ylabel('Average Inertia')
plt.title('Clustering Inertia vs. Number of Clusters')
plt.grid(True)
plt.xticks(range(2, 9))
plt.show()

# %%
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
#from sklearn.metrics import adjusted_rand_score
#from sklearn.utils import resample

# Function to perform bootstrapping and compute ARI for each number of clusters
def boot_kmeans(X, min_clusters, max_clusters, nrep=10, nboot=100):
    ari_scores = np.zeros((max_clusters - min_clusters + 1, nboot))
    cluster_range = range(min_clusters, max_clusters + 1)
    
    for n_clusters in cluster_range:
        for b in range(nboot):
            # Bootstrap sampling
            X_bootstrap = resample(X, random_state=np.random.randint(0, 10000))
            
            # Fit KMeans
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=np.random.randint(0, 10000))
            kmeans.fit(X_bootstrap)
            
            # Compute ARI with the original data
            kmeans_original = KMeans(n_clusters=n_clusters, n_init=1, random_state=np.random.randint(0, 10000))
            kmeans_original.fit(X)
            
            ari_scores[n_clusters - min_clusters, b] = adjusted_rand_score(kmeans_original.labels_, kmeans.labels_)
    
    avg_ari_scores = np.mean(ari_scores, axis=1)
    return cluster_range, avg_ari_scores

# Example data, replace this with your actual data
MD_x = np.random.rand(100, 5)  # replace with your actual data

# Running the bootstrapping process
cluster_range, avg_ari_scores = boot_kmeans(MD_x, 2, 8, nrep=10, nboot=100)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, avg_ari_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Segments (Clusters)')
plt.ylabel('Adjusted Rand Index')
plt.title('Adjusted Rand Index vs. Number of Clusters')
plt.grid(True)
plt.xticks(cluster_range)
plt.show()

# %%
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans

# Example data, replace this with your actual data
MD_x = np.random.rand(100, 5)  # replace with your actual data

# Perform KMeans clustering with 4 clusters
kmeans = KMeans(n_clusters=4, n_init=10, random_state=1234)
kmeans.fit(MD_x)
cluster_labels = kmeans.labels_

# Plotting histogram of the cluster assignments
plt.figure(figsize=(10, 6))
plt.hist(cluster_labels, bins=np.arange(-0.5, 4.5, 1), edgecolor='k', alpha=0.7)
plt.xlabel('Cluster Assignment')
plt.ylabel('Frequency')
plt.title('Histogram of Cluster Assignments for 4 Clusters')
plt.xticks(ticks=np.arange(4), labels=np.arange(4))
plt.xlim(-0.5, 3.5)  # Adjust limits to fit the cluster range
plt.grid(True)
plt.show()

# %%
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score

# Example data, replace this with your actual data
MD_x = np.random.rand(100, 5)  # replace with your actual data
#%%
# Perform KMeans clustering with 4 clusters
kmeans = KMeans(n_clusters=4, n_init=10, random_state=1234)
kmeans.fit(MD_x)
cluster_labels = kmeans.labels_
#%%
# Compute silhouette scores for each cluster
def compute_segment_stability(X, labels, n_clusters):
    stability_scores = np.zeros(n_clusters)
    for cluster in range(n_clusters):
        # Get data points in the current cluster
        cluster_points = X[labels == cluster]
        if len(cluster_points) > 1:
            # Compute the silhouette score for this cluster
            #stability_scores[cluster] = silhouette_score(X, labels, labels == cluster)
            stability_scores[cluster] = silhouette_score(X, labels)

        else:
            # If a cluster has only one sample, stability score is not defined
            stability_scores[cluster] = np.nan
    return stability_scores
#%%
# Compute stability scores for 4 clusters
#stability_scores[MD_x] = compute_segment_stability(MD_x, cluster_labels, 4)
stability_scores = compute_segment_stability(MD_x, cluster_labels, 4)
# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(range(4), stability_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Segment Number')
plt.ylabel('Segment Stability')
plt.title('Segment Stability vs. Segment Number')
plt.ylim(0, 1)  # y-axis limits
plt.grid(True)
plt.xticks(range(4))  # x-axis ticks
plt.show()

# %%
#import numpy as np
#import pandas as pd
#from sklearn.cluster import KMeans
#from sklearn.mixture import GaussianMixture
#from sklearn.metrics import silhouette_score
#import matplotlib.pyplot as plt

# Assuming `MD_x` is the data you're working with
# MD_x = pd.read_csv('data.csv')  # Example: loading data

# K-means clustering from k=2 to k=8
results = []
silhouette_scores = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=1234, n_init=10).fit(MD_x)
    results.append(kmeans)
    silhouette_scores.append(silhouette_score(MD_x, kmeans.labels_))

# Plot silhouette scores (equivalent to AIC/BIC/ICL plot)
plt.figure(figsize=(10, 6))
plt.plot(range(2, 9), silhouette_scores, marker='o')
plt.title('Silhouette Scores for K-means Clustering')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Select a specific k, say k=4 (similar to selecting k=4 in R)
k = 4
kmeans = KMeans(n_clusters=k, random_state=1234, n_init=10).fit(MD_x)

# Mixture Model (equivalent to FLXMCmvbinary)
gmm = GaussianMixture(n_components=k, random_state=1234).fit(MD_x)

# Compare clusters from K-means and Mixture Model
kmeans_clusters = kmeans.labels_
gmm_clusters = gmm.predict(MD_x)

comparison_table = pd.crosstab(kmeans_clusters, gmm_clusters)
print(comparison_table)

# Log-likelihood of the mixture model
log_likelihood = gmm.score(MD_x) * MD_x.shape[0]
print(f"Log Likelihood of the Mixture Model: {log_likelihood}")
#%%
#import pandas as pd

# Example: loading the McDonald's dataset
mcdonalds = pd.read_csv(r'C:\\Users\\sauma\\Documents\\Mcdonalds\\McDonalds Case Study\\mcdonalds.csv')

# Create a mapping for the 'Like' column
like_mapping = {
    "I hate it!-5": -5,
    "-4": -4,
    "-3": -3,
    "-2": -2,
    "-1": -1,
    "0": 0,
    "+1": 1,
    "+2": 2,
    "+3": 3,
    "+4": 4,
    "I love it!+5": 5
}

# Map the 'Like' column to numeric values
mcdonalds['Like_n'] = mcdonalds['Like'].replace(like_mapping)

# Reverse the scale: 6 - Like_n
mcdonalds['Like_n_reversed'] = 6 - mcdonalds['Like_n']

# Display the frequency table similar to R's `table(mcdonalds$Like.n)`
like_n_reversed_counts = mcdonalds['Like_n_reversed'].value_counts().sort_index(ascending=False)

# Print the reversed table
print(like_n_reversed_counts)
#%%
#import pandas as pd
#from patsy import dmatrices

# Assuming the McDonald's dataset is already loaded into the `mcdonalds` DataFrame

# Select the first 11 column names to include as features
features = mcdonalds.columns[:11].tolist()

# Create a formula string
formula = "Like_n ~ " + " + ".join(features)

# Display the formula
print(formula)

# To create the design matrices (like model.matrix in R)
y, X = dmatrices(formula, data=mcdonalds, return_type='dataframe')

# `y` is the dependent variable (Like_n), and `X` is the feature matrix

#%%
#import numpy as np
#from sklearn.mixture import GaussianMixture
#from sklearn.linear_model import LinearRegression

# Random seed for reproducibility
np.random.seed(1234)

# Load your dataset (similar to `mcdonalds` in R)
# Replace with your actual data
# For example purposes, I'll generate random data
X = np.random.rand(1453, 2)  # Independent variables
y = np.random.rand(1453)     # Dependent variable

# Fit the Gaussian Mixture Model (GMM) with 2 components
gmm = GaussianMixture(n_components=2, n_init=10, verbose=0, random_state=1234)
gmm.fit(X)

# Predict the cluster assignment
clusters = gmm.predict(X)

# Fit linear regression models for each cluster
regressions = []
for i in range(2):
    # Get the data points assigned to the current cluster
    X_cluster = X[clusters == i]
    y_cluster = y[clusters == i]

    # Fit linear regression
    model = LinearRegression()
    model.fit(X_cluster, y_cluster)

    regressions.append(model)

# Output the number of data points per cluster and model information
cluster_sizes = np.bincount(clusters)
print("Cluster sizes:", cluster_sizes)

for i, model in enumerate(regressions):
    print(f"Cluster {i+1} - Coefficients: {model.coef_}, Intercept: {model.intercept_}")

# %%
#import numpy as np
#from sklearn.mixture import GaussianMixture
#import statsmodels.api as sm

# Random seed for reproducibility
np.random.seed(1234)

# Load your dataset (similar to `mcdonalds` in R)
# Replace with your actual data
# For example purposes, I'll generate random data
X = np.random.rand(1453, 10)  # Independent variables
y = np.random.rand(1453)      # Dependent variable

# Add an intercept term to X for linear regression
X = sm.add_constant(X)

# Fit the Gaussian Mixture Model (GMM) with 2 components
gmm = GaussianMixture(n_components=2, n_init=10, verbose=0, random_state=1234)
gmm.fit(X)

# Predict the cluster assignment
clusters = gmm.predict(X)

# Fit linear regression models for each cluster
results = []
for i in range(2):
    # Get the data points assigned to the current cluster
    X_cluster = X[clusters == i]
    y_cluster = y[clusters == i]

    # Fit the model using statsmodels' OLS (Ordinary Least Squares)
    model = sm.OLS(y_cluster, X_cluster)
    result = model.fit()

    results.append(result)

# Output the summary for each cluster
for i, result in enumerate(results):
    print(f"Summary for Cluster {i+1}:")
    print(result.summary())

# %%
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

# Assuming results is a list of statsmodels regression result objects
# (from the previous code)
significance_level = 0.05

# Prepare data for plotting
plot_data = []
for i, result in enumerate(results):
    # Extract the summary table from the result
    summary_table = result.summary2().tables[1]
    
    # Collect relevant statistics
    for idx, row in summary_table.iterrows():
        coefficient = row['Coef.']
        std_error = row['Std.Err.']
        p_value = row['P>|t|']
        
        # Create a dictionary with the relevant info for plotting
        plot_data.append({
            'Component': f'Component {i+1}',
            'Variable': idx,
            'Coefficient': coefficient,
            'Lower CI': coefficient - 1.96 * std_error,
            'Upper CI': coefficient + 1.96 * std_error,
            'Significant': p_value < significance_level
        })

# Convert list of dictionaries to a pandas DataFrame for easier plotting
#import pandas as pd
df_plot = pd.DataFrame(plot_data)

# Create the plot
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Plot coefficients with confidence intervals
for i, component in enumerate(df_plot['Component'].unique()):
    component_data = df_plot[df_plot['Component'] == component]
    
    # Plot confidence intervals
    plt.errorbar(component_data['Variable'], component_data['Coefficient'], 
                 yerr=[component_data['Coefficient'] - component_data['Lower CI'], 
                       component_data['Upper CI'] - component_data['Coefficient']],
                 fmt='o', label=component)
    
    # Highlight significant coefficients
    sig_data = component_data[component_data['Significant']]
    plt.scatter(sig_data['Variable'], sig_data['Coefficient'], color='red', 
                label=f'{component} Significant', zorder=5)

# Add labels and legend
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Coefficient')
plt.title('Coefficients and Significance by Component')
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()

# %%
#import numpy as np
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#from scipy.spatial.distance import pdist
#from scipy.cluster.hierarchy import linkage, dendrogram

# Assuming MD.x is a DataFrame with your data
# Replace with your actual data
MD_x = pd.DataFrame(np.random.rand(10, 5), columns=list('ABCDE'))

# Step 1: Hierarchical clustering
# Calculate the distance matrix (like dist() in R)
dist_matrix = pdist(MD_x.T)  # Taking the distance of the transpose to cluster variables

# Perform hierarchical clustering (like hclust() in R)
Z = linkage(dist_matrix, method='ward')  # Using Ward's method for clustering

# Step 2: Plot a dendrogram to get the order of clustering (like MD.vclust$order in R)
plt.figure(figsize=(8, 4))
dendro = dendrogram(Z, labels=MD_x.columns, orientation='top', leaf_rotation=90)

# Extract the order of clustered variables
cluster_order = dendro['ivl']  # Order of variables after clustering

# Step 3: Plot a barchart (or heatmap) in the order of hierarchical clustering
# Reorder the columns of the data according to the clustering order
MD_x_ordered = MD_x[cluster_order]

# Step 4: Create a heatmap (equivalent to shaded barchart in R)
plt.figure(figsize=(8, 6))
sns.heatmap(MD_x_ordered.T, cmap='coolwarm', annot=True, cbar=True, linewidths=0.5)
plt.title('Clustered Heatmap of MD.x')
plt.show()

# %%
#import numpy as np
#import pandas as pd
#from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
#import seaborn as sns

# Assuming MD_x is a DataFrame with your data
# Replace this with your actual dataset
MD_x = pd.DataFrame(np.random.rand(1453, 10), columns=list('ABCDEFGHIJ'))

# Assuming MD_k4 is a list or array with cluster labels for each point
# Replace this with your actual cluster assignments
MD_k4 = np.random.randint(0, 4, size=1453)  # Example with 4 clusters

# Step 1: Perform PCA on the data
pca = PCA(n_components=2)
MD_pca = pca.fit_transform(MD_x)

# Step 2: Create a scatter plot of the projected data (first two principal components)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=MD_pca[:, 0], y=MD_pca[:, 1], hue=MD_k4, palette='Set1', s=100)

# Label the axes
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Projection of Clusters onto Principal Components')

# Optional: Draw the principal component axes (projAxes equivalent)
# Project the feature vectors onto the PCA space
for i in range(pca.components_.shape[1]):  # Loop through each original feature
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
              color='black', head_width=0.02, alpha=0.8)
    plt.text(pca.components_[0, i], pca.components_[1, i], MD_x.columns[i], 
             color='black', ha='center', va='center')

# Show the plot
plt.grid(True)
plt.show()

# %%
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from statsmodels.graphics.mosaicplot import mosaic

# Assuming you have the cluster labels (k4) and the 'Like' column from your dataset
# Replace with your actual data
# Example cluster assignments
k4 = np.random.randint(0, 4, size=1453)  # Example with 4 clusters
# Example 'Like' column (could be Yes/No or a scale)
mcdonalds_like = np.random.choice(['Yes', 'No'], size=1453)

# Create a DataFrame similar to the R `table(k4, mcdonalds$Like)`
data = pd.DataFrame({'Segment': k4, 'Like': mcdonalds_like})

# Create a contingency table (like table() in R)
contingency_table = pd.crosstab(data['Segment'], data['Like'])

# Plot the mosaic plot
plt.figure(figsize=(8, 6))
mosaic(contingency_table.stack(), gap=0.02, title="", labelizer=lambda k: "")
plt.xlabel("Segment Number")
plt.ylabel("Proportion")
plt.show()

# %%
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from statsmodels.graphics.mosaicplot import mosaic

# Assuming you have the cluster labels (k4) and the 'Gender' column from your dataset
# Replace with your actual data
# Example cluster assignments
k4 = np.random.randint(0, 4, size=1453)  # Example with 4 clusters
# Example 'Gender' column (Male/Female or other categories)
mcdonalds_gender = np.random.choice(['Male', 'Female'], size=1453)

# Create a DataFrame similar to the R `table(k4, mcdonalds$Gender)`
data = pd.DataFrame({'Segment': k4, 'Gender': mcdonalds_gender})

# Create a contingency table (like table() in R)
contingency_table = pd.crosstab(data['Segment'], data['Gender'])

# Plot the mosaic plot
plt.figure(figsize=(8, 6))
mosaic(contingency_table.stack(), gap=0.02, title="", labelizer=lambda k: "")
plt.xlabel("Segment Number")
plt.ylabel("Proportion")
plt.show()

# %%
#import pandas as pd
#import numpy as np
#from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
#import matplotlib.pyplot as plt

# Example data (replace with your actual data)
np.random.seed(123)
age = np.random.randint(18, 65, size=1453)
like_n = np.random.choice(['Yes', 'No'], size=1453)
visit_frequency = np.random.choice(['High', 'Medium', 'Low'], size=1453)
gender = np.random.choice(['Male', 'Female'], size=1453)
k4 = np.random.randint(0, 4, size=1453)  # Example with 4 clusters

# Create a DataFrame
data = pd.DataFrame({
    'Like.n': like_n,
    'Age': age,
    'VisitFrequency': visit_frequency,
    'Gender': gender,
    'k4': k4
})

# Define the target and features
X = data[['Like.n', 'Age', 'VisitFrequency', 'Gender']]
y = (data['k4'] == 3).astype(int)  # Target variable

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Create and fit the decision tree classifier
tree = DecisionTreeClassifier(criterion='entropy')  # Using 'entropy' as it is similar to conditional inference
tree.fit(X, y)

# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=X.columns, class_names=['Not Cluster 3', 'Cluster 3'], filled=True)
plt.title('Decision Tree')
plt.show()

# Print textual representation of the tree
tree_text = export_text(tree, feature_names=list(X.columns))
print(tree_text)

# %%
#import pandas as pd
#import numpy as np

# Example data (replace with your actual data)
np.random.seed(123)
age = np.random.randint(18, 65, size=1453)
like_n = np.random.choice(['Yes', 'No'], size=1453)
visit_frequency = np.random.choice([1, 2, 3, 4], size=1453)  # Numeric encoding of visit frequency
gender = np.random.choice(['Male', 'Female'], size=1453)
k4 = np.random.randint(1, 5, size=1453)  # Cluster assignments

# Create a DataFrame
data = pd.DataFrame({
    'VisitFrequency': visit_frequency,
    'k4': k4
})

# Calculate the mean visit frequency for each cluster
visit_means = data.groupby('k4')['VisitFrequency'].mean()

# Display the result
print(visit_means)

# %%
#import pandas as pd
#import numpy as np

# Example data (replace with your actual data)
np.random.seed(123)
like_n = np.random.normal(size=1453)  # Simulated 'Like.n' as continuous values
k4 = np.random.randint(1, 5, size=1453)  # Cluster assignments

# Create a DataFrame
data = pd.DataFrame({
    'Like.n': like_n,
    'k4': k4
})

# Calculate the mean 'Like.n' for each cluster
like_means = data.groupby('k4')['Like.n'].mean()

# Display the result
print(like_means)

# %%
#import pandas as pd
#import numpy as np

# Example data (replace with your actual data)
np.random.seed(123)
gender = np.random.choice(['Male', 'Female'], size=1453)
k4 = np.random.randint(1, 5, size=1453)  # Cluster assignments

# Create a DataFrame
data = pd.DataFrame({
    'Gender': gender,
    'k4': k4
})

# Create a binary indicator for Female
data['IsFemale'] = (data['Gender'] == 'Female').astype(int)

# Calculate the mean proportion of females for each cluster
female_proportions = data.groupby('k4')['IsFemale'].mean()

# Display the result
print(female_proportions)

# %%
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

# Example data (replace with your actual data)
visit = np.array([3.040426, 2.482490, 3.891975, 3.950249])  # Example visit frequencies
like = np.array([-0.1319149, -2.4902724, 2.2870370, 2.7114428])  # Example like values
female_proportions = np.array([0.5851064, 0.4319066, 0.4783951, 0.6144279])  # Example female proportions

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(visit, like, s=10 * female_proportions * 100, alpha=0.7)  # Scale sizes
plt.xlim(2, 4.5)
plt.ylim(-3, 3)

# Add text labels
for i, label in enumerate(range(1, 5)):
    plt.text(visit[i], like[i], str(label), fontsize=12, ha='center', va='center')

# Set labels and title
plt.xlabel('Visit Frequency')
plt.ylabel('Like')
plt.title('Scatter Plot of Visit Frequency vs Like')
plt.grid(True)
plt.show()

# %%
