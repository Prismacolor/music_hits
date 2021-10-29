from matplotlib import pyplot as plt
import numpy
import pandas
import xlrd

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

music_df = pandas.read_excel('wm_project.xlsx')


# ## PREPROCESSING ## #
# Dropping the id and URI columns as those are only identifiers
# will also fill in the N/As in the hit column with 0, I feel it's safer to assume "not a hit"
# I will leave artist name, because it's possible an artist's name has some bearing
# maybe if the artist is already popular, does that increase the chances of a hit?
# As to the name of the song, will leave that too, may drop later if unrelated
# Will keep decade as maybe certain kinds of songs were hits only in certain decades
# scaling is probably needed as the data values have different ranges, will also encode categorical data
decade_mappings = {'60s': 60, '70s': 70, '80s': 80, '90s': 90, '00s': 0, '10s': 10}
features = ['track', 'artist', 'danceability', 'energy', 'key', 'loudness',	'mode', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'chorus_hit',
            'sections', 'decade']
final_features = []

music_df.drop(['uri'], axis=1, inplace=True)
music_df.drop(['id'], axis=1, inplace=True)
music_df.dropna(inplace=True)
music_df['artist'] = music_df['artist'].astype('string')
music_df['track'] = music_df['track'].astype('string')

df_encoder = LabelEncoder()
music_df['artist'] = df_encoder.fit_transform(music_df['artist'])
music_df['track'] = df_encoder.fit_transform(music_df['track'])
music_df['decade'] = music_df['decade'].map(decade_mappings)
# print(music_df.head(12))

music_df = music_df.reset_index(drop=True)
X_base = music_df.drop(['hit'], axis=1)
y = music_df['hit']

X = (X_base - X_base.min())/(X_base.max() - X_base.min())
X = X.astype(float).round(decimals=6)


# ## Feature analysis ## #
# will try a clustering algo with plots to see if there's a clear separation between hits/non hits
# chose to do the mini batch version, due to size of data and easier computation,
# minibatch will use small batches stored in memory, and with each new iteration uses random sample to update clusters
# I failed a bit on the clusters but in interest of time will go with what I have
# using all features didn't show a clear pattern, but there was a slight shift when plotting danceability to hits
# the centroids from kmeans were far apart, so there could be separation between the two clusters
kmeans = MiniBatchKMeans(n_clusters=2, batch_size=2048, max_iter=250, tol=0.0001).fit(X)
centroids = kmeans.cluster_centers_

plt.figure(1)
plt.scatter(X.iloc[:, 0], y)
plt.xlabel('Features')
plt.ylabel('Hit')
plt.savefig('features.png')
# plt.show()

plt.figure(2)
plt.scatter(X['danceability'], y)
plt.xlabel('danceability')
plt.ylabel('Hit')
plt.savefig('danceability.png')
# plt.show()

plt.figure(3)
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, marker='*', color='black')
plt.savefig('centroids.png')
# plt.show()

# let's see if we can figure out which features are the most likely to help with predictions
forest = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0)
forest.fit(X, y)
''' 
feat_importances = [0.05514875 0.06180678 0.08680834 0.07188301 0.02781902 0.06624802
 0.00810917 0.06590953 0.08802874 0.12571001 0.04965432 0.06213206
 0.05147428 0.06229211 0.0049293  0.04821463 0.03278136 0.03105057]
 #9 and #10 seem to be the highest with 3 and 4 following (Acoustic, instrumentalness, danceability, energy)
'''

# we can also try this another way, I think the results were a little more robust here
X_feat = SelectPercentile(chi2, percentile=30).fit(X, y).get_support().tolist()
'''
list of scores, with list of bool for who made the top six
[2.51708012e+00 3.84346911e-01 2.02987717e+02 9.89472049e+01, 3.06605916e-01 3.17920813e+01 5.71455206e+01 
5.33257119e+00, 5.48595496e+02 2.83728737e+03 1.39473494e+01 2.34937898e+02, 9.16214480e-01 2.59153865e+00 
2.89799049e+00 1.39443597e+00, 1.53612878e+00 3.15476628e-01]
[False, False, True, True, False, False, True, False, True, True, False, True, False, False, False, False, False, False]
['danceability', 'energy', 'mode', 'acousticness', 'instrumentalness', 'valence']
'''

for feat_, feature_ in zip(X_feat, features):
    if feat_:
        final_features.append(feature_)

for feature in features:
    if feature not in final_features:
        X.drop([feature], axis=1, inplace=True)


# ## BUILDING THE FINAL MODEL ## #
# not sure which model to use now, so will try grid search to test a few with a few diff parameters
# Can also make a validation set to test params before running final tests
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

models = [RandomForestClassifier(), LogisticRegression()]
params = [{'n_estimators': [100, 250, 500, 1200], 'max_depth': [6, 10], 'criterion': ['gini', 'entropy'], 'random_state': [0]},
          {'penalty': ['none', 'l2'], 'random_state': [0]}]


def get_models(models_list, params_list, Xtrain, ytrain):
    model_review = []
    for index, model in enumerate(models_list):
        print('start')
        use_all_processors = -1

        try:
            print(model)
            params = params_list[index]
            the_clf = GridSearchCV(estimator=model, param_grid=params, n_jobs=use_all_processors, verbose=100,
                                   error_score='raise')
        except Exception as e:
            print(e)
            continue

        print('processing')
        the_clf.fit(Xtrain, ytrain)
        model_review.append([the_clf, the_clf.best_params_, the_clf.best_score_])
        print('end')

    return model_review


final_models = get_models(models, params, X_train, y_train)

ideal_score = 0
ideal_model = ''

for model in final_models:
    if model[2] > ideal_score:
        best_score = model[2]
        ideal_model = model
    else:
        continue

# the initial winner is a Random Forest, w/ 250 estimators, and gini impurity
# final is 1200 estimators and entropy with max depth of 10
# made slight adjustments: upped est to 1500 and max depth to 12, don't want to go too deep to avoid overfitting
best_model = RandomForestClassifier(n_estimators=1500, criterion='entropy', max_depth=12, random_state=0)
best_model.fit(X_train, y_train)

# the initial result is 68.58% (model w/ 250 ests, and gini)
# my final result was... if you drop instead of fill the rate goes higher: 76.3
best_model.predict(x_test)
best_score = best_model.score(x_test, y_test)
print(best_score)