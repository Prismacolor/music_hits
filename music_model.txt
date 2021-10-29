'''
Copyright <YEAR> <COPYRIGHT HOLDER>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


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
from sklearn.metrics import confusion_matrix

music_df0 = pandas.read_excel('data\\wm_project.xlsx')

decade_mappings = {'60s': 60, '70s': 70, '80s': 80, '90s': 90, '00s': 0, '10s': 10}
features = ['track', 'artist', 'danceability', 'energy', 'key', 'loudness',	'mode', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'chorus_hit',
            'sections', 'decade']


def get_stats(df):
    stats_list = []

    for col in df.iloc[:, 2: -2]:
        col_mean = df[col].mean()
        col_std = df[col].std()
        col_var = col_std ** 2
        min_val = df[col].min()
        max_val = df[col].max()
        stats_list.append({col: ['mean: ' + str(col_mean), 'std: ' + str(col_std), 'var: ' + str(col_var),
                                 'min: ' + str(min_val), 'max: ' + str(max_val)]})

    return stats_list


def clean_data(df):
    df.drop(['uri'], axis=1, inplace=True)
    df.drop(['id'], axis=1, inplace=True)
    df['artist'] = df['artist'].astype('string')
    df['track'] = df['track'].astype('string')

    df_encoder = LabelEncoder()
    df['artist'] = df_encoder.fit_transform(df['artist'])
    df['track'] = df_encoder.fit_transform(df['track'])
    df['decade'] = df['decade'].map(decade_mappings)
    # print(df.head(9))

    return df


def scale_data(df):
    df = (df - df.min()) / (df.max() - df.min())
    df = df.astype(float).round(decimals=6)

    return df


def make_plots(x_df, y_df):
    plt_count = 1
    feature_list = ['danceability', 'acousticness', 'instrumentalness']
    for f in feature_list:
        plt.figure(plt_count)
        plt.scatter(x_df[f], y_df)
        plt.xlabel(f)
        plt.ylabel('Hit')
        plt.savefig('plots\\' + f + '.png')
        # plt.show()

        plt_count += 1


def get_final_feat(xdf, xfeat, xfeatures):
    final_features_ = []
    for xfeat_, xfeature_ in zip(xfeat, xfeatures):
        if xfeat_:
            final_features_.append(xfeature_)

    for xfeature in xfeatures:
        if xfeature not in final_features_:
            xdf.drop([xfeature], axis=1, inplace=True)

    return xdf, final_features_


def get_models(models_list, params_list, xtrain, ytrain):
    model_review = []
    for index, model_ in enumerate(models_list):
        print('start')
        use_all_processors = -1

        try:
            params_ = params_list[index]
            the_clf = GridSearchCV(estimator=model_, param_grid=params_, n_jobs=use_all_processors, verbose=100,
                                   error_score='raise')
        except Exception as e:
            print(e)
            continue

        print('processing')
        the_clf.fit(xtrain, ytrain)
        model_review.append([the_clf, the_clf.best_params_, the_clf.best_score_])
        print('end')

    return model_review


# ## PREPROCESSING ## #
# Will separate hits and nonhits for initial analysis
# Dropped the id and URI columns as those are only identifiers and have no bearing on the target label
# Scaling is also probably needed as the data values have different ranges, and will encode categorical data
# I initially filled the N/As in the hit column with 0, later dropped them which led to an increase in accuracy
# We will keep all features to start, and maybe focus on the most important ones later
# Finally, will separate features from the target data
music_df_hits = music_df0.loc[music_df0['hit'] == 1]
music_df_nohits = music_df0.loc[music_df0['hit'] == 0]
music_df_na = music_df0.loc[music_df0['hit'].isnull()]

# clean all the data frames: main, hits, no hits, unlabelled
print(music_df0.head())
music_df = clean_data(music_df0)
music_df_hits = clean_data(music_df_hits)
music_df_nohits = clean_data(music_df_nohits)
music_df_test_set = clean_data(music_df_na)

# scale and set up the main data set
music_df.dropna(inplace=True)
music_df = music_df.reset_index(drop=True)
X_base = music_df.drop(['hit'], axis=1)
X = scale_data(X_base)
y = music_df['hit']

# scale unlabeled data set
music_df_test_set = scale_data(music_df_test_set)


# ## EDA ## #
# I started by comparing the stats for the hits and nonhits data sets
# I also tried to use a clustering algo and plots to see if there's a clear separation between hits/non hits (2 classes)
# I chose to do the mini batch version, due to size of data and easier computation,
# I didn't quite get the results I wanted with the clusters
# If I had more time, I would probably explore my plotting methods a little more in depth
# However the centroids were fairly far apart, so there could possibly be some separation between the two classes
hits_stat_list = get_stats(music_df_hits)
nohits_stat_list = get_stats(music_df_nohits)
print(hits_stat_list, nohits_stat_list)

''' 
Hits: 
[{'danceability': ['mean: 0.6014738972989006', 'std: 0.1513359513602319', 'var: 0.022902570174106473', 'min: 0.0', 'max: 0.988']}, 
{'energy': ['mean: 0.6242159988127037', 'std: 0.19801756578504526', 'var: 0.03921095635943472', 'min: 0.0181', 'max: 0.997']}, 
{'key': ['mean: 5.235500148411992', 'std: 3.5627559956371755', 'var: 12.693230284448642', 'min: 0', 'max: 11']}, 
{'loudness': ['mean: -8.699173701394992', 'std: 3.6070892556996683', 'var: 13.011092898583987', 'min: -28.03', 'max: -0.716']}, 
{'mode': ['mean: 0.7309587414663105', 'std: 0.44347461595808146', 'var: 0.19666973499916784', 'min: 0', 'max: 1']}, 
{'speechiness': ['mean: 0.06926384683882444', 'std: 0.07623737191299736', 'var: 0.005812136876200678', 'min: 0.0', 'max: 0.95']}, 
{'acousticness': ['mean: 0.28098553173226654', 'std: 0.27424203996317253', 'var: 0.07520869648316232', 'min: 2.32e-06', 'max: 0.992']}, 
{'instrumentalness': ['mean: 0.03018668445889007', 'std: 0.12865233098675932', 'var: 0.016551422268326673', 'min: 0.0', 'max: 0.982']}, 
{'liveness': ['mean: 0.1920253190857824', 'std: 0.16211473267836485', 'var: 0.026281186551377695', 'min: 0.013', 'max: 0.999']}, 
{'valence': ['mean: 0.6099101810626297', 'std: 0.2361709188223831', 'var: 0.05577670289740867', 'min: 0.0', 'max: 0.991']}, 
{'tempo': ['mean: 120.29440273078093', 'std: 27.729623278056845', 'var: 768.9320071429521', 'min: 0.0', 'max: 241.009']}, 
{'duration_ms': ['mean: 225525.8533689522', 'std: 65112.106706300765', 'var: 4239586439.732697', 'min: 74000', 'max: 1561000']}, 
{'time_signature': ['mean: 3.938082517067379', 'std: 0.30727930139089565', 'var: 0.0944205690632769', 'min: 0', 'max: 5']}, 
{'chorus_hit': ['mean: 39.17506287681794', 'std: 17.09185304017138', 'var: 292.1314403468156', 'min: 13.11714', 'max: 219.63624']}, 
{'sections': ['mean: 10.15897892549718', 'std: 2.8798080858946493', 'var: 8.293294611584203', 'min: 3', 'max: 64']}]

No-Hits: 
[{'danceability': ['mean: 0.478465639296892', 'std: 0.18068286920408982', 'var: 0.03264629922382223', 'min: 0.0576', 'max: 0.978']}, 
{'energy': ['mean: 0.5344021525403325', 'std: 0.2901078553508343', 'var: 0.08416256773626059', 'min: 0.000251', 'max: 1.0']}, 
{'key': ['mean: 5.186612087647484', 'std: 3.509162242660586', 'var: 12.314219645314674', 'min: 0', 'max: 11']}, 
{'loudness': ['mean: -11.759998374668873', 'std: 6.241679955399893', 'var: 38.95856866564081', 'min: -49.253', 'max: 3.744']}, 
{'mode': ['mean: 0.6557909944618349', 'std: 0.4751239366546781', 'var: 0.22574275518223857', 'min: 0', 'max: 1']}, 
{'speechiness': ['mean: 0.07645830122802846', 'std: 0.094272822627219', 'var: 0.008887365086103096', 'min: 0.0223', 'max: 0.96']}, 
{'acousticness': ['mean: 0.44954928778413245', 'std: 0.3748793522006691', 'var: 0.1405345287063933', 'min: 0.0', 'max: 0.996']}, 
{'instrumentalness': ['mean: 0.279189764699615', 'std: 0.37002710641895903', 'var: 0.13692005948478764', 'min: 0.0', 'max: 0.999']}, 
{'liveness': ['mean: 0.21188727425957182', 'std: 0.18331352906914838', 'var: 0.03360384993978551', 'min: 0.0146', 'max: 0.99']}, 
{'valence': ['mean: 0.4756786046231625', 'std: 0.27923706509349916', 'var: 0.07797333852203109', 'min: 0.0', 'max: 0.996']}, 
{'tempo': ['mean: 118.33267776306302', 'std: 30.409218543787038', 'var: 924.7205724438014', 'min: 31.988', 'max: 241.423']}, 
{'duration_ms': ['mean: 243868.5889718276', 'std: 156448.9141070844', 'var: 24476262725.28587', 'min: 15000', 'max: 4170000']}, 
{'time_signature': ['mean: 3.8493859860341924', 'std: 0.5119229646442115', 'var: 0.2620651217301186', 'min: 0', 'max: 5']}, 
{'chorus_hit': ['mean: 40.883971287021225', 'std: 20.624116475016105', 'var: 425.35418037503075', 'min: 0.0', 'max: 433.182']}, 
{'sections': ['mean: 10.771069106669877', 'std: 6.293956859656559', 'var: 39.613892951217856', 'min: 0', 'max: 169']}]
'''

kmeans = MiniBatchKMeans(n_clusters=2, batch_size=2048, max_iter=250, tol=0.0001).fit(X)
centroids = kmeans.cluster_centers_

plt.figure(0)
plt.scatter(X.iloc[:, 0], y)
plt.xlabel('Features')
plt.ylabel('Hit')
plt.savefig('plots\\features.png')
# plt.show()

plt.figure(6)
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, marker='*', color='black')
plt.savefig('plots\\centroids.png')
# plt.show()

# ## Feature analysis ## #
# Using all features in the plots didn't show clear patterns,
# but there seemed to be a slight shift when plotting danceability to hits,
# so I tried to see if any other plots showed trends
# After this, I tried two methods to see if I could determine which features were most important
# The second method, using SKL's Percentile provided a clearer separation of features than the RF
make_plots(X, y)

forest = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0)
forest.fit(X, y)
''' 
feat_importances = [0.05514875 0.06180678 0.08680834 0.07188301 0.02781902 0.06624802
 0.00810917 0.06590953 0.08802874 0.12571001 0.04965432 0.06213206
 0.05147428 0.06229211 0.0049293  0.04821463 0.03278136 0.03105057]
 #9 and #10 seem to be the highest with 3 and 4 following (Acoustic, instrumentalness, danceability, energy)
'''

X_feat = SelectPercentile(chi2, percentile=30).fit(X, y).get_support().tolist()
'''
list of scores, with list of bool for who made the top six (top 30th percentile)
[2.51708012e+00 3.84346911e-01 *2.02987717e+02* *9.89472049e+01*, 3.06605916e-01 3.17920813e+01 5.71455206e+01 
5.33257119e+00, *5.48595496e+02* *2.83728737e+03* 1.39473494e+01 2.34937898e+02, 9.16214480e-01 2.59153865e+00 
2.89799049e+00 1.39443597e+00, 1.53612878e+00 3.15476628e-01]
[False, False, True, True, False, False, True, False, True, True, False, True, False, False, False, False, False, False]
['danceability', 'energy', 'mode', 'acousticness', 'instrumentalness', 'valence']
'''

X, final_features = get_final_feat(X, X_feat, features)


# ## BUILDING THE MODEL ## #
# not sure which model to use, so will try grid search to test a few with a few diff parameters
# Decided to compare random forest with logistic regression since this is basically a binary classification
# Random Forest in general did better, so I added a few more variations of parameters to test out to improve accuracy
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

models = [RandomForestClassifier(), LogisticRegression()]
params = [{'n_estimators': [100, 250, 500, 1200], 'max_depth': [6, 10], 'criterion': ['gini', 'entropy'], 'random_state': [0]},
          {'penalty': ['none', 'l2'], 'random_state': [0]}]

final_models = get_models(models, params, X_train, y_train)

best_score = 0
ideal_model = ''

for model in final_models:
    if model[2] > best_score:
        best_score = model[2]
        ideal_model = model
    else:
        continue

best_model = RandomForestClassifier(n_estimators=1600, criterion='entropy', max_depth=12, random_state=0)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(x_test)
best_score = best_model.score(x_test, y_test)

print(confusion_matrix(y_test, y_pred))
print(best_score)

# ## RUNNING NEW PREDICTIONS ## #
for col, _ in music_df_test_set.iteritems():
    if col not in final_features:
        music_df_test_set.drop([col], axis=1, inplace=True)

new_pred = best_model.predict(music_df_test_set)
music_df_na['hit'] = new_pred
music_df_na.to_excel('Predictions\\new_preds.xlsx')

# ## SUMMARY ## #
''' 
After reviewing the data initially and comparing the stats between hits and no hits, we did see some clear differences.
Danceability and energy were higher on average in hit songs, with less spread away from the mean. I noticed that mode 
also had a higher mean in hit songs. No-hit songs scored significantly higher in acousticness and instrumentalness. Hit 
songs had higher valence and tempo. Though initial plots of the data didn't really show much, that may have been due to
some error on my end and with more time may have been fleshed out. 

However, based on the actual stats it does seem fair to say that some features were better indicators of hit songs than
others, and this was confirmed using some of scikit-learn's feature extraction methods. Instrumentaility, acousticness, 
danceability, and mode all went into the final list of features used in the model. 

The final result of the model after GridSearch and additional fine tuning was 76.3% accuracy. Based on the confusion 
matrix, there was an issue of false positives (predicting hits that were actually not hits, 1109) but there was a high
number of true positives (2273) and true negatives (2833), enough that I think with additional data, features, and 
tuning, we could create a model that could reasonably predict whether a song is going to be a hit.
'''