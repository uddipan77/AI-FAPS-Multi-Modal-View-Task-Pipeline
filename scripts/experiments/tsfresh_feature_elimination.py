from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier

train_val_path = '/home/hpc/iwfa/iwfa028h/work_data/data/LinearWindingDataset/fabian/dataset_linear_winding_multimodal/train_val/tsfresh_force_features_coil_level_train_val_2024-01-16.CSV'

data = pd.read_csv(train_val_path)
data = data.dropna(axis=1, how='all')

X = data.drop(['label', 'coil_id'], axis=1)
y = data['label']

reg = XGBClassifier()
# cv = KFold(5)

rfecv = RFECV(
    estimator=reg,
    step=1,
    # cv=cv,
    scoring="f1",
    min_features_to_select=1,
    n_jobs=16,
)

rfecv.fit(X, y)

X_reduced = pd.DataFrame(rfecv.transform(X), columns=rfecv.get_feature_names_out())
X_reduced['coil_id'] = data['coil_id']
X_reduced['label'] = data['label']

print(f"Optimal number of features: {rfecv.n_features_}")