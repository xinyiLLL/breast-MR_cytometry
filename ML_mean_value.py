import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.base import clone
import warnings
import os
        

from sklearn.svm import SVC

warnings.filterwarnings("ignore")


class RadiomicsClassifier:
    def __init__(self, train_path, external_path=None, feature_type="all"):
        self.train_path = train_path
        self.external_path = external_path
        self.feature_type = feature_type.lower()

    def load_data(self):
        self.train_df = pd.read_excel(self.train_path)

        feature_map = {
            "adc": ["ADC_N_1", "ADC_N_2", "ADC_PGSE"],
            "micro": ["fit_d", "fit_vin", "fit_cell", "fit_Dex"],
            "all": ["ADC_N_1", "ADC_N_2", "ADC_PGSE",
                    "fit_d", "fit_vin", "fit_cell", "fit_Dex"]
        }

        self.features = feature_map[self.feature_type]
        self.X = self.train_df[self.features]
        self.y = self.train_df["label"].values

        print(f"training set: {self.X.shape}, label: {np.bincount(self.y)}")

    def preprocess_train(self):
        self.mm = MinMaxScaler()
        X_mm = self.mm.fit_transform(self.X)

        if self.feature_type == "all":
            self.std = StandardScaler()
            X_std = self.std.fit_transform(X_mm)

            self.pca = PCA(n_components=0.97, random_state=42)
            self.X_train_final = self.pca.fit_transform(X_std)
            print(f"dimension after PCA: {self.X_train_final.shape[1]}")

            n_components = self.pca.n_components_
            for i in range(n_components):
                component_weights = self.pca.components_[i]
                variance_ratio = self.pca.explained_variance_ratio_[i]
                
                expression_parts = []
                for j, weight in enumerate(component_weights):
                    if abs(weight) > 0.001: 
                        feature_name = self.features[j] if j < len(self.features) else f'X{j+1}'
                        expression_parts.append(f"{weight:.2f}*{feature_name}")
                
                if expression_parts:
                    expression = " + ".join(expression_parts)
                    print(f"PC{i+1} (explained variance ratio: {variance_ratio:.3f}):")
                    print(f"  {expression}")
        else:
            self.X_train_final = X_mm
        
    def load_external(self):
        if self.external_path is None:
            return None, None

        df = pd.read_excel(self.external_path)
        X = df[self.features]
        y = df["label"].values

        X = self.mm.transform(X)

        if self.feature_type == "all":
            X = self.std.transform(X)
            X = self.pca.transform(X)

        print(f"external test set: {X.shape}, label: {np.bincount(y)}")
        return X, y

    def get_models_and_params(self):
        models = {
            "GaussianNB": GaussianNB(),
            "SVM": SVC(probability=True, random_state=42, class_weight="balanced"),
            "LR": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "RandomForest": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
            "KNN": KNeighborsClassifier()
        }
        # adc
        # params = {
        #     'GaussianNB': {
        #         'var_smoothing': [1e-8, 1e-4, 0.1, 1, 10, 100]
        #     },
        #     'SVM': {
        #         'C': np.linspace(10,100,10),
        #         'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        #         'kernel': ['rbf', 'linear']
        #     },
        #     'LR': {
        #         'C': [0.01, 0.1, 1, 10, 100],
        #         'solver': ['liblinear', 'lbfgs','sparse_cg']
        #     },
        #     'RandomForest': {
        #         'n_estimators': [50, 100, 150, 200],
        #         'max_depth': [5, 7, 9, 11]
        #     },
        #     'GradientBoosting': {
        #         'n_estimators': [50, 100, 150, 200, 250],
        #         'learning_rate': [0.01, 0.1, 0.5, 1],
        #         'max_depth': [3, 4,5,6, 7, 8,9]
        #     },
        #     'KNN': {
        #         'n_neighbors': [1, 3, 5, 7, 9, 11, 13],
        #         'weights': ['uniform', 'distance'],
        #         'metric': ['euclidean', 'manhattan','minkowski'],
        #         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        #     }
        # }
        
        # micro
        # params = {
        #         'GaussianNB': {
        #         'var_smoothing': np.logspace(-9, -1, 10)
        #         },
        #         'SVM': {
        #             'C':[0.1, 1, 10, 100],
        #             'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        #             'kernel': ['rbf', 'linear']
        #         },
        #         'LR': {
        #             'C': [0.01, 0.1, 1, 10, 100],
        #             'solver': ['liblinear', 'lbfgs','sparse_cg']
        #         },
        #         'RandomForest': {
        #             'n_estimators': [100, 150, 200, 250],
        #             'max_depth': [7,9,11],
        #         },
        #         'GradientBoosting': {
        #             'n_estimators': [50, 100, 150, 200, 250],
        #             'learning_rate': [0.01, 0.1, 0.5, 1],
        #             'max_depth': [3, 5, 7, 9]
        #         },
        #         'KNN': {
        #             'n_neighbors': [1, 3, 5, 7, 9, 11],
        #             'weights': ['uniform', 'distance'],
        #             'metric': ['euclidean', 'manhattan']
        #         }
        #     }

        # all 0.97
        params = {
            'GaussianNB': {
                'var_smoothing': [1e-6,1e-3,1,100]
            },
            'SVM': {
                'C': np.arange(1, 10, 2), 
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear','poly']
            },
            'LR': {
                'C': [0.001,0.01,0.5,1,10],
                'solver': ['liblinear', 'lbfgs','sparse_cg'],
                'penalty' : ['l1', 'l2']
            },
            'RandomForest': {
                'n_estimators': [200, 250,300,350],
                'max_depth': [7,9,11,13],
                'min_samples_leaf':[3, 5, 7],
                'min_samples_split': [5, 7, 9, 11]
            },
            'GradientBoosting': {
                'n_estimators': [25,50, 75,100],
                'learning_rate': [0.001,0.05, 0.1, 1],
                'max_depth': [3,5,7,9,11]
            },
            'KNN': {
                'n_neighbors': [9,11,13],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
        return models, params

    def specificity(self, y_true, y_pred):
        tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp + 1e-8)

    def youden_threshold(self, y_true, y_prob):
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        return thr[np.argmax(tpr - fpr)]

    def get_prob(self, model, X):
        return model.predict_proba(X)[:, 1]

    def cross_validation(self, X_ext=None, y_ext=None):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        models, params = self.get_models_and_params()

        cv_results = {m: [] for m in models}
        ext_results = {m: [] for m in models}

        for fold, (tr, va) in enumerate(skf.split(self.X_train_final, self.y), 1):
            print(f"\n===== Fold {fold} =====")

            X_tr = self.X_train_final[tr]
            X_va = self.X_train_final[va]
            y_tr, y_va = self.y[tr], self.y[va]

            for name, base_model in models.items():
                gs = GridSearchCV(
                    base_model,
                    params[name],
                    scoring="roc_auc",
                    cv=5,
                    n_jobs=4
                )
                gs.fit(X_tr, y_tr)

                best_params = gs.best_params_
                print(f"{name} best params: {best_params}")

                model = clone(base_model).set_params(**best_params)
                model.fit(X_tr, y_tr)

                thr = self.youden_threshold(y_tr, self.get_prob(model, X_tr))

                y_va_prob = self.get_prob(model, X_va)
                y_va_pred = (y_va_prob >= thr).astype(int)

                cv_results[name].append({
                    "auc": roc_auc_score(y_va, y_va_prob),
                    "accuracy": accuracy_score(y_va, y_va_pred),
                    "recall": recall_score(y_va, y_va_pred),
                    "specificity": self.specificity(y_va, y_va_pred),
                    "f1": f1_score(y_va, y_va_pred)
                })

                if X_ext is not None:
                    y_ext_prob = self.get_prob(model, X_ext)
                    y_ext_pred = (y_ext_prob >= thr).astype(int)
                    
                    ext_results[name].append({
                        "auc": roc_auc_score(y_ext, y_ext_prob),
                        "accuracy": accuracy_score(y_ext, y_ext_pred),
                        "recall": recall_score(y_ext, y_ext_pred),
                        "specificity": self.specificity(y_ext, y_ext_pred),
                        "f1": f1_score(y_ext, y_ext_pred),
                        "probabilities": y_ext_prob,  
                        "predictions": y_ext_pred,    
                        "threshold": thr,             
                        "fold": fold                  
                    })

        return cv_results, ext_results

    def summarize(self, results, title):
        print("\n" + "=" * 60)
        print(title)
        for m, res in results.items():
            aucs = [r["auc"] for r in res]
            print(f"{m:10s} | AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    def run(self):
        self.load_data()
        self.preprocess_train()
        X_ext, y_ext = self.load_external()

        cv_res, ext_res = self.cross_validation(X_ext, y_ext)

        self.summarize(cv_res, "5-fold internal validation")
        if X_ext is not None:
            self.summarize(ext_res, "External test set")
            
            for model_name, results in ext_res.items():
                print(f"\n{model_name}:")
                for fold_result in results:
                    fold_num = fold_result.get('fold', 0)
                    print(f"  fold{fold_num}: AUC={fold_result['auc']:.3f}, "
                          f"ACC={fold_result['accuracy']:.3f}, "
                          f"SEN={fold_result['recall']:.3f}, "
                          f"SPE={fold_result['specificity']:.3f}, "
                          f"F1={fold_result['f1']:.3f}, ")


if __name__ == "__main__":
    feature_type = "all"   # adc / micro / all
    train_path = "../mean_feature/QH.xlsx"
    external_path = "../mean_feature/external.xlsx"
    clf = RadiomicsClassifier(
        train_path=train_path,
        external_path=external_path,
        feature_type=feature_type
    )
    clf.run()
