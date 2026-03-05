import datetime
import logging
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.base import clone
import warnings
import os

warnings.filterwarnings("ignore")


class RadiomicsClassifier:

    def __init__(self, train_path, feature_type, external_path=None):
        self.train_path = train_path
        self.external_path = external_path
        self.feature_type = feature_type
        self.alpha_map = {'ADC': 0.03, 'micro': 0.02, 'all': 0.025}
        self.alpha = self.alpha_map[feature_type]

    def setup_logging(self):
        log_dir = f"logs/{self.feature_type}"
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/validation_{ts}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger()
        return log_file

    def log(self, msg):
        self.logger.info(msg)
        
    def load_train_data(self):
        data = pd.read_excel(self.train_path)
        self.X = data.drop(columns=["Patient", "label"])
        self.y = data["label"].values
        self.log(f"training set: {self.X.shape}, label: {np.bincount(self.y)}")

    # ============ LASSO + PCA============
    def preprocess_train(self):
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(self.X)

        lasso = Lasso(alpha=self.alpha, max_iter=2000, random_state=42)
        lasso.fit(X_scaled, self.y)
        self.selector = SelectFromModel(lasso, prefit=True)
        X_sel = self.selector.transform(X_scaled)

        self.std = StandardScaler()
        X_std = self.std.fit_transform(X_sel)

        self.pca = PCA(n_components=0.95, random_state=42)
        self.X_reduced = self.pca.fit_transform(X_std)

        self.log(f"dimension after LASSO & PCA: {self.X_reduced.shape[1]}")

    def load_external(self):
        if self.external_path is None:
            return None, None

        external_data = pd.read_excel(self.external_path)

        X = external_data.drop(columns=["Patient", "label"])
        y = external_data["label"].values

        X = self.scaler.transform(X)
        X = self.selector.transform(X)
        X = self.std.transform(X)
        X = self.pca.transform(X)

        self.log(f"external test set: {X.shape}, label: {np.bincount(y)}")
        return X, y

    def save_detailed_results(self, int_results, ext_results):
        result_dir = "results/" + self.feature_type + "/"
        os.makedirs(result_dir, exist_ok=True)
        
        int_data = []
        for model_name, fold_results in int_results.items():
            for fold_result in fold_results:
                int_data.append({
                    'Model': model_name,
                    'Fold': fold_result['fold'],
                    'AUC': fold_result['auc'],
                    'Accuracy': fold_result['accuracy'],
                    'Sensitivity': fold_result['sensitivity'],
                    'Specificity': fold_result['specificity'],
                    'F1': fold_result['f1'],
                    'Threshold': fold_result['threshold']
                })
        
        if int_data:
            int_df = pd.DataFrame(int_data)
            int_filename = os.path.join(result_dir, "internal_validation_detailed.csv")
            int_df.to_csv(int_filename, index=False, encoding='utf-8-sig')
            self.log(f"internal validation detailed results saved to: {int_filename}")
        
        if ext_results and any(ext_results.values()):
            ext_data = []
            for model_name, fold_results in ext_results.items():
                for fold_result in fold_results:
                    ext_data.append({
                        'Model': model_name,
                        'Fold': fold_result.get('fold', 0),
                        'AUC': fold_result['auc'],
                        'Accuracy': fold_result['acc'],
                        'Sensitivity': fold_result['sen'],
                        'Specificity': fold_result['spe'],
                        'F1': fold_result['f1']
                    })
            
            ext_df = pd.DataFrame(ext_data)
            ext_filename = os.path.join(result_dir, "external_validation_detailed.csv")
            ext_df.to_csv(ext_filename, index=False, encoding='utf-8-sig')
            self.log(f"external test detailed results saved to: {ext_filename}")
    def get_models(self):
        return {
            "GaussianNB": (
                GaussianNB(),
                {'var_smoothing': np.logspace(-9, -1, 10)}
            ),
            "SVM": (
                SVC(probability=True, random_state=42, class_weight="balanced"),
                {'C':[0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear']}
            ),
            "LR": (
                LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
                {'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs','sparse_cg']}
            ),
            "RF": (
                RandomForestClassifier(class_weight="balanced", random_state=42),
                {'n_estimators': [50, 100, 150, 200, 250],
                'max_depth': [1, 3, 5, 7, None],
                'min_samples_split': [1, 3, 5, 7, 10, 12]}
            ),
            "GB": (
                GradientBoostingClassifier(random_state=42),
                {'n_estimators': [50, 100, 150, 200, 250],
                'learning_rate': [0.01, 0.1, 0.5, 1],
                'max_depth': [3, 5, 7, 9]}
            ),
            "KNN": (
                KNeighborsClassifier(),
                { 
                'n_neighbors': [1, 3, 5, 7, 9, 11, 13],# adc micro
                # 'n_neighbors': [1, 3, 5, 7, 9, 11], # all
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']}
            )
        }

    def specificity(self, y_true, y_pred):
        tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)

    def youden_threshold(self, y_true, y_prob):
        fpr, tpr, th = roc_curve(y_true, y_prob)
        return th[np.argmax(tpr - fpr)]

    def prob(self, model, X):
        return model.predict_proba(X)[:, 1]

    def cross_validate(self, X_ext=None, y_ext=None):
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        models = self.get_models()

        int_results = {m: [] for m in models} 
        ext_results = {m: [] for m in models} 

        for fold, (tr, va) in enumerate(skf.split(self.X_reduced, self.y), 1):
            self.log(f"\n===== Fold {fold} =====")

            X_tr, X_va = self.X_reduced[tr], self.X_reduced[va]
            y_tr, y_va = self.y[tr], self.y[va]

            for name, (base_model, param_grid) in models.items():
                gs = GridSearchCV(
                    base_model,
                    param_grid,
                    scoring="roc_auc",
                    cv=5,
                    n_jobs=1
                )
                gs.fit(X_tr, y_tr)

                best_params = gs.best_params_

                model = clone(base_model).set_params(**best_params)
                model.fit(X_tr, y_tr)

                thr = self.youden_threshold(y_tr, self.prob(model, X_tr))

                y_va_prob = self.prob(model, X_va)
                y_va_pred = (y_va_prob >= thr).astype(int)
                metrics = {
                    "fold": fold,
                    "auc": roc_auc_score(y_va, y_va_prob) if len(np.unique(y_va)) > 1 else 0.0,
                    "accuracy": accuracy_score(y_va, y_va_pred),
                    "sensitivity": recall_score(y_va, y_va_pred, zero_division=0),
                    "specificity": self.specificity(y_va, y_va_pred),
                    "f1": f1_score(y_va, y_va_pred, zero_division=0),
                    "threshold": thr
                }

                int_results[name].append(metrics)

                self.log(f"{name:15s} | in-va | AUC={metrics['auc']:.3f} | ACC={metrics['accuracy']:.3f} | "
                                f"SEN={metrics['sensitivity']:.3f} | SPE={metrics['specificity']:.3f} | F1={metrics['f1']:.3f}")
                    
                if X_ext is not None:
                    y_ext_prob = self.prob(model, X_ext)
                    y_ext_pred = (y_ext_prob >= thr).astype(int)

                    ext_results[name].append({
                        "auc": roc_auc_score(y_ext, y_ext_prob),
                        "acc": accuracy_score(y_ext, y_ext_pred),
                        "sen": recall_score(y_ext, y_ext_pred),
                        "spe": self.specificity(y_ext, y_ext_pred),
                        "f1": f1_score(y_ext, y_ext_pred)
                    })

        return int_results, ext_results

    def summarize(self, int_results, ext_results, title="result"):
        self.log(f"\n{'='*60}")
        self.log(f"{title}")
        self.log(f"{'='*60}")
        
        self.log("\ninternal validation")
        for m, res in int_results.items():
            aucs = [r["auc"] for r in res]
            accs = [r["accuracy"] for r in res]
            sens = [r["sensitivity"] for r in res]
            specs = [r["specificity"] for r in res]
            f1s = [r["f1"] for r in res]
            
            self.log(f"{m:12s} | AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}")
            self.log(f"{' ':12s} | ACC={np.mean(accs):.3f}±{np.std(accs):.3f}")
            self.log(f"{' ':12s} | SEN={np.mean(sens):.3f}±{np.std(sens):.3f}")
            self.log(f"{' ':12s} | SPE={np.mean(specs):.3f}±{np.std(specs):.3f}")
            self.log(f"{' ':12s} | F1 ={np.mean(f1s):.3f}±{np.std(f1s):.3f}")
        
        if ext_results and any(ext_results.values()):
            self.log("\neternal test")
            for m, res in ext_results.items():
                aucs = [r["auc"] for r in res]
                accs = [r["acc"] for r in res]
                sens = [r["sen"] for r in res]
                specs = [r["spe"] for r in res]
                f1s = [r["f1"] for r in res]
                
                self.log(f"{m:12s} | AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}")
                self.log(f"{' ':12s} | ACC={np.mean(accs):.3f}±{np.std(accs):.3f}")
                self.log(f"{' ':12s} | SEN={np.mean(sens):.3f}±{np.std(sens):.3f}")
                self.log(f"{' ':12s} | SPE={np.mean(specs):.3f}±{np.std(specs):.3f}")
                self.log(f"{' ':12s} | F1 ={np.mean(f1s):.3f}±{np.std(f1s):.3f}")

    def run(self):
        log_file = self.setup_logging()
        self.load_train_data()
        self.preprocess_train()
        X_ext, y_ext = self.load_external()

        int_res, ext_res = self.cross_validate(X_ext, y_ext)
        
        if X_ext is not None:
            self.summarize(int_res, ext_res, "5-fold CV result (internal + external test set)")
        else:
            self.summarize(int_res, {}, "5-fold CV result (internal validation set)")
        
        self.save_detailed_results(int_res, ext_res)


if __name__ == "__main__":
    feature_type = "micro"

    train_path = f"../radiomics_feature_normalize/QH/{feature_type}+mean.xlsx"
    external_path = f"../radiomics_feature_normalize/external/{feature_type}+mean.xlsx"

    clf = RadiomicsClassifier(
        train_path=train_path,
        feature_type=feature_type,
        external_path=external_path
    )
    clf.run()
