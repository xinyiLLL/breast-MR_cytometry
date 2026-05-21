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
from scipy import stats
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

        print(f"Train set: {self.X.shape}, Label distribution: {np.bincount(self.y)}")

    def preprocess_train(self):
        self.mm = MinMaxScaler()
        X_mm = self.mm.fit_transform(self.X)

        if self.feature_type == "all":
            self.std = StandardScaler()
            X_std = self.std.fit_transform(X_mm)

            self.pca = PCA(n_components=0.999, random_state=42)
            self.X_train_final = self.pca.fit_transform(X_std)
            print(f"Dimensions after PCA: {self.X_train_final.shape[1]}")
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

        print(f"External validation set: {X.shape}, Label distribution: {np.bincount(y)}")
        return X, y

    def get_models_and_params(self):
        models = {
            "GaussianNB": GaussianNB(),
            # "SVM": SVC(probability=True, random_state=42, class_weight="balanced"),
            # "RandomForest": RandomForestClassifier(random_state=42, class_weight="balanced"),
        }
        
        params = {
            'GaussianNB': {
                'var_smoothing': np.logspace(-20, -1, 10)
            },
            # 'SVM': {
            #     'C': np.linspace(0.5,1,21),
            #     'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            #     'kernel': ['rbf', 'linear']
            # },
            # 'RandomForest': {
            #     'n_estimators': [50, 100, 150, 200],
            #     'max_depth': [7,8,9,10,11,12,13],
            #     'min_samples_split': [3, 5, 7, 9, 11],
            #     'min_samples_leaf':[3, 5, 7, 9, 11],
            # },
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

    def bootstrap_auc(self, y_true, y_prob, n_bootstraps=1000, random_state=42):
        rng = np.random.RandomState(random_state)
        bootstrapped_scores = []
        original_auc = roc_auc_score(y_true, y_prob)

        for _ in range(n_bootstraps):
            indices = rng.randint(0, len(y_prob), len(y_prob))
            if len(np.unique(y_true[indices])) < 2:
                continue
            
            score = roc_auc_score(y_true[indices], y_prob[indices])
            bootstrapped_scores.append(score)

        if not bootstrapped_scores:
            return original_auc, original_auc, original_auc

        lower_bound = np.percentile(bootstrapped_scores, 2.5)
        upper_bound = np.percentile(bootstrapped_scores, 97.5)

        return original_auc, lower_bound, upper_bound

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
                print(f"{name} Best params: {best_params}")

                model = clone(base_model).set_params(**best_params)
                model.fit(X_tr, y_tr)

                thr = self.youden_threshold(y_tr, self.get_prob(model, X_tr))

                y_va_prob = self.get_prob(model, X_va)
                y_va_pred = (y_va_prob >= thr).astype(int)

                cv_results[name].append({
                    "y_true": y_va.copy(),
                    "y_prob": y_va_prob.copy(),
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
                        "y_true": y_ext.copy(),
                        "y_prob": y_ext_prob.copy(),
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

    def format_metrics_table(self, results, dataset_name):
        table_rows = []
        
        for model_name, model_results in results.items():
            if len(model_results) == 0:
                continue

            fold_aucs = [r.get("auc", 0) for r in model_results]
            auc_mean = np.mean(fold_aucs)
            
            if "train_cv" in dataset_name:
                y_true_all = np.concatenate([r["y_true"] for r in model_results])
                y_prob_all = np.concatenate([r["y_prob"] for r in model_results])
            else:
                y_true_all = model_results[0]["y_true"]
                y_prob_all = np.mean([r["y_prob"] for r in model_results], axis=0)
            
            _, auc_lower, auc_upper = self.bootstrap_auc(y_true_all, y_prob_all)
            
            accuracies = [r["accuracy"] for r in model_results]
            recalls = [r["recall"] for r in model_results]
            specificities = [r["specificity"] for r in model_results]
            f1s = [r["f1"] for r in model_results]
            
            acc_mean = np.mean(accuracies)
            sen_mean = np.mean(recalls)
            spe_mean = np.mean(specificities)
            f1_mean = np.mean(f1s)
            
            print(f"{model_name}:")
            print(f"  AUC: {auc_mean:.3f} (CI: {auc_lower:.3f}-{auc_upper:.3f})")
            print(f"  Accuracy: {acc_mean:.3f} ± {np.std(accuracies):.3f}")
            print(f"  Sensitivity: {sen_mean:.3f} ± {np.std(recalls):.3f}")
            print(f"  Specificity: {spe_mean:.3f} ± {np.std(specificities):.3f}")
            print(f"  F1: {f1_mean:.3f} ± {np.std(f1s):.3f}")
            
            row = {
                "Model": model_name,
                "AUC (95% CI)": f"{auc_mean:.3f} ({auc_lower:.3f}-{auc_upper:.3f})",
                "accuracy": f"{acc_mean:.3f}",
                "sensitivity": f"{sen_mean:.3f}",
                "specificity": f"{spe_mean:.3f}",
                "F1": f"{f1_mean:.3f}"
            }
            table_rows.append(row)
        
        df = pd.DataFrame(table_rows)
        if df.empty:
            return None

        column_order = ["Model", "AUC (95% CI)", "accuracy", "sensitivity", "specificity", "F1"]
        df = df[column_order]
        
        os.makedirs("results", exist_ok=True)
        filename = f"results/{dataset_name}_{self.feature_type}_metrics.xlsx"
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f"\nSaved to: {filename}")
        
        return df

    def print_formatted_table(self, df, title):
        print("\n" + "=" * 80)
        print(f"{title}")
        print("=" * 80)
        
        header = f"{'Model':<15} | {'AUC (95% CI)':<20} | {'accuracy':<10} | {'sensitivity':<12} | {'specificity':<12} | {'F1':<8}"
        print(header)
        print("-" * 80)
        
        for _, row in df.iterrows():
            line = f"{row['Model']:<15} | {row['AUC (95% CI)']:<20} | {row['accuracy']:<10} | {row['sensitivity']:<12} | {row['specificity']:<12} | {row['F1']:<8}"
            print(line)

    def summarize(self, results, title, dataset_name):
        print("\n" + "=" * 60)
        print(title)
        
        df = self.format_metrics_table(results, dataset_name)
        if df is not None:
            self.print_formatted_table(df, title)
        return df
    
    def run(self):
        self.load_data()
        self.preprocess_train()
        X_ext, y_ext = self.load_external()

        cv_res, ext_res = self.cross_validation(X_ext, y_ext)

        cv_df = self.summarize(cv_res, "Internal 5-Fold CV Results", "train_cv")
        
        if X_ext is not None:
            ext_df = self.summarize(ext_res, "External Test Set Results", "external_test")
           

if __name__ == "__main__":
    feature_type = "adc"
    train_path = "../mean_feature/train.xlsx"
    external_path = f"../mean_feature/mean.xlsx"
    clf = RadiomicsClassifier(
        train_path=train_path,
        external_path=external_path,
        feature_type=feature_type
    )
    clf.run()
