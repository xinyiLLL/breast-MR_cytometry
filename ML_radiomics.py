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
from scipy import stats

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
        self.log(f"Train set: {self.X.shape}, Label distribution: {np.bincount(self.y)}")

    def preprocess_train(self):
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(self.X)

        lasso = Lasso(alpha=self.alpha, max_iter=2000, random_state=42)
        lasso.fit(X_scaled, self.y)
        self.selector = SelectFromModel(lasso, prefit=True)
        X_sel = self.selector.transform(X_scaled)
        
        selected_indices = np.where(lasso.coef_ != 0)[0]
        feature_names = self.X.columns.tolist()
        self.selected_feature_names = [feature_names[i] for i in selected_indices]
        
        print(f"Number of selected features: {len(self.selected_feature_names)}")
        print(f"Selected features: {self.selected_feature_names}")

        self.std = StandardScaler()
        X_std = self.std.fit_transform(X_sel)

        self.pca = PCA(n_components=0.95, random_state=42)
        self.X_reduced = self.pca.fit_transform(X_std)

        print(f"Dimensions after PCA: {self.X_reduced.shape[1]}")
        print(f"Cumulative explained variance: {np.cumsum(self.pca.explained_variance_ratio_)}")

    def load_external(self):
        if self.external_path is None:
            return None, None

        data = pd.read_excel(self.external_path)
        X = data.drop(columns=["Patient", "label"])
        y = data["label"].values

        X = self.scaler.transform(X)
        X = self.selector.transform(X)
        
        X = self.std.transform(X)
        X = self.pca.transform(X)

        self.log(f"External validation set: {X.shape}, Label distribution: {np.bincount(y)}")
        return X, y

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
                'n_neighbors': [1, 3, 5, 7, 9, 11, 13],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']}
            )
        }
            
       
    def specificity(self, y_true, y_pred):
        tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp + 1e-8)

    def youden_threshold(self, y_true, y_prob):
        fpr, tpr, th = roc_curve(y_true, y_prob)
        return th[np.argmax(tpr - fpr)]

    def prob(self, model, X):
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

    def cross_validate(self, X_ext=None, y_ext=None):
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        models = self.get_models()

        internal_results = {m: [] for m in models}
        external_results = {m: [] for m in models}

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
                    n_jobs=4
                )
                gs.fit(X_tr, y_tr)

                best_params = gs.best_params_
                print(f"{name} Best params: {best_params}")
                model = clone(base_model).set_params(**best_params)
                model.fit(X_tr, y_tr)

                thr = self.youden_threshold(y_tr, self.prob(model, X_tr))

                y_va_prob = self.prob(model, X_va)
                y_va_pred = (y_va_prob >= thr).astype(int)
                
                internal_metrics = {
                    "fold": fold,
                    "y_true": y_va.copy(),
                    "y_prob": y_va_prob.copy(),
                    "auc": roc_auc_score(y_va, y_va_prob) if len(np.unique(y_va)) > 1 else 0.0,
                    "accuracy": accuracy_score(y_va, y_va_pred),
                    "sensitivity": recall_score(y_va, y_va_pred, zero_division=0),
                    "specificity": self.specificity(y_va, y_va_pred),
                    "f1": f1_score(y_va, y_va_pred, zero_division=0),
                    "threshold": thr
                }
                internal_results[name].append(internal_metrics)

                self.log(f"{name:15s} | AUC={internal_metrics['auc']:.3f} | ACC={internal_metrics['accuracy']:.3f} | "
                         f"SEN={internal_metrics['sensitivity']:.3f} | SPE={internal_metrics['specificity']:.3f} | F1={internal_metrics['f1']:.3f}")
                    
                if X_ext is not None:
                    y_ext_prob = self.prob(model, X_ext)
                    y_ext_pred = (y_ext_prob >= thr).astype(int)
                    
                    external_results[name].append({
                        "y_true": y_ext.copy(),
                        "y_prob": y_ext_prob.copy(),
                        "auc": roc_auc_score(y_ext, y_ext_prob) if len(np.unique(y_ext)) > 1 else 0.0,
                        "accuracy": accuracy_score(y_ext, y_ext_pred),
                        "sensitivity": recall_score(y_ext, y_ext_pred, zero_division=0),
                        "specificity": self.specificity(y_ext, y_ext_pred),
                        "f1": f1_score(y_ext, y_ext_pred, zero_division=0)
                    })

        return internal_results, external_results

    def print_formatted_table(self, df, title):
        print("\n" + "=" * 90)
        print(f"{title}")
        print("=" * 90)
        header = f"{'Model':<15} | {'AUC (95% CI)':<25} | {'accuracy':<10} | {'sensitivity':<12} | {'specificity':<12} | {'F1':<8}"
        print(header)
        print("-" * 90)
        for _, row in df.iterrows():
            line = f"{row['Model']:<15} | {row['AUC (95% CI)']:<25} | {row['accuracy']:<10} | {row['sensitivity']:<12} | {row['specificity']:<12} | {row['F1']:<8}"
            print(line)

    def format_metrics_table(self, results, dataset_name):
        
        table_rows = []
        all_models_metrics = {
            "auc": [],
            "acc": [],
            "sen": [],
            "spe": [],
            "f1": []
        }
        
        for model_name, model_results in results.items():
            if len(model_results) == 0:
                continue
            
            fold_aucs = [r["auc"] for r in model_results]
            auc_mean = np.mean(fold_aucs)
            
            if "internal" in dataset_name or "train" in dataset_name:
                y_true_all = np.concatenate([r["y_true"] for r in model_results])
                y_prob_all = np.concatenate([r["y_prob"] for r in model_results])
            else:
                y_true_all = model_results[0]["y_true"]
                y_prob_all = np.mean([r["y_prob"] for r in model_results], axis=0)
            
            _, auc_lower, auc_upper = self.bootstrap_auc(y_true_all, y_prob_all)
            
            accuracies = [r.get("accuracy", 0) for r in model_results]
            sensitivities = [r.get("sensitivity", 0) for r in model_results]
            specificities = [r.get("specificity", 0) for r in model_results]
            f1s = [r.get("f1", 0) for r in model_results]
            
            acc_mean = np.mean(accuracies)
            sen_mean = np.mean(sensitivities)
            spe_mean = np.mean(specificities)
            f1_mean = np.mean(f1s)
            
            all_models_metrics["auc"].append(auc_mean)
            all_models_metrics["acc"].append(acc_mean)
            all_models_metrics["sen"].append(sen_mean)
            all_models_metrics["spe"].append(spe_mean)
            all_models_metrics["f1"].append(f1_mean)
            
            row = {
                "Model": model_name,
                "AUC (95% CI)": f"{auc_mean:.3f} ({auc_lower:.3f}-{auc_upper:.3f})",
                "accuracy": f"{acc_mean:.3f}",
                "sensitivity": f"{sen_mean:.3f}",
                "specificity": f"{spe_mean:.3f}",
                "F1": f"{f1_mean:.3f}"
            }
            table_rows.append(row)
        
        if len(table_rows) == 0:
            return None, None
            
        df = pd.DataFrame(table_rows)
        column_order = ["Model", "AUC (95% CI)", "accuracy", "sensitivity", "specificity", "F1"]
        df = df[column_order]
        
        os.makedirs("AUC_results", exist_ok=True)
        filename = f"AUC_results/{dataset_name}_{self.feature_type}_metrics.xlsx"
        df.to_excel(filename, index=False, engine='openpyxl')
        self.log(f"\nSaved to: {filename}")
        
        return df, all_models_metrics

    def summarize(self, internal_results, external_results):
        if internal_results:
            internal_df, internal_metrics = self.format_metrics_table(internal_results, "internal_train_cv")
            if internal_df is not None:
                self.print_formatted_table(internal_df, "Internal 5-Fold CV Results")
                
                mean_auc = np.mean(internal_metrics["auc"])
                mean_acc = np.mean(internal_metrics["acc"])
                mean_sen = np.mean(internal_metrics["sen"])
                mean_spe = np.mean(internal_metrics["spe"])
                mean_f1  = np.mean(internal_metrics["f1"])
                num_models = len(internal_metrics["auc"])
                
                print("-" * 90)
                self.log(f"Summary (Internal CV) - Average of {num_models} models: "
                         f"AUC={mean_auc:.3f} | ACC={mean_acc:.3f} | "
                         f"SEN={mean_sen:.3f} | SPE={mean_spe:.3f} | F1={mean_f1:.3f}\n")
        
        if external_results:
            external_df, external_metrics = self.format_metrics_table(external_results, "external_test")
            if external_df is not None:
                self.print_formatted_table(external_df, "External Test Set Results")
                
                mean_auc = np.mean(external_metrics["auc"])
                mean_acc = np.mean(external_metrics["acc"])
                mean_sen = np.mean(external_metrics["sen"])
                mean_spe = np.mean(external_metrics["spe"])
                mean_f1  = np.mean(external_metrics["f1"])
                num_models = len(external_metrics["auc"])
                
                print("-" * 90)
                self.log(f"Summary (External Test) - Average of {num_models} models: "
                         f"AUC={mean_auc:.3f} | ACC={mean_acc:.3f} | "
                         f"SEN={mean_sen:.3f} | SPE={mean_spe:.3f} | F1={mean_f1:.3f}\n")
    
    def run(self):
        log_file = self.setup_logging()
        self.load_train_data()
        self.preprocess_train()
        X_ext, y_ext = self.load_external()

        internal_res, external_res = self.cross_validate(X_ext, y_ext)
        self.summarize(internal_res, external_res)

        print(f"\nTraining completed. Log file: {log_file}")


if __name__ == "__main__":
    feature_type = "micro" #ADC micro all
    train_path = f"../radiomics_feature_normalize/train/{feature_type}+mean.xlsx"
    external_path = f"../radiomics_feature_normalize/external/{feature_type}+mean.xlsx"
    clf = RadiomicsClassifier(
        train_path=train_path,
        feature_type=feature_type,
        external_path=external_path
    )
    clf.run()
