import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve, confusion_matrix,
                           classification_report)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RadiomicsClassifier:
    def __init__(self, data_path='example.xlsx'):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.X_reduced = None
        self.X_scaled = None
        self.scaler = None
        self.lasso_selector = None
        self.results = {}
        self.global_best_params = None  
        
        os.makedirs('results', exist_ok=True)
        os.makedirs('fig', exist_ok=True)
        
    def load_data(self):
        self.data = pd.read_excel(self.data_path)
        print(f"data shape: {self.data.shape}")
        
        print(f"label distribution:")
        print(self.data['label'].value_counts())
        
       
        feature_cols = [col for col in self.data.columns if col not in ['Patient', 'label','age']]
        self.X = self.data[feature_cols]
        self.y = self.data['label']
        self.feature_names = feature_cols
        
        print(f"feature number: {self.X.shape[1]}")
        print(f"sample number: {self.X.shape[0]}")
        
        return self.X, self.y
    
    def normalize_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_reduced = self.scaler.fit_transform(self.X)
        
        print(f"normalized feature range:")
        print(f"min: {self.X_reduced.min().min():.6f}")
        print(f"max: {self.X_reduced.max().max():.6f}")
        
        return self.X_reduced
    
    def lasso_feature_selection(self, n_components):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_reduced)
       
        lasso = LassoCV(cv=5, random_state=42, max_iter=2000)

        lasso.fit(X_scaled, self.y)
        
        select_feature_indices = np.where(lasso.coef_ != 0)[0]
        selected_features = [self.feature_names[i] for i in select_feature_indices]
        print(f"LASSO selected feature number: {len(selected_features)}")
        print(f"selected feature names: {selected_features}")
        
        lasso_selector = SelectFromModel(lasso, prefit=True)
        X_lasso = lasso_selector.transform(X_scaled)
        
        if X_lasso.shape[1] >= 2:
            pca = PCA(n_components=n_components, random_state=42)
            X_lasso = pca.fit_transform(X_lasso)
            
            print("\nPCA reduction:")
            print(f"reduction dimension: {X_lasso.shape[1]}")
            print(f"explained variance ratio: {pca.explained_variance_ratio_}")
            print(f"cumulative explained variance ratio: {np.cumsum(pca.explained_variance_ratio_)}")
        
        self.X_reduced = X_lasso
        
        return self.X_reduced
    
    def lasso_feature_selection1(self, n_component, feature_type):  
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_reduced)
        
        if feature_type == 'micro':
            selected_cols = ['squareroot_firstorder_10Percentile_1','wavelet-HLL_glszm_GrayLevelNonUniformityNormalized_2',
                             'wavelet-HHL_glszm_GrayLevelNonUniformityNormalized_2','wavelet-LLL_glszm_SmallAreaEmphasis_2',
                             'wavelet-LLL_glszm_LowGrayLevelZoneEmphasis_2','square_gldm_LowGrayLevelEmphasis_2',
                             'square_glrlm_LowGrayLevelRunEmphasis_2','square_glcm_InverseVariance_2']# ,'logarithm_glrlm_ShortRunLowGrayLevelEmphasis_2'
        elif feature_type == 'ADC':
            selected_cols = ['wavelet-LLH_glrlm_GrayLevelVariance_PGSE','logarithm_firstorder_Skewness_PGSE']
        else:  
            selected_cols = ['squareroot_firstorder_10Percentile_1','wavelet-HHL_glszm_GrayLevelNonUniformityNormalized_2','wavelet-LLL_glszm_SmallAreaEmphasis_2',
                             'square_glcm_InverseVariance_2','wavelet-HLH_glcm_Contrast_50hz','logarithm_gldm_DependenceNonUniformityNormalized_50hz',
                             'logarithm_glcm_ClusterProminence_50hz','wavelet-LLH_glcm_ClusterProminence_PGSE','logarithm_firstorder_Skewness_PGSE']

        selected_indices = [self.feature_names.index(col) for col in selected_cols]
        X_selected = X_scaled[:, selected_indices]
        
        pca = PCA(n_components=n_component, random_state=42)
        self.X_reduced = pca.fit_transform(X_selected)
        
        print("\nPCA reduction:")
        print(f"reduction dimension: {self.X_reduced.shape[1]}")
        print(f"explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"cumulative explained variance ratio: {np.cumsum(pca.explained_variance_ratio_)}")
        
        # principal component
        n_components = pca.n_components_
        
        selected_feature_names = [self.feature_names[i] for i in selected_indices]
        print(f"PCA feature number: {len(selected_feature_names)}")
        print(f"PCA input feature names: {selected_feature_names}")
        
        for i in range(n_components):
            component_weights = pca.components_[i]
            variance_ratio = pca.explained_variance_ratio_[i]
            
            expression_parts = []
            for j, weight in enumerate(component_weights):
                if abs(weight) > 0.001: 
                    feature_name = selected_feature_names[j] if j < len(selected_feature_names) else f'X{j+1}'
                    expression_parts.append(f"{weight:.4f}*{feature_name}")
            
            if expression_parts:
                expression = " + ".join(expression_parts)
                print(f"PC{i+1} (explained variance ratio: {variance_ratio:.3f}):")
                print(f"  {expression}")
        components_df = pd.DataFrame(pca.components_.T, 
                                    index=selected_feature_names,
                                    columns=[f'PC{i+1}' for i in range(n_components)])
        
        os.makedirs('results/pca_components', exist_ok=True)
        components_df.to_csv(f'results/pca_components/radiomics_{feature_type}_pca_components_{feature_type}.csv', encoding='utf-8-sig')
        print(f"\nsave to: results/pca_components/radiomics_{feature_type}_pca_components_{feature_type}.csv")
        
        return self.X_reduced
    def get_classifiers_and_params(self):
        classifiers = {
            'GaussianNB': GaussianNB(),
            'SVM': SVC(probability=True, random_state=42),
            'LR': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'KNN': KNeighborsClassifier()
        }
        
        return classifiers
    
    def calculate_specificity(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    
    def calculate_youden_index(self, y_true, y_prob):
        """
        Args:
            y_true
            y_prob
            
        Returns:
            best_threshold
            best_youden
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_index = tpr - fpr   # Youden = TPR + (1-FPR) - 1 = TPR - FPR
        
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]
        
        return best_threshold
    
    def predict_with_threshold(self, model, X, threshold):
        """
        Args:
            model
            X
            threshold
            
        Returns:
            y_pred
        """
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(X)
            from sklearn.preprocessing import MinMaxScaler
            scaler_prob = MinMaxScaler()
            y_prob = scaler_prob.fit_transform(decision_scores.reshape(-1, 1)).ravel()
        else:
            y_prob = model.predict(X).astype(float)
        
        y_pred = (y_prob >= threshold).astype(int)
        return y_pred
    
    def get_model_probabilities(self, model, X):
        """
        Args:
            model
            X
            
        Returns:
            y_prob
        """
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(X)
            from sklearn.preprocessing import MinMaxScaler
            scaler_prob = MinMaxScaler()
            y_prob = scaler_prob.fit_transform(decision_scores.reshape(-1, 1)).ravel()
        else:
            y_prob = model.predict(X).astype(float)
        
        return y_prob
    
    def cross_validation(self, feature_type):
        classifiers = self.get_classifiers_and_params()
        
        X_cv = self.X_reduced
        y_cv = self.y
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name in classifiers.keys():
            cv_results[name] = {
                'fold_results': [], 
                'best_thresholds_per_fold': [] 
            }
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_cv, y_cv)):
            print(f"\n{'='*20}{fold_idx + 1} flod {'='*20}")
            
            X_fold_train, X_fold_test = X_cv[train_idx], X_cv[test_idx]
            y_fold_train, y_fold_test = y_cv[train_idx], y_cv[test_idx]
            
            print(f"train set size: {len(X_fold_train)}, test set size: {len(X_fold_test)}")
            print(f"train set label distribution: {pd.Series(y_fold_train).value_counts().to_dict()}")
            print(f"test set label distribution: {pd.Series(y_fold_test).value_counts().to_dict()}")
            
            for name, base_clf in classifiers.items():
                print(f"\n处理模型: {name}")
                
                if feature_type == 'ADC':
                    if name == 'SVM':
                        fold_clf = SVC(probability=True, random_state=42,C=10,gamma=0.01,kernel='rbf')
                    elif name == 'GaussianNB':
                        fold_clf = GaussianNB(var_smoothing=0.1)
                    elif name == 'LR':
                        fold_clf = LogisticRegression(C=100,solver='liblinear',random_state=42, max_iter=1000)
                    elif name == 'RandomForest':
                        fold_clf = RandomForestClassifier(random_state=42,max_depth=1,
                                                        min_samples_split=3,n_estimators=100)
                    elif name == 'GradientBoosting':
                        fold_clf = GradientBoostingClassifier(learning_rate=0.01,max_depth=1,n_estimators=150,random_state=42)
                    elif name == 'KNN':
                        fold_clf = KNeighborsClassifier(metric='euclidean',n_neighbors=13,weights='uniform')
                
                elif feature_type == 'micro':
                    if name == 'SVM':
                        fold_clf = SVC(probability=True, random_state=42,C=0.1,gamma='scale',kernel='linear')
                    elif name == 'GaussianNB':
                        fold_clf = GaussianNB(var_smoothing=0.1)
                    elif name == 'LR':
                        fold_clf = LogisticRegression(C=0.1,solver='liblinear',random_state=42, max_iter=1000)
                    elif name == 'RandomForest':
                        fold_clf = RandomForestClassifier(random_state=42,max_depth=2,
                                                        min_samples_split=7,n_estimators=100)
                    elif name == 'GradientBoosting':
                        fold_clf = GradientBoostingClassifier(learning_rate=0.08,max_depth=1,n_estimators=20,random_state=42)
                    elif name == 'KNN':
                        fold_clf = KNeighborsClassifier(metric='euclidean',n_neighbors=8,weights='uniform')

                
                elif feature_type == 'all':
                    if name == 'SVM':
                        fold_clf = SVC(probability=True, random_state=42,C=10,gamma=0.001,kernel='rbf')
                    elif name == 'GaussianNB':
                        fold_clf = GaussianNB(var_smoothing=1)
                    elif name == 'LR':
                        fold_clf = LogisticRegression(C=0.01,solver='liblinear',random_state=42, max_iter=1000)
                    elif name == 'RandomForest':
                        fold_clf = RandomForestClassifier(random_state=42,max_depth=5,
                                                        min_samples_split=10,n_estimators=250)
                    elif name == 'GradientBoosting':
                        fold_clf = GradientBoostingClassifier(learning_rate=0.1,max_depth=3,n_estimators=250,random_state=42)
                    elif name == 'KNN':
                        fold_clf = KNeighborsClassifier(metric='manhattan',n_neighbors=8,weights='distance')

                fold_clf.fit(X_fold_train, y_fold_train)
                
                y_pred_default = fold_clf.predict(X_fold_test)
                y_pred_proba = self.get_model_probabilities(fold_clf, X_fold_test)
                
                try:
                    best_threshold = self.calculate_youden_index(y_fold_test, y_pred_proba)
                except:
                    best_threshold = 0.5, 0.0
                
                y_pred_youden = self.predict_with_threshold(fold_clf, X_fold_test, best_threshold)
                
                fold_result = {
                    'fold': fold_idx + 1,
                    'accuracy_youden': accuracy_score(y_fold_test, y_pred_youden),
                    'recall_youden': recall_score(y_fold_test, y_pred_youden, zero_division=0),
                    'specificity_youden': self.calculate_specificity(y_fold_test, y_pred_youden),
                    'auc': roc_auc_score(y_fold_test, y_pred_proba) if len(np.unique(y_fold_test)) > 1 else 0.0,
                    'best_threshold': best_threshold
                }
                
                cv_results[name]['fold_results'].append(fold_result)
                cv_results[name]['best_thresholds_per_fold'].append(best_threshold)
                
                print(f"  {name}{fold_idx+1}fold test result:")
                print(f"    AUC: {fold_result['auc']:.3f}")
                print(f"    Youden threshold: {best_threshold:.3f}")
                print(f"    Youden accuracy: {fold_result['accuracy_youden']:.3f}")
        
        final_results = {}
        for name in classifiers.keys():
            fold_results = cv_results[name]['fold_results']
            
            metrics = ['accuracy_youden', 'recall_youden', 'specificity_youden', 'auc']
            
            final_results[name] = {}
            for metric in metrics:
                values = [result[metric] for result in fold_results]
                final_results[name][f'{metric}_mean'] = np.mean(values)
                final_results[name][f'{metric}_std'] = np.std(values)
                final_results[name][f'{metric}_values'] = values
            
            thresholds = cv_results[name]['best_thresholds_per_fold']
            final_results[name]['best_threshold_mean'] = np.mean(thresholds)
            final_results[name]['best_threshold_std'] = np.std(thresholds)
            final_results[name]['best_threshold_values'] = thresholds
            lower= final_results[name]['auc_mean']-1.96 *final_results[name]['auc_std']/np.sqrt(5)
            upper= final_results[name]['auc_mean']+1.96 *final_results[name]['auc_std']/np.sqrt(5)
            
            print(f"\n{name} final result of 5-fold cross-validation:")
            print(f"  AUC: {final_results[name]['auc_mean']:.3f} ± {final_results[name]['auc_std']:.3f}")
            print(f"  95%CI: {lower:.3f}-{upper:.3f}")
            print(f"  ACC: {final_results[name]['accuracy_youden_mean']:.3f} ± {final_results[name]['accuracy_youden_std']:.3f}")
            print(f"  SEN: {final_results[name]['recall_youden_mean']:.3f} ± {final_results[name]['recall_youden_std']:.3f}")
            print(f"  SPE: {final_results[name]['specificity_youden_mean']:.3f} ± {final_results[name]['specificity_youden_std']:.3f}")
            print(f"  best threshold: {final_results[name]['best_threshold_mean']:.3f} ± {final_results[name]['best_threshold_std']:.3f}")
            
        return final_results, cv_results
    def run_complete_pipeline(self, feature_type):
        self.load_data()
        
        self.normalize_data()

        self.lasso_feature_selection(n_components=0.9)

        # self.lasso_feature_selection1(feature_type=feature_type)
        
        final_results, cv_results = self.cross_validation(feature_type)
       
        return final_results, cv_results

if __name__ == "__main__":
    feature_type='all' #ADC/micro/all
    model_type='IMPULSED'
    classifier = RadiomicsClassifier(f'../radiomics_feature/{model_type}/{feature_type}_normalize_binwidth0.1_{model_type}.xlsx') 
    final_results, cv_results = classifier.run_complete_pipeline(feature_type)
