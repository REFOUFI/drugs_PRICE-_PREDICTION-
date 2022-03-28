#import necessary python packages 

import pandas as pd
import numpy as np
import argparse
import unicodedata
import re
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time, math
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')



# # loading data and merge

# In[124]:
def load_merge_data(train_path,ing_path,label_path):
    train=pd.read_csv(train_path)#('drugs_train.csv') #encoding='utf-8' sep=';'
    ingredient=pd.read_csv(ing_path)#('new_ingredient.csv')
    drug_label=pd.read_csv(label_path)#('drug_label_feature_eng.csv' )


    train['lower_description']=train['description'].str.lower().apply(lambda x : ' '.join(x.split()))
    train['pharmaceutical_companies']= train['pharmaceutical_companies'].str.lower().apply(lambda x : ' '.join(x.split()))
    drug_label['lower_description']=drug_label['description'].str.lower().apply(lambda x : ' '.join(x.split()))

    drug_label = drug_label.drop_duplicates(['label_plaquette', 'label_ampoule', 'label_flacon',
       'label_tube', 'label_stylo', 'label_seringue', 'label_pilulier',
       'label_sachet', 'label_comprime', 'label_gelule', 'label_film',
       'label_poche', 'label_capsule', 'count_plaquette', 'count_ampoule',
       'count_flacon', 'count_tube', 'count_stylo', 'count_seringue',
       'count_pilulier', 'count_sachet', 'count_comprime', 'count_gelule',
       'count_film', 'count_poche', 'count_capsule', 'count_ml',
       'lower_description'])

    train=train.merge(drug_label,on=['lower_description'],how='inner')

    train=train.merge(ingredient,on= ['drug_id'],how='inner' )


    # prepocess reimbursement_rate

    train['reimbursement_rate'] = train['reimbursement_rate'].str.replace('%','')
    train['reimbursement_rate'] = train['reimbursement_rate'].astype(int)/100

    # preocess dates and generate new fatures called cycles 
    train['marketing_declaration_cycle'] = (pd.to_datetime(train['marketing_declaration_date'])-datetime.now())/ pd.Timedelta(weeks=1)
    train['marketing_authorization_cycle'] = (pd.to_datetime(train['marketing_authorization_date'])-datetime.now())/ pd.Timedelta(weeks=1)
    train['marketing_cycle']=(pd.to_datetime(train['marketing_declaration_date'])-pd.to_datetime(train['marketing_authorization_cycle']))/ pd.Timedelta(weeks=1)

    return train





# generating text features throughout TF-IDF 

# Text Features  retained 'route_of_administration','pharmaceutical_companies',


def apply_tfidf_pca(feature):                           
    vectorizer = TfidfVectorizer(encoding='utf-8', strip_accents='ascii', lowercase=True)
    tfidf_train = vectorizer.fit_transform(train[feature]).toarray()   # on training set
    
    pca = PCA(n_components=0.9, whiten=True)# 0.90 of variance # PCA: 90% of the variance is retained
    tfidf_train = pca.fit_transform(tfidf_train)  # on the training and test set
    return tfidf_train , vectorizer , pca


def extract_text_features(train,feat_text):

    feat_text_tf = []
    vectorizers = dict()
    pcas = dict()
    Xpca_train = np.empty([train.shape[0], 0])
    for feat in feat_text:
        # computes a tfidf matrix, apply PCA, for each text feature:
        tfidf_pca_train,vectorizer,pca = apply_tfidf_pca(feat)
        print( 'Dimension of tf-idf after PCA for', feat, ' - train:',tfidf_pca_train.shape)
        # builds a matrix containing all the reduced components of each tfidf matrix:
        Xpca_train = np.hstack((Xpca_train, tfidf_pca_train))
        #Xpca_test = np.hstack((Xpca_test, tfidf_pca_test))
        # creates a name for each component:  
        vectorizers[feat] = vectorizer
        pcas[feat] = pca
        feat_text_tf = feat_text_tf + ['feat_'+feat[0:5]+'_'+str(x) for x in range(0,tfidf_pca_train.shape[1])]

    # creates dataframe out of Xpca_train   
    tfidf_train_df = pd.DataFrame(data=Xpca_train, index=train.index, columns=feat_text_tf)   

    train = train.join(tfidf_train_df)

    print ('Total ---', 'train:', tfidf_train_df.shape)

    return train,feat_text_tf,vectorizers,pcas

#to fill missing value (will be nedded on the test)
def cal_mean_values(train,FEATURES,feat_cat):
    mean_values = dict()

    for feat in FEATURES:
        if feat in (feat_cat):
            mean_values[feat]=train[feat].mode()[0]
        else:
            mean_values[feat]=train[feat].mean()

    return mean_values
        

#encoding categorical features 
def encode_cat_cols(train,feat_cat):

    encoders = dict()
    for c in feat_cat:
        le = LabelEncoder()
        code_list = train[c].tolist()
        code_list.append('<unknown>')
        le.fit(code_list)    
        train[c] = le.transform(train[c])   # transforming on train
        encoders[c]=le
    
    return train,encoders
       

#preprocess the target 
def process_target(train):
    train['logprice'] = train['price'].apply(np.log)
    return train


# # model and training 

# In[143]:

def get_hyperparams(train,FEATURES,seed,init_points=10,n_iter=20):

    X = train[FEATURES].values
    y = train['logprice'].values
    #random_state = 120
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed,shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed,shuffle=True)





    def xgbc_cv(subsample,max_depth,learning_rate,n_estimators,base_score,min_child_weight):
        from sklearn.metrics import mean_absolute_percentage_error
        import numpy as np


        estimator_function = xgb.XGBRegressor(tree_method='hist', nthread=-1, use_label_encoder=False, eval_metric='mape',
                                            gamma=0, colsample_bytree=subsample,max_depth=int(max_depth),
                                           learning_rate= learning_rate,
                                           n_estimators= int(n_estimators),seed = seed,objective='reg:squaredlogerror',base_score=base_score,min_child_weight=min_child_weight)
        # Fit the estimator
        estimator_function.fit(X_train,y_train)

        # calculate out-of-the-box using validation set 1
        probs = estimator_function.predict(X_val)
    
        val1_mape = mean_absolute_percentage_error(y_val,probs)

    
        # return the mean validation score to be maximized
        #return np.array([val1_roc,val2_roc]).mean()
        return 1-val1_mape



    from bayes_opt import BayesianOptimization

    # alpha is a parameter for the gaussian process
    # Note that this is itself a hyperparemter that can be optimized.
    gp_params = {"alpha": 1e-10}

    # We create the BayesianOptimization objects using the functions that utilize
    # the respective classifiers and return cross-validated scores to be optimized.


    # We create the bayes_opt object and pass the function to be maximized
    # together with the parameters names and their bounds.
    # Note the syntax of bayes_opt package: bounds of hyperparameters are passed as two-tuples

    hyperparameter_space = {
        'subsample': (0,1),
        'max_depth': (5,15),
        'learning_rate':(0.01,0.5),
        'n_estimators':(100,500),
        'base_score':(0.1,0.8),
        'min_child_weight': (0.2,0.8)
    }

    xgbcBO = BayesianOptimization(f = xgbc_cv,
                              pbounds =  hyperparameter_space,
                              random_state = seed,
                              verbose = 10)

    # Finally we call .maximize method of the optimizer with the appropriate arguments
    # kappa is a measure of 'aggressiveness' of the bayesian optimization process
    # The algorithm will randomly choose 3 points to establish a 'prior', then will perform
    # 10 interations to maximize the value of estimator function
    xgbcBO.maximize(init_points=init_points,n_iter=n_iter,acq='ucb', kappa= 3, **gp_params)



    targets = []
    for i, rs in enumerate(xgbcBO.res):
        targets.append(rs["target"])
    best_params = xgbcBO.res[targets.index(max(targets))]["params"]
    best_params['n_estimators'] = round(best_params['n_estimators'])
    best_params['max_depth'] = round(best_params['max_depth'])
    return best_params






# In[149]:


#params = {
#    'subsample': 0.85,
#    'max_depth': 6,
#    'learning_rate':0.15,
#    'n_estimators':400,
#    'base_score':0.21,
#    'min_child_weight': 0.6
#}

def kfold_training(X_train,y_train,best_params,n_iter=1,nfold=5):
    #import metrics for evaluation 
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.metrics import mean_absolute_percentage_error
    
    
    
    
    mapes = []
    models = []
    for k in range(n_iter):
        kfold = KFold(nfold, random_state = 42 + k, shuffle = True)
        for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(X_train)):
            print("-----------")
            print("-----------")

            #model = MultiLGBMClassifier(resolution = 5, params = params)

            model = xgb.XGBRegressor(**best_params)

            model.fit(X_train[tr_inds], y_train[tr_inds],
                      eval_set=[(X_train[tr_inds], y_train[tr_inds]),
                                (X_train[val_inds], y_train[val_inds])],
                      eval_metric='mape',early_stopping_rounds=250,verbose=False) #maximize=False,verbose_eval=15

            #model = xgboost.train(params, dtrain, 500, watchlist, early_stopping_rounds=250,
            #          maximize=False, verbose_eval=15)

            #model.fit(train_x[tr_inds], train_y_raw.values[tr_inds])
            y_pred = model.predict(X_train[val_inds])
            y_true = np.array(y_train[val_inds])
            mape = mean_absolute_percentage_error(y_true , y_pred)
            
            models.append(model)
            print(mape)
            mapes.append(mape)

    best_model = models[np.argmin(mapes)]
    return best_model


def mape_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_model(X_test,y_test,model):
    from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,r2_score
    y_predict = model.predict(X_test)
    print('MAPE: ',mean_absolute_percentage_error(y_test,y_predict))
    print('MSE: ',mean_squared_error(y_test,y_predict))
    print('R2: ',r2_score(y_test,y_predict))
    

    


def save_model(version,feat_num,feat_cat,feat_text,mean_values,vectorizers,pcas,encoders,xgb_model):

    from joblib import dump
    ## save model
    model_object={
        'feat_num': feat_num,
        'feat_cat':feat_cat,
        'feat_text':feat_text,
        'mean_values':mean_values,
        'vectorizers':vectorizers,
        'pcas':pcas,
        'encoders':encoders,
        'xgb_model':xgb_model
    }

    dump(model_object,f'./model_{version}.bst')
    
def get_args():

    parser = argparse.ArgumentParser(description = "Drugs Price prediction" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    # arguments for training
    parser.add_argument('--v', type = str , default = 'latest',help='model version')
    parser.add_argument('--initp', type=int , default=20,help='number of init point buyesian optimisation')
    parser.add_argument('--hiter', type=int, default=10,help='number of iteration buyesian optimisation')
    parser.add_argument('--nfold', type=int, default=5,help='number of folds train')
    parser.add_argument('--kiter', type=int, default=1,help='number of iteration folds train')
    parser.add_argument('--seed', type=int, default=120,help='seed')
   

    #parser.add_argument('--load_model', type=str, default=None, help='.pth file path to load model')
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    version = args.v

    init_points = args.initp
    h_n_iter = args.hiter
    nfold = args.nfold
    k_n_iter = args.kiter 

    feat_text = ['active_ingredient','pharmaceutical_companies']


    feat_num = ['marketing_declaration_cycle','marketing_authorization_cycle',
            'marketing_cycle']+['label_plaquette', 'label_ampoule','label_flacon', 
            'label_tube', 'label_stylo', 'label_seringue',
            'label_pilulier', 'label_sachet', 'label_comprime', 
            'label_gelule', 'label_film', 'label_poche',
            'label_capsule'] + ['count_plaquette', 'count_ampoule', 
            'count_flacon', 'count_tube', 'count_stylo', 'count_seringue',
            'count_pilulier', 'count_sachet', 'count_comprime', 'count_gelule', 
            'count_film', 'count_poche', 'count_capsule', 'count_ml']+['reimbursement_rate']

    feat_cat = ['marketing_authorization_status', 'marketing_status', 'approved_for_hospital_use',
             'marketing_authorization_process', 'dosage_form', 'route_of_administration']

    train_path = './drugs_train.csv'
    ing_path = './new_ingredient.csv'
    label_path = './drug_label_feature_eng.csv'

    train = load_merge_data(train_path,ing_path,label_path)
    train, feat_text_tf,vectorizers,pcas = extract_text_features(train,feat_text)

    

    FEATURES = feat_num +feat_cat+feat_text_tf

    mean_values = cal_mean_values(train,FEATURES,feat_cat)

    train,encoders = encode_cat_cols(train,feat_cat)

    train = process_target(train)



    seed = args.seed
    best_params = get_hyperparams(train,FEATURES,seed,init_points,h_n_iter)

    X = train[FEATURES].values
    y = train['logprice'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed,shuffle=True)


    best_model = kfold_training(X_train,y_train,best_params,n_iter=k_n_iter,nfold=nfold)

    evaluate_model(X_test,y_test,best_model)

    save_model(version,feat_num,feat_cat,feat_text,mean_values,vectorizers,pcas,encoders,best_model)

