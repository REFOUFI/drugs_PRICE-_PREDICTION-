import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime
import argparse


def get_args():

    parser = argparse.ArgumentParser(description = "Drugs Price prediction" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    # arguments for training
    parser.add_argument('--v', type = str , default = 'latest',help='model version')
    #parser.add_argument('--load_model', type=str, default=None, help='.pth file path to load model')
    return parser.parse_args()

if __name__ == "__main__":
    ## load training selected features, encoders , and models
    args = get_args()
    version=args.v
    model_object = load(f'./model_{version}.bst')

    feat_num,feat_cat,feat_text,mean_values,vectorizers,pcas,encoders,xgb_model = model_object['feat_num'],model_object['feat_cat'],model_object['feat_text'],model_object['mean_values'],model_object['vectorizers'],model_object['pcas'],model_object['encoders'],model_object['xgb_model']
    
    #load and prpcess data for testing :
    
    ## load test data(all necessary files )
    ingredient=pd.read_csv('new_ingredient.csv')
    drug_label=pd.read_csv('drug_label_feature_eng.csv' )
    test=pd.read_csv('drugs_test.csv')

    ## process description and paharmacetical companies texts
    test['pharmaceutical_companies']= test['pharmaceutical_companies'].str.lower().apply(lambda x : ' '.join(x.split()))
    test['lower_description']=test['description'].str.lower().apply(lambda x : ' '.join(x.split()))
    drug_label['lower_description']=drug_label['description'].str.lower().apply(lambda x : ' '.join(x.split()))
    ##drop duplicates 
    drug_label = drug_label.drop_duplicates(['label_plaquette', 'label_ampoule', 'label_flacon',
       'label_tube', 'label_stylo', 'label_seringue', 'label_pilulier',
       'label_sachet', 'label_comprime', 'label_gelule', 'label_film',
       'label_poche', 'label_capsule', 'count_plaquette', 'count_ampoule',
       'count_flacon', 'count_tube', 'count_stylo', 'count_seringue',
       'count_pilulier', 'count_sachet', 'count_comprime', 'count_gelule',
       'count_film', 'count_poche', 'count_capsule', 'count_ml',
       'lower_description'])

    ## merge test
    test=test.merge(drug_label,on=['lower_description'],how='left')
    test=test.merge(ingredient,on= ['drug_id'],how='inner')
    Xpca_test = np.empty([test.shape[0], 0])


    ## extract features from text using TF-IDF and PCA
    feat_text_tf=[]
    for feat in feat_text:
    
        tfidf_test =  vectorizers[feat].transform(test[feat]).toarray()
        tfidf_pca_test = pcas[feat].transform(tfidf_test)
        print( 'Dimension of tf-idf after PCA for', feat, ' and test:',tfidf_pca_test.shape)
        Xpca_test = np.hstack((Xpca_test, tfidf_pca_test))
        feat_text_tf = feat_text_tf + ['feat_'+feat[0:5]+'_'+str(x) for x in range(0,tfidf_pca_test.shape[1])]
        

    tfidf_test_df = pd.DataFrame(data=Xpca_test, index=test.index, columns=feat_text_tf)    
    test = test.join(tfidf_test_df)
    print ('Total ---', 'test:', tfidf_test_df.shape)
    ## making reimbursourement rates as a numerical feature
    test['reimbursement_rate'] = test['reimbursement_rate'].str.replace('%','')
    test['reimbursement_rate'] = test['reimbursement_rate'].astype(int)/100
    #creating date features 
    test['marketing_declaration_cycle'] = (pd.to_datetime(test['marketing_declaration_date'])-datetime.now())/ pd.Timedelta(weeks=1)
    test['marketing_authorization_cycle'] = (pd.to_datetime(test['marketing_authorization_date'])-datetime.now())/ pd.Timedelta(weeks=1)
    test['marketing_cycle']=(pd.to_datetime(test['marketing_declaration_date'])-pd.to_datetime(test['marketing_authorization_cycle']))/ pd.Timedelta(weeks=1)

    ## features used in the model
    FEATURES = feat_num +feat_cat+feat_text_tf
    
    ## fill null values - merging concequences-(left instead inner )
    for feat in FEATURES:
        test[feat] = test[feat].fillna(mean_values[feat])

    ## encode categorical features and hidden unseen labels
    for c in feat_cat:
        test[c] = test[c].apply(lambda x : x if x in encoders[c].classes_ else '<unknown>')
        test[c] = encoders[c].transform(test[c]) 
    
    X = test[FEATURES].values

    ## make prediction
    y_predict = xgb_model.predict(X)

    test['price'] = np.exp(y_predict)

    ## create submission file
    test[['drug_id','price']].to_csv(f'submission_{version}.csv',index=False)