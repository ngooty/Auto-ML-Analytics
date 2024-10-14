from mareana_ml_code.mareana_machine_learning.transformers.categorical import CategoricalFeatures
from mareana_ml_code.mareana_machine_learning.transformers.numerical import NumericalFeatures
from mareana_ml_code.mareana_machine_learning.transformers.imputer import Imputers
from mareana_ml_code.mareana_machine_learning.models.classification import Classification
from mareana_ml_code.mareana_machine_learning.models.pls import PLS
from mareana_ml_code.mareana_machine_learning.models.pca import PcaAlgo
from mareana_ml_code.mareana_machine_learning.models.regression import Regression
from mareana_ml_code.mareana_machine_learning.models.clustering import Clustering
from mareana_ml_code.mareana_machine_learning.models.metrics import Metric
from mareana_ml_code.mareana_machine_learning.splits.tt_split import create_test_train_splits
from mareana_ml_code.mareana_machine_learning.feature_union.transformation import FeatureUnion
from mareana_ml_code.mareana_machine_learning.plots.DrawPlots import ClassificationPlots,RegressionPlots,ClusterPlots, PlsPlots, PcaPlots
from mareana_ml_code.mareana_machine_learning.utils.database import PostGres
from mareana_ml_code.mareana_machine_learning.transformers.outliers import Outliers
from mareana_ml_code.mareana_machine_learning.transformers.binning import Binning 
from mareana_ml_code.config_reader import COMMON_URL
import pandas as pd
import json
import numpy as np
import warnings
import requests
from pandas.api.types import is_string_dtype
from sklearn import preprocessing

warnings.filterwarnings("ignore")
import os
from urllib.parse import urlparse




#train_df = pd.read_csv('/Users/aman/Documents/11_sept/mareana_ml/V348.csv', encoding_errors='ignore')
#docker connection
#os.system('docker run -d -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgrespass --name db-my -p 5432:5432 --restart=always postgres')

# train_df.loc[50:99,'Species']='Iris-virginica'

def best_model_evaluation(metrics_json):
    list1 = ['m_max_error','m_median_absolute_error','m_mean_poisson_deviance','m_mean_gamma_deviance','m_mean_tweedie_deviance',
        'm_mean_absolute_error','m_mean_squared_error','m_mean_squared_log_error','m_mean_absolute_percentage_error',
        'm_mean_pinball_loss','m_brier_score_loss','m_hamming_loss','m_hinge_loss','m_log_loss','m_zero_one_loss']
    list_na = ['m_classification_report','m_confusion_matrix','m_det_curve','m_auc','m_matthews_corrcoef','m_multilabel_confusion_matrix',
          'm_precision_recall_curve','m_precision_recall_fscore_support','m_roc_curve']
    
    df = pd.DataFrame.from_dict(metrics_json)
    algorithms=list(df.columns)
    
    df2 = df[~df.index.isin(list_na)]
    
    if (len(df2)>0):
        for i in range(len(df2)):
            for j in range(len(algorithms)):
                df2.iloc[i][j] = df2.iloc[i][j]['test_score']
    
    df2['value']= df2.values.tolist()
    df2['metrics'] = df2.index
    df2['Evaluation'] = ''
    
    if (len(df2)>0):
        for i in range(len(df2)):
            if df2['metrics'][i] in list1:
                min_index = df2['value'][i].index(min(df2['value'][i]))
                df2['Evaluation'][i] = algorithms[min_index]
            else:
                max_index = df2['value'][i].index(max(df2['value'][i]))
                df2['Evaluation'][i] = algorithms[max_index]
                
    df3 = df2['Evaluation']
    df3_dict = df3.to_dict()
    metrics_json['Evaluation'] = df3_dict
    return metrics_json

def print_classification_metrics(jdata,y_train,y_test,_train_prediction,_predictions):
    result_dict={}
    _multiclass=[1 if len(np.unique(y_test.values))>2 else 0]
    METRICS_MAP = {
                'm_accuracys': "Accuracy",
                'm_recall_scores': "Recall",
                'm_precision_scores': "Precision",
                'm_f1_scores': "F1"
            }
    #print("Training metrics for %s Model" %_model['model_name'])
    for m_index, _metrics in jdata['metrics'].items():
        for _metric in _metrics:
            if _metric in METRICS_MAP:
                metric=METRICS_MAP[_metric]
            else:
                metric = _metric

            if metric not in result_dict:
                result_dict[metric] = {}

            if metric=="Accuracy" or _multiclass[0]==0:
                print(
                "%s score for train model is  :: "%metric,
                Metric.METRICS[_metric](
                    y_train,
                    _train_prediction
                ))
                # result_dict[metric]['train_score']=round(float(Metric.METRICS[_metric](y_train,_train_prediction).item()), 2)
                result_dict[metric]['train_score']=round(Metric.METRICS[_metric](
                    y_train,
                    _train_prediction
                ), 2)
            else:
                print(
                "%s score for train model is  :: "%metric,
                Metric.METRICS[_metric](
                    y_train,
                    _train_prediction
                ))
                # result_dict[metric]['train_score']=round(float(Metric.METRICS[_metric](y_train,_train_prediction).item()), 2)
                result_dict[metric]['train_score']=round(Metric.METRICS[_metric](
                    y_train,
                    _train_prediction
                ), 2)
    #print("\nTest metrics for %s Model" %_model['model_name'])
    for m_index, _metrics in jdata['metrics'].items():
        for _metric in _metrics:

            if _metric in METRICS_MAP:
                metric=METRICS_MAP[_metric]
            else:
                metric = _metric

            if metric not in result_dict:
                result_dict[metric] = {}

            if metric=="Accuracy" or _multiclass[0]==0:
                print(
                "%s score for test model is  :: "%metric,
                Metric.METRICS[_metric](
                    y_test,
                    _predictions
                ))
                # result_dict[metric]['test_score']=round(float(Metric.METRICS[_metric](y_test,_predictions).item()), 2)
                result_dict[metric]['test_score']=round(Metric.METRICS[_metric](
                    y_test,
                    _predictions
                ), 2)
            else:
                print(
                "%s score for test model is  :: "%metric,
                Metric.METRICS[_metric](
                    y_test,
                    _predictions
                ))
                # result_dict[metric]['test_score']=round(float(Metric.METRICS[_metric](y_test,_predictions).item()), 2)
                result_dict[metric]['test_score']=round(Metric.METRICS[_metric](
                    y_test,
                    _predictions
                ), 2)
    return result_dict

def print_regression_metrics(jdata,y_train,y_test,_train_prediction,_predictions):
    #print("Training metrics for %s Model" %_model['model_name'])
    result_dict={}
    for m_index, _metrics in jdata['metrics'].items():
        for _metric in _metrics:
            print(_metric)
            if _metric in ['m_mean_tweedie_deviance', 'm_mean_poisson_deviance'] :
                _train_prediction = abs(_train_prediction)
                _predictions = abs(_predictions)
                y_train = abs(y_train)
                y_test = abs(y_test)
            print(
                "%s value for train model is  :: "%_metric,
                Metric.METRICS[_metric](
                    y_train,
                    _train_prediction
                ))
            if _metric in result_dict:
                result_dict[_metric]['train_score']=round(Metric.METRICS[_metric](
                        y_train,
                        _train_prediction
                    ), 2)
            else:
                result_dict[_metric] = {}
                result_dict[_metric]['train_score']=round(Metric.METRICS[_metric](
                        y_train,
                        _train_prediction
                    ), 2)
    #print("\nTest metrics for %s Model" %_model['model_name'])
    for m_index, _metrics in jdata['metrics'].items():
        for _metric in _metrics:
            if _metric in ['m_mean_tweedie_deviance', 'm_mean_poisson_deviance'] :
                _train_prediction = abs(_train_prediction)
                _predictions = abs(_predictions)
                y_train = abs(y_train)
                y_test = abs(y_test)
            print(
                "%s value for test model is  :: "%_metric,
                Metric.METRICS[_metric](
                    y_test,
                    _predictions
                ))

            if _metric in result_dict:
                result_dict[_metric]['test_score']=round(Metric.METRICS[_metric](
                        y_test,
                        _predictions
                    ), 2)
            else:
                result_dict[_metric] = {}
                result_dict[_metric]['test_score']=round(Metric.METRICS[_metric](
                        y_test,
                        _predictions
                    ), 2)
    return result_dict

def getting_metrics_and_charts_data(model,jdata,y_train,y_test,_train_prediction,_predictions,_model,X_test,X_train,cvresult,_estimator,train_df):
            if "Classification" in str(model):
                _result_dict=print_classification_metrics(jdata,y_train,y_test,_train_prediction,_predictions)
                _multiclass=[1 if len(np.unique(y_test.values))>2 else 0]
                _fig_obj=ClassificationPlots().render(_model,X_test,y_test.squeeze(),_multiclass)
                
            elif "Regression" in str(model):
                rp_obj=RegressionPlots()
                _result_dict=print_regression_metrics(jdata,y_train,y_test,_train_prediction,_predictions)
                _fig_obj=rp_obj.render(_model,X_train,y_train,X_test,y_test,cvresult,jdata)

            elif "PLS" in str(model):
                pls_plot = PlsPlots()
                if 'PLSSVD' in str(_model):
                    _train_prediction = _train_prediction[0]
                    _predictions = _predictions[0]

                _result_dict=print_regression_metrics(jdata,y_train,y_test,_train_prediction,_predictions)
                _fig_obj = pls_plot.render(_model, X_train, y_train, X_test, y_test, cvresult)
            
            elif "pca" in str(model).lower():
                pls_plot = PcaPlots()
                _result_dict=print_regression_metrics(jdata,y_train,y_test,_train_prediction,_predictions)
                _fig_obj = pls_plot.render(_estimator, _model, X_train, y_train, X_test, y_test, cvresult)

            else:
                print('Results of clustering...')
                _cluster_plots=ClusterPlots()
                print(_predictions)
                _fig_obj=rp_obj.render(_model,train_df)
                _result_dict=""

                query = ("INSERT INTO work_pipeline_results (cust_key, pipeline_disp_id, charts_data, metric_vals,created_by,created_on) "
                "    VALUES ('1000',%(pipeline_disp_id)s , %(charts_data)s, %(metric_vals)s,'system@mareana.com',now()) "
                "    ON CONFLICT (pipeline_disp_id) "
                "    DO UPDATE SET charts_data=%(charts_data)s, metric_vals=%(metric_vals)s;")

            return _result_dict, _fig_obj

def run_main(app_id):
    MAP = {
        'categorical': CategoricalFeatures,
        'numerical': NumericalFeatures,
        'Imputer': Imputers,
        'classification': Classification,
        'regression': Regression,
        'cross_decomposition': PLS,
        'clustering': Clustering,
        'decomposition': PcaAlgo
    }
    db=PostGres()

    cur=db.connect()
    cur.execute(f"select view_disp_id,view_version,pipeline_data from work_pipeline_master where pipeline_disp_id='{app_id}'")
    _result=cur.fetchall()
  
    view_id=_result[0][0]
    view_version=_result[0][1]
    view_json=_result[0][2]
    
    baseurl = COMMON_URL + 'services/v1/analysis-preprocessing'
    headers = {'x-access-token':'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6InN5c3RlbV9zdXBlckBtYXJlYW5hLmNvbSIsInVuaXhfdGltZXN0YW1wIjoxNjcxNTI3MDUyLjc0NTg3NCwidGltZXN0YW1wIjoiMjAvMTIvMjAyMiAxNDozNDoxMiIsImV4cCI6NDgyNTE0Njg1MiwiYWRfcm9sZSI6ZmFsc2UsIm1kaF9yb2xlIjoiU1VQRVJfQURNSU4iLCJlbWFpbF9pZCI6InN5c3RlbV9zdXBlckBtYXJlYW5hLmNvbSIsImN1c3Rfa2V5IjoiMTAwMCJ9.fXLHF2djt-2ZXOwHWIUVTni-GeD9ki01Gzl2l_z4ZwI',
               'resource-name': 'ANALYTICS'}
    payload = {
      "batch_filter": view_json[0]['batch_filter'],
      "data_filter": view_json[0]['data_filter'],
      "view_disp_id": view_json[0]['view_disp_id'],
      "view_version": view_json[0]['view_version'],
      "pipeline_disp_id":app_id
    }

    # This is post request
    response = requests.post(baseurl, json=payload, headers=headers)
    view_data = json.loads(response.text)
    train_df= pd.DataFrame(view_data['data']['compressed_output'])
    train_df.pop('batch_set_date')
    train_df.pop('sequence_id')

    # Convert all values in self.cat_col to strings (For handling mix of int and strings)
    cat_col=[x for x in train_df.columns if train_df[x].dtype=='object']
    train_df[cat_col] = train_df[cat_col].astype(str)
    
    jdata=view_json[0]
    hash_variable = {}
    data_variable = []
    target_variable = None
    for variable in jdata['variable_mapping']:
        hash_variable[variable['variable_id']] = variable
        if variable['variable_id'] in jdata['X'] or variable['variable_id'] in jdata['Y']: 
            data_variable.append(variable['variable_name'])
        if variable['variable_id'] in jdata['Y']:
            target_variable = variable['variable_name']
    try:
        train_df = train_df[data_variable]
    except Exception as e:
        print(e)
        print("Error in choosing parameters, check the json !!")

    #Code to change to hash variable category will be removed after handling in UI
    if(len(jdata['categorical_mapping'].items())>0):
        if len(jdata['categorical_mapping']['variable_list']) > 0:
            for i in range(len(hash_variable)):
                if hash_variable[i]['variable_name'] in jdata['categorical_mapping']['variable_list']:
                    hash_variable[i]['variable_category'] = "categorical"
    
    # try:
    #     #Outlier treatment code 
    #     if(len(jdata['outlier_mapping'].items())>0):
    #         index, outlier_mapping = jdata['outlier_mapping'].items()
    #         for i, variable in enumerate(outlier_mapping[1]):
    #             _option = outlier_mapping[1][i]
    #             train_df = Outliers(train_df,index[1][i],_option).TreatOutliers()
    #     #End of outlier treatment
    # except Exception as e:
    #     print(e)
    #     print("Error in Outlier treatment, check the json !!")

    # try:
    #     #Binning/make_categorical code
    #     if(len(jdata['categorical_mapping'].items())>0):
    #         index = jdata['categorical_mapping']['variable_list']
    #         for i in index:
    #             train_df = Binning(train_df,i).make_categorical()
    #     #End of Binning/make_categorical
    # except Exception as e:
    #     print(e)
    #     print("Error in Categorical Mapping, check the json !!")

    try:
        test_split_ratio = jdata['test_split']/100
    except:
        test_split_ratio = 0.3
    train_df.dropna(subset=[target_variable], inplace=True)
    X_train, X_test, y_train, y_test = create_test_train_splits(
        train_df, target_variable, test_split_ratio
    )

    if is_string_dtype(y_train[target_variable]):
        label_encoder = preprocessing.LabelEncoder()
        y_train[target_variable]= label_encoder.fit_transform(y_train[target_variable])
        y_test[target_variable]= label_encoder.fit_transform(y_test[target_variable])

    global_trans_steps = {}
    transformation_mapping = {}
    # for index, transformer_mapping in jdata['transformation_mapping'].items():
    #     for variable in transformer_mapping['variable_list']:
    #         if hash_variable[variable]['variable_category'] == 'categorical' and transformer_mapping['type'] == "Imputer":
    #             datatype_transformer = MAP[
    #                 transformer_mapping['type']
    #             ]
    #         else:
    #             datatype_transformer = MAP[
    #                 hash_variable[variable]['variable_category']
    #             ]
    #         if variable in jdata['X']:
    #             _trans = datatype_transformer(
    #                 X_train,
    #                 hash_variable[variable]['variable_name'],
    #                 transformer_mapping['transformation'],
    #                 transformer_mapping['parameters']
    #             )
    #             X_train = _trans.fit_transform()
    #             X_test = _trans.transform(X_test)
    #         else:
    #             _trans = datatype_transformer(
    #                 y_train,
    #                 hash_variable[variable]['variable_name'],
    #                 transformer_mapping['transformation'],
    #                 transformer_mapping['parameters']
    #             )
    #             y_train = _trans.fit_transform()
    #             y_test = _trans.transform(y_test)
    #         transformation_mapping[index] = _trans.returnasset()
    # global_trans_steps['transformation_mapping'] = transformation_mapping

    feature_union_steps = {}
    for index, feature_union_mapping in jdata['feature_union_mapping'].items():
        variable_names = []
        for variable in feature_union_mapping['variable_list']:
            variable_names.append(hash_variable[variable]['variable_name'])

        variable_type = []
        for variable in feature_union_mapping['variable_list']:
            variable_type.append(hash_variable[variable]['variable_category'])

        if len(variable_names) > 0:
            _trans = FeatureUnion(
                        X_train,
                        variable_names,
                        variable_type,
                        feature_union_mapping['transformation'],
                        feature_union_mapping['type'],
                        feature_union_mapping['parameters']
                    )
            # print("X_train before transformation")
            # print(X_train)
            # print(variable_names)
            X_train = _trans.fit_transform()
            X_test = _trans.transform(X_test)
            feature_union_steps[index] = _trans.returnasset()
    global_trans_steps['feature_union_mapping'] = feature_union_steps

    X_train=X_train.select_dtypes(exclude='object')
    X_test=X_test.select_dtypes(exclude='object')

    models = {}
    chart_data = {}
    metrics_data = {}

    if jdata['meta_estimator']:
        X_train_meta = pd.DataFrame()
        X_test_meta = pd.DataFrame()

        for index, _model in jdata['estimator'].items():
            if _model['estimator_type']=='decomposition':
                model_name = _model['model_name']
                model = MAP[_model['estimator_type']]
                variable_type_df = pd.DataFrame(jdata['variable_mapping'])
                target_variable_category = variable_type_df[variable_type_df.variable_name == target_variable].variable_category.values[0]
                _estimator = model(
                    X_train,
                    y_train,
                    target_variable_category,
                    _model['model_name'],
                    _model['hyperparamters']
                )
                _model=_estimator.tune()
                _train_prediction = _estimator.predict(X_train)
                _predictions = _estimator.predict(X_test)
                cvresult=_estimator.cv_results

                X_train_meta[model_name] = pd.DataFrame(_train_prediction)
                X_test_meta[model_name] =  pd.DataFrame(_predictions)

            else:
                model_name = _model['model_name']
                model = MAP[_model['estimator_type']]
                _estimator = model(
                    X_train,
                    y_train,
                    _model['model_name'],
                    _model['hyperparamters']
                )
                _model=_estimator.tune()
                _train_prediction = _estimator.predict(X_train)
                _predictions = _estimator.predict(X_test)
                cvresult=_estimator.cv_results

                

                X_train_meta[model_name] = pd.DataFrame(_train_prediction)
                X_test_meta[model_name] =  pd.DataFrame(_predictions)

        for index, _model in jdata['meta_estimator'].items():
            if _model['estimator_type']=='decomposition':
                model = MAP[_model['estimator_type']]
                variable_type_df = pd.DataFrame(jdata['variable_mapping'])
                target_variable_category = variable_type_df[variable_type_df.variable_name == target_variable].variable_category.values[0]
                _estimator = model(
                    X_train_meta,
                    y_train,
                    target_variable_category,
                    _model['model_name'],
                    _model['hyperparamters']
                )
                _model=_estimator.tune()
                _train_prediction = _estimator.predict(X_train_meta)
                _predictions = _estimator.predict(X_test_meta)
                cvresult=_estimator.cv_results
            
            else:
                model = MAP[_model['estimator_type']]
                _estimator = model(
                    X_train_meta,
                    y_train,
                    _model['model_name'],
                    _model['hyperparamters']
                )
                _model=_estimator.tune()
                _train_prediction = _estimator.predict(X_train_meta)
                _predictions = _estimator.predict(X_test_meta)
                cvresult=_estimator.cv_results
            
            metrics_data[jdata['meta_estimator'][index]['model_name']], chart_data[jdata['meta_estimator'][index]['model_name']] = getting_metrics_and_charts_data(model,jdata,y_train,y_test,_train_prediction,_predictions,_model,X_test_meta,X_train_meta,cvresult,_estimator,train_df)
            models[index] = _estimator.returnasset()
            
    else:
        for index, _model in jdata['estimator'].items():
            if _model['estimator_type']=='clustering':
                model = MAP[_model['estimator_type']]
                _estimator = model(
                    train_df,
                    _model['model_name'],
                    _model['hyperparamters']
                )
                _model=_estimator.tune()
                _clusters = _estimator.predict(train_df)
                train_df['Clusters']=_clusters

            elif _model['estimator_type']=='decomposition':
                model = MAP[_model['estimator_type']]
                variable_type_df = pd.DataFrame(jdata['variable_mapping'])
                target_variable_category = variable_type_df[variable_type_df.variable_name == target_variable].variable_category.values[0]
                _estimator = model(
                    X_train,
                    y_train,
                    target_variable_category,
                    _model['model_name'],
                    _model['hyperparamters']
                )
                _model=_estimator.tune()
                _train_prediction = _estimator.predict(X_train)
                _predictions = _estimator.predict(X_test)
                cvresult=_estimator.cv_results
            else:
                model = MAP[_model['estimator_type']]
                _estimator = model(
                    X_train,
                    y_train,
                    _model['model_name'],
                    _model['hyperparamters']
                )
                _model=_estimator.tune()
                _train_prediction = _estimator.predict(X_train)
                _predictions = _estimator.predict(X_test)
                cvresult=_estimator.cv_results

            metrics_data[jdata['estimator'][index]['model_name']], chart_data[jdata['estimator'][index]['model_name']] = getting_metrics_and_charts_data(model,jdata,y_train,y_test,_train_prediction,_predictions,_model,X_test,X_train,cvresult,_estimator,train_df)
            models[index] = _estimator.returnasset()

        # Need to change the full_url to view name and unique identifier
        full_url = str(app_id) + '-' +  str(view_id) + '-' + str(view_version)
        model_name=full_url

        

        #ClassificationPlots().render(_model,X_test,y_test.squeeze())

        # with open('estimator.pickle', 'wb') as f:
        #    pickle.dump(_estimator.returnasset(), f)
        # Check the target variable is a multiclass or not. 
        # If yes, then recall and precison score functions should have "average='micro' parameter"


    metrics_data = best_model_evaluation(metrics_data)
    params={}
    query = ("INSERT INTO work_pipeline_results (cust_key, pipeline_disp_id, charts_data, metric_vals,created_by,created_on) "
    "    VALUES ('1000',%(pipeline_disp_id)s , %(charts_data)s, %(metric_vals)s,'system@mareana.com',now()) "
    "    ON CONFLICT (pipeline_disp_id) "
    "    DO UPDATE SET charts_data=%(charts_data)s, metric_vals=%(metric_vals)s;")
    params['pipeline_disp_id'] = app_id
    params['charts_data']=json.dumps(chart_data)
    params['metric_vals']=json.dumps(metrics_data)
    cur.execute(query,params)
    db.commit()

    query = ("UPDATE work_pipeline_results SET  run_status=%(run_status)s, res_message=%(res_message)s WHERE pipeline_disp_id = %(pipeline_disp_id)s")
    params = {}
    params['pipeline_disp_id'] = app_id
    params['run_status'] = 'Successful'
    params['res_message'] = 'Successful'
    cur.execute(query,params)
    db.commit()

def update_res_status(app_id, run_status, res_message):
    db=PostGres()
    cur=db.connect()
    query = ("INSERT INTO work_pipeline_results (cust_key, pipeline_disp_id, run_status, res_message,created_by, created_on) "
    "    VALUES ('1000',%(pipeline_disp_id)s , %(run_status)s, %(res_message)s,'system@mareana.com', now()) "
    "    ON CONFLICT (pipeline_disp_id) "
    "    DO UPDATE SET run_status=%(run_status)s, res_message=%(res_message)s;")
    params = {}
    params['pipeline_disp_id'] = app_id
    params['run_status'] = run_status
    params['res_message'] = res_message
    cur.execute(query,params)
    db.commit()

def run(app_id):
    try:
        update_res_status(app_id, 'Pending', 'Model is running in airflow.')
        run_main(app_id)
    except Exception as e:
        print(e)
        update_res_status(app_id, 'Failed', str(e))
        raise 
    

# if __name__ == "__main__":
#     run('P555')
