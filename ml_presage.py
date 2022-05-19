# pip install scikit-learn
# pip install numpy 
# pip install scipy
# pip install matplotlib
# pip install ast

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import requests
import ast

import time
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from scipy.stats import normaltest
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter

list_item1_same_len=0
list_item2_same_len=0
list_item3_same_len=0

def ml_filter(matrixum,numww=1,model_on=[1,2,3,4,5,6]):
    resulutor = []
    numw=numww
    dbe=400
    matrixumm = [[h for h in i[:-1]] for i in matrixum]
    x_train, y_train, x_test, y_test = np.array([i[:] for i in matrixumm[:-numw]]),np.array([i[-1] for i in matrixum[:-numw]]),np.array([i[:] for i in [matrixumm[-numw]]]),np.array([i[-1] for i in [matrixum[-numw]]])
    ################################################################################
    if 1 in model_on:
        print("compute model 1")
        # Instantiate
        tree_model = tree.DecisionTreeClassifier(max_depth=dbe)
        # Fit a decision tree
        tree_model = tree_model.fit(x_train, y_train)
        # Predictions/probs on the test dataset
        predicted = tree_model.predict(x_test)[0]
        resulutor.append([predicted,tree_model.score(x_train, y_train)])
    ################################################################################
    if 2 in model_on:
        print("compute model 2")
        # Instantiate
        rf = RandomForestClassifier()    
        # Fit
        rf_model = rf.fit(x_train, y_train)
        # Predictions/probs on the test dataset
        predicted = rf_model.predict(x_test)[0]
        resulutor.append([predicted,rf_model.score(x_train, y_train)])
        ###############################################################################
    if 3 in model_on:
        print("compute model 3")
        # Instantiate
        svm_model = SVC()
        # Fit
        svm_model = svm_model.fit(x_train, y_train)
        # Predictions/probs on the test dataset
        predicted = svm_model.predict(x_test)[0]
        resulutor.append([predicted,svm_model.score(x_train, y_train)])
        ###############################################################################
    if 4 in model_on:
        print("compute model 4")
        # instantiate
        knn_model = KNeighborsClassifier(n_neighbors=dbe)
        # fit the model
        knn_model.fit(x_train, y_train)
        # Predictions/probs on the test dataset
        predicted = knn_model.predict(x_test)[0]
        resulutor.append([predicted,knn_model.score(x_train, y_train)])
        ###############################################################################
    if 5 in model_on:
        print("compute model 5")
        # Instantiate
        bayes_model = GaussianNB()
        # Fit the model
        bayes_model.fit(x_train, y_train)
        # Predictions/probs on the test dataset
        predicted = bayes_model.predict(x_test)[0]
        resulutor.append([predicted,bayes_model.score(x_train, y_train)])
        ################################################################################
    if 6 in model_on:
        print("compute model 6")
        # Instantiate
        MLP_model = MLPClassifier(learning_rate='adaptive',hidden_layer_sizes=(10000), activation='relu',shuffle=True, max_iter=30,n_iter_no_change=1000,verbose=True)
        # Fit the model
        MLP_model.fit(x_train, y_train)
        # Predictions/probs on the test dataset
        predicted = MLP_model.predict(x_test)[0]
        resulutor.append([predicted,MLP_model.score(x_train, y_train)])
        ###################################################################################
    return resulutor,y_test

def rebuilder(aa,imo):
    a = aa
    imoo = np.rot90(imo,1)
    origine = a[:,-1].reshape(imoo.shape[0],1)
    for i in range(len(imoo[0])):
        if i == 0:
            orix = np.append(origine , origine*imoo[:,i].reshape(imoo.shape[0],1), axis=1)
        else:
            orix = np.append(orix , orix[:,-1].reshape(imoo.shape[0],1)*imoo[:,i].reshape(imoo.shape[0],1), axis=1)
    return(np.rot90(orix,-1))

def data_converter_and_cleaner(initial_data):
    dataset = np.nan_to_num(np.rot90(np.array([(i[1:]/i[:-1]) for i in initial_data]),-1), copy=False, nan=1, posinf=1, neginf=1)
    dataset[dataset == 0] = 1
    return dataset

def packing(a,stop):
    cn = 0
    ls_f=[]
    for sb in a:
        for ib in sb:
            ls_f+=[ib]
            cn+=1
            if cn == stop:
                return ls_f
    return ls_f

def predictor(dataset,block_size_to_predict_next_block=10,block_size_bein_predict=5,mod=[1]):
    print("compute cluster_1")
    first_cluster_size = int(len(dataset)/2.5)
    first_cl = KMeans(n_clusters=first_cluster_size,random_state=0).fit(dataset)
    mydict = {i: dataset[np.where(first_cl.labels_ == i)[0]] for i in range(first_cl.n_clusters)}
    print("loading datas clust")
    #minimum 2
    dataset_cluster = first_cl.predict(dataset)
    block_size_to_predict_next_block = block_size_to_predict_next_block
    block_size_bein_predict = block_size_bein_predict

    #dataset_clustering_generator
    block_use_to_forecast  = np.array([dataset_cluster[i:i+block_size_to_predict_next_block] for i in range(0,len(dataset_cluster)-block_size_to_predict_next_block+1)])
    block_cluster_forecasted  = np.array([dataset_cluster[i:i+block_size_bein_predict] for i in range(0,len(dataset_cluster)-block_size_bein_predict+1)])
    print("compute cluster_2")
    #cluster_model
    cluster_size_of_forecast_block = [int(x) if x > 3 else 3 for x in [int(len(block_cluster_forecasted)/2.5)]][0]
    cluster_of_forecasted_block = KMeans(n_clusters=cluster_size_of_forecast_block,random_state=0).fit(block_cluster_forecasted)

    #clustering
    mydict2 = {i: block_cluster_forecasted[np.where(cluster_of_forecasted_block.labels_ == i)[0]] for i in range(cluster_of_forecasted_block.n_clusters)}
    print("shaping dataset")
    #dataset
    dataseter = []
    dataset = []
    for i in range(len(block_use_to_forecast)):
        block_to_learn = np.array(block_use_to_forecast[i])
        block_to_predict = np.array(packing(block_use_to_forecast[i+1:],block_size_bein_predict))
        if len(block_to_predict) != block_size_bein_predict:
            pass
        else:
            dataseter.append([block_to_learn,block_to_predict])

    lsor = len(block_cluster_forecasted)
    block_to_predict_clustered = cluster_of_forecasted_block.predict(np.array([i[-1] for i in dataseter[:]], dtype='int32'))
    dataset = [np.append(dataseter[i][0],block_to_predict_clustered[i]) for i in range(len(block_to_predict_clustered))]
    print("Start compute ML")       
    #prediction_ml
    res1,_ = ml_filter(dataset,numww=1,model_on=mod)
    print("return result of ML")
    #result_transformatiom
    try:
        result_first_method,class_f  = [[[[mydict[y][:,i].mean() for i in range(len(mydict[y][0]))] for y in mydict2[i[0]][z]] for z in range(len(mydict2[i[0]]))]for i in res1],res1
        print("DONE")
        return result_first_method,class_f
    except:
        return [mydict,mydict2],res1

def klinedata(symb='BTCUSDT'):
    overall=[]
    ui = 18
    for rum in range(ui):
        daist =720-(40*(rum)) 
        daiet =720-(40*(rum+1))   
        st = datetime.now() - timedelta(days=daist, hours=0)
        start_time = str(int(st.timestamp() * 1000))
        et = datetime.now() - timedelta(days=daiet, hours=0)
        end_time = int(et.timestamp() * 1000)
        response = requests.get(f'https://api.binance.com/api/v3/klines?symbol={symb}&interval=1h&startTime={start_time}&endTime={end_time}&limit=1000')
        time.sleep(2)
        res = ast.literal_eval(response.__dict__['_content'].decode('UTF-8'))
        overall+=res
        print(datetime.fromtimestamp(res[0][0]/1000.0),datetime.fromtimestamp(res[-1][0]/1000.0),daist,daiet)
    res_rf = [[float(h) for h in i] for i in overall]
    return res_rf

def ploxyx(x,y,verbose=True):
    result=[]
    mk = x
    for h in range(len(x)):
        print(f"prediction index| {h} |")
        for e in range(len(mk[h][0])):
            rm1 = [i[e] for i in mk[h][:-1]]
            rm2 = y[e]
            rm3 = gaussian_filter(rm1, sigma=8)
            if verbose:
                plt.plot(rm1,'-k')
                plt.plot(rm2,'-b')
                plt.plot(rm3,'-r')
                plt.show()
            result.append([rm1,rm2,rm2])
    return result

def ploxyxx(x,verbose=True):
    result=[]
    mk = x
    for h in range(len(x)):
        print(f"prediction index| {h} |")
        for e in range(len(mk[h][0])):
            rm1 = [i[e] for i in mk[h][:-1]]
            rm3 = gaussian_filter(rm1, sigma=8)
            if verbose:
                plt.plot(rm1,'-k')
                plt.plot(rm3,'-r')
                plt.show()
            result.append([rm1,rm3])
    return result
 
def mode_model(x):
    model = {'DecisionTreeClassifier': 1 ,'RandomForestClassifier':2,'SVC':3,'KNeighborsClassifier':4,'GaussianNB':5,'MLPClassifier':6}
    resume_model=[model[v] for v in x]
    return resume_model

def mode_data(x):
    candle = {"open_time":0,"open":1,"high":2,"low":3,"close":4,"volume":5,"close_time":6,"quote_asset_volume":7,"number_of_trades":8,"taker_by_qav":9,"taker_by_bav":10,"ignored":11}
    resume_data=[candle[v] for v in x]
    return resume_data

        
def ml_presage_predict_crypto(symbol='BTCUSDT',
                            block_size_to_predict_next_block=800, 
                            block_size_bein_predict=100, 
                            ml_model=['DecisionTreeClassifier','RandomForestClassifier','SVC','KNeighborsClassifier','GaussianNB','MLPClassifier'],
                            forecast_candle_1h=['open','high','low','close'],
                            verbose=True):
    
    if len(forecast_candle_1h) < 2 or len(ml_model) < 1 or len(symbol) < 0:
        return print("need at least 2 candle or one model or one symbol")
    
    resum=[]
    datalo = klinedata(symb=symbol)
    # open_time,open,high,low,close,volume,close_time,quote_asset_volume,number_of_trades,taker_by_qav,taker_by_bav,ignored
    database = [[ i[h] for i in datalo] for h in mode_data(forecast_candle_1h)]

    stom = None
    stam = -4000

    initial_data = np.array([h[stam:stom] for h in database]).astype(np.float) 
    
    dataset = data_converter_and_cleaner(initial_data)
    #Prediction
    result_first_method,class_f = predictor(dataset,block_size_to_predict_next_block=block_size_to_predict_next_block,
                                            block_size_bein_predict=block_size_bein_predict,
                                            mod = mode_model(ml_model))

    model={}
    for i in range(len(result_first_method)):
        print(f"Result Model {i} |you will get 2 possible outcome. Can recover the model output in dict -> model , index: model[Model_{i}|")
        print([rebuilder(initial_data[:,-10:],np.array(result_first_method[i][h])) for h in range(len(result_first_method[i]))])
        model[f"Model_{i}"]=[rebuilder(initial_data[:,-10:],result_first_method[0][i])]
    ###############################################################################################################################
        

    for i in range(len(model)):
        print(f"######################## Model_{ml_model[i]} ##########################")
        res0 = model[f"Model_{i}"]
        resh = ploxyxx(res0,verbose=verbose)
        resum.append(resh) 
    return 
    
    
def ml_presage_predict_crypto_back_test(symbol='BTCUSDT',block_size_to_predict_next_block=800,
                                        block_size_bein_predict=100,
                                        ml_model=['DecisionTreeClassifier','RandomForestClassifier','SVC','KNeighborsClassifier','GaussianNB','MLPClassifier'],
                                        forecast_candle_1h=['open','high','low','close'],
                                        verbose=True):
    
    
    if len(forecast_candle_1h) < 2 or len(ml_model) < 1 or len(symbol) < 0:
        return print("need at least 2 candle or one model or one symbol")
    
    resum=[]
    datalo = klinedata(symb=symbol)
    database = [[ i[h] for i in datalo] for h in mode_data(forecast_candle_1h)]


    for yt in range(4000,12000,block_size_bein_predict):
        stam = -yt
        if stam+block_size_bein_predict == 0:
            stom = None
        else:
            stom = stam -(stam+block_size_bein_predict)

        initial_data = np.array([h[stam:stom] for h in database]).astype(np.float)  
        
        dataset = data_converter_and_cleaner(initial_data)
        #Prediction
        result_first_method,class_f = predictor(dataset,block_size_to_predict_next_block=block_size_to_predict_next_block,
                                                block_size_bein_predict=block_size_bein_predict,
                                                mod = mode_model(ml_model))

        model={}
        for i in range(len(result_first_method)):
            print(f"Result Model {i} |you will get 2 possible outcome. Can recover the model output in dict -> model , index: model[Model_{i}|")
            print([rebuilder(initial_data[:,-10:],np.array(result_first_method[i][h])) for h in range(len(result_first_method[i]))])
            model[f"Model_{i}"]=[rebuilder(initial_data[:,-10:],result_first_method[0][i])]
        ###############################################################################################################################
                    

        nyp = stom
        if nyp+100 == 0:
            nypp = None
        else:
            nypp = nyp+100

        ini_b= np.array([h[nyp:nypp] for h in database]).astype(np.float)  
            

        for i in range(len(model)):
            print(f"######################## Model_{ml_model[i]} ##########################")
            res0 = model[f"Model_{i}"]
            resh = ploxyx(res0,ini_b,verbose=verbose)
            resum.append(resh)
    return resum
        

        
    
def ml_presage_predict_generalize(dataset_global =[list_item1_same_len,list_item2_same_len,list_item3_same_len],
                       dataset_global_slicer=slice(-4000,None),
                       block_size_to_predict_next_block=800, 
                       block_size_bein_predict=100, 
                       ml_model=['DecisionTreeClassifier','RandomForestClassifier','SVC','KNeighborsClassifier','GaussianNB','MLPClassifier'],
                       verbose=True):
    
    if len(dataset_global) < 2 or 0 in [len(dataset_global[i]) for i in range(len(dataset_global))] :
        return print(" dataset_global get less than 2 list or not enough data in list")
    

    initial_data = np.array([h[dataset_global_slicer] for h in dataset_global]).astype(np.float) 
    dataset = data_converter_and_cleaner(initial_data)
    #Prediction
    result_first_method,class_f = predictor(dataset,block_size_to_predict_next_block=block_size_to_predict_next_block,
                                            block_size_bein_predict=block_size_bein_predict,
                                            mod = mode_model(ml_model))

    model={}
    for i in range(len(result_first_method)):
        print(f"Result Model {i} |you will get 2 possible outcome. Can recover the model output in dict -> model , index: model[Model_{i}|")
        print([rebuilder(initial_data[:,-10:],np.array(result_first_method[i][h])) for h in range(len(result_first_method[i]))])
        model[f"Model_{i}"]=[rebuilder(initial_data[:,-10:],result_first_method[0][i])]
    ###############################################################################################################################
        

    for i in range(len(model)):
        print(f"######################## Model_{ml_model[i]} ##########################")
        res0 = model[f"Model_{i}"]
        resh = ploxyxx(res0,verbose=verbose)
        resum.append(resh) 
    return resum

