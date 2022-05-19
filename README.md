Finding clusters of invariant structure using k-means algorithm and generate forecast with the next structure having the highest probability of occurrence. Testing multiple machine learning simple model (without hyperparameter tuning, model analysis or cloud pipeline). One can use any of the following model available:
|'DecisionTreeClassifier','RandomForestClassifier','SVC','KNeighborsClassifier','GaussianNB','MLPClassifier'| 
for machine learning.


##Install dependencies
```
pip install -r /path/to/pip_requirment.txt
```

##Code example (Run code on jupyterlab or jupyter if verbose True)

```
import ml_presage


# || block_size_to_predict_next_block=800 || is the lenght of the information use to forecast the next ||block_size_bein_predict=100||

#Different parameter for crypto:
#    forecast_candle_1h= ["open_time","open",
#                            "high","low",
#                            "close","volume",
#                            "close_time","quote_asset_volume",
#                            "number_of_trades","taker_by_qav",
#                            "taker_by_bav"]

#Different model to use:
#    ml_model= ['DecisionTreeClassifier','RandomForestClassifier','SVC','KNeighborsClassifier','GaussianNB','MLPClassifier'],



#Predict the next 100 hours of BTCUSDT candle value chosen (time to compute < 11min):

result = ml_presage_predict_crypto(symbol='BTCUSDT',
                             block_size_to_predict_next_block=800, 
                             block_size_bein_predict=100, 
                             ml_model=['DecisionTreeClassifier'],
                             forecast_candle_1h=['open','high','low','close'],
                             verbose=True)
    


#Predict the past 100 hours BTCUSDT candle value chosen. &but
#Loop on past 100hours and predict candle value chosen and remove future value from the dataset.
#Recompute new model for each iteration:

result = ml_presage_predict_crypto_back_test(symbol='BTCUSDT',
                                    block_size_to_predict_next_block=800,
                                    block_size_bein_predict=100,
                                    ml_model=['DecisionTreeClassifier','RandomForestClassifier','SVC','KNeighborsClassifier','GaussianNB','MLPClassifier'],
                                    forecast_candle_1h=['open','high','low','close'],
                                    verbose=True)
    
    

#Example dataset: dataset_global is shape like [[x1,x2...xX]...[y1,y2...yY]] then get convert to [[x1,y1],[x2,y2]...]        
#Generalize the ml_presage to any type of input list of int and float and predict the future value of lenght block_size_bein_predict:

result = ml_presage_predict_generalize(dataset_global =[list_item1_same_len,list_item2_same_len,list_item3_same_len],
                                dataset_global_slicer=slice(-4000,None),
                                block_size_to_predict_next_block=800, 
                                block_size_bein_predict=100, 
                                ml_model=['DecisionTreeClassifier','RandomForestClassifier','SVC','KNeighborsClassifier','GaussianNB','MLPClassifier'],
                                verbose=True)

```
