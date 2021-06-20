import time
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import os
import ta
import datetime 

#################################################
#Class And Attributes Definiton
class trade:
    def __init__(self):
        ###################
        #Initioal Variabels
        global timeframe_index,Time_Frame

        self.User=None
        self.Password=None
        self.server="MetaQoutes-Demo"
        self.path="C:\\Program Files\\MetaTrader 5\\terminal64.exe"
      
        self.Start_Candle=1
        self.Time_Span=5000

        self.deal_start_gmt=8
        self.deal_stop_gmt=23

        self.timeframe_index="M5"
        self.symbol="EURUSD"
        
        self.Time_Frame={
        "M1":mt5.TIMEFRAME_M1,
        "M2":mt5.TIMEFRAME_M2,
        "M3":mt5.TIMEFRAME_M3,
        "M4":mt5.TIMEFRAME_M4,
        "M5":mt5.TIMEFRAME_M5,
        "M6":mt5.TIMEFRAME_M6,
        "M10":mt5.TIMEFRAME_M10,
        "M12":mt5.TIMEFRAME_M12,
        "M15":mt5.TIMEFRAME_M15,
        "M20":mt5.TIMEFRAME_M20,
        "M30":mt5.TIMEFRAME_M30,
        "H1":mt5.TIMEFRAME_H1,
        "H2":mt5.TIMEFRAME_H2,
        "H3":mt5.TIMEFRAME_H3,
        "H4":mt5.TIMEFRAME_H4,
        "H4":mt5.TIMEFRAME_D1,
        }
        self.Repeat_Dpl_Train=False
        return None

    def Get_rates(self,symbol=None,tmf_index=None,st_cndl=None,tm_spn=None):
        if symbol==None:
            symbol=self.symbol
        if tmf_index==None:
            tmf_index=self.timeframe_index
        if st_cndl==None:
            st_cndl=self.Start_Candle
        if tm_spn==None:
            tm_spn=self.Time_Span
        global rates_df
        if not mt5.initialize(path=self.path,login=self.User,server=self.server, password=self.Password):
            print("initialize() failed, error code =",mt5.last_error())
            quit()
        rates = mt5.copy_rates_from_pos(symbol,self.Time_Frame[tmf_index], st_cndl, tm_spn)
        rates_df=pd.DataFrame(rates)
        rates_df['second']=rates_df['time']
        rates_df['time']=pd.to_datetime(rates_df['time'], unit='s')
        rates_df['HLC']=(rates_df['high']+rates_df['low']+rates_df['close'])/3
        return rates_df    

    ##################################################
    #Functions Definition

    def minimum_n_candle(self,n=10):
        self.Get_rates()
        min_value=rates_df.iloc[-1]['low']
        min_index=-1
        for i in range(n):
            if rates_df.iloc[-1-i]['low']<min_value :
                min_value=rates_df.iloc[-1-i]['low']
                min_index=i
        mt5.shutdown()
        return min_value,min_index

    def maximum_n_candle(self,n=10):
        self.Get_rates()
        max_value=rates_df.iloc[-1]['low']
        max_index=-1
        for i in range(n):
            if rates_df.iloc[-1-i]['low']>max_value :
                max_value=rates_df.iloc[-1-i]['low']
                max_index=i
        mt5.shutdown()
        return max_value,max_index
    
    def buy_order(self,symbol=None,price=None,volume=None,stop=None):
        self.Get_rates()
        if symbol==None:
            symbol=self.symbol

        if price==None:
            price=mt5.symbol_info_tick(symbol).ask
        #if stop==None:
        #    stop=mt5.symbol_info_tick(symbol).ask
        request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price":price,
        "sl": stop,#allocate stop lost equal one pointlower than low price of last candle 
        #"tp": tp,
        "deviation": 2,
        "magic": 1400,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
        }
        # send a trading request
        result = mt5.order_send(request)
        
        # check the execution result
        if result!=None and result.retcode==mt5.TRADE_RETCODE_DONE:
            print("Buy_order_sent:")
            mt5.shutdown()
            return result.order
            
        elif result==None:
            print("Buy order send failed")
            mt5.shutdown()
            return None
        elif result.retcode!=mt5.TRADE_RETCODE_DONE and result.retcode!=None:
            print("retcod is :",result.retcode)
            mt5.shutdown()
            return None
            
    def sell_order(self,symbol=None,price=None,volume=None,stop=None):
        self.Get_rates()
        if symbol==None:
            symbol=self.symbol

        if price==None:
            price=mt5.symbol_info_tick(symbol).bid
        #if stop==None:
        #    stop=mt5.symbol_info_tick(symbol).bid

        request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL,
        "price":price,
        "sl": stop,#allocate stop lost equal one pointlower than low price of last candle 
        #"tp": tp,
        "deviation": 2,
        "magic": 1400,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
        }

        # send a trading request
        result = mt5.order_send(request)
        
        # check the execution result
        if result!=None and result.retcode==mt5.TRADE_RETCODE_DONE:
            print("Buy_order_sent")
            mt5.shutdown()
            return result.order
        elif result==None:
            print("Buy order send failed")
            mt5.shutdown()
            return None
        elif result.retcode!=mt5.TRADE_RETCODE_DONE and result.retcode!=None:
            print("retcod is :",result.retcode)
            mt5.shutdown()
            return None

    def close_order(self,ticket=None):
        self.Get_rates()
        res=[]
        if ticket==None :
            positions=mt5.positions_get()
            if positions==None:
                print("There is No Open position")
                return None
            elif len(positions)>0:
                pass

        elif ticket!=None :
            positions=mt5.positions_get(ticket=ticket)
        
        for position in positions:
            if position.type==mt5.ORDER_TYPE_BUY :
                order_type=mt5.ORDER_TYPE_SELL
            elif  position.type==mt5.ORDER_TYPE_SELL:
                order_type=mt5.ORDER_TYPE_BUY
                
            request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "price":position.price_current,
            "deviation": 2,
            "magic": 1400,
            "comment": "python script",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
            }

            # send a trading request
            result = mt5.order_send(request)
            
            # check the execution result
            if result!=None and result.retcode==mt5.TRADE_RETCODE_DONE:
                print("Buy_order_sent:")
                res.append(result.order)
            elif result==None:
                print("Buy order send failed")
                
            elif result.retcode!=mt5.TRADE_RETCODE_DONE and result.retcode!=None:
                print("retcod is :",result.retcode)
        mt5.shutdown()
        return res

    def fractal(self,n=None):
        if n==None:
            no_candles=len(rates_df)
        else :
            no_candles=n

        self.Get_rates()
        fractal_index=[]
        fractal_second=[]
        fractal_max_min=[]
        
        for i in range(2,no_candles-3):
            if rates_df.iloc[i]['high']>=max(rates_df.iloc[i+1]['high'],rates_df.iloc[i+2]['high'],rates_df.iloc[i-1]['high'],rates_df.iloc[i-2]['high']):
                fractal_index.append(i)
                fractal_second.append(rates_df.iloc[i]['second'])
                fractal_max_min.append("max")

            if rates_df.iloc[i]['low']<=min(rates_df.iloc[i+1]['low'],rates_df.iloc[i+2]['low'],rates_df.iloc[i-1]['low'],rates_df.iloc[i-2]['low']):
                fractal_index.append(i)
                fractal_second.append(rates_df.iloc[i]['second'])
                fractal_max_min.append("min")
        fractal_dict={"index":fractal_index,"second":fractal_second,"max_min":fractal_max_min}
        fractal=pd.DataFrame(fractal_dict)
        return fractal

    def line_equ():
        pass

    def Dpl_LSTM_train(self,symbol=None,time_frame_index=None,n_backlook=None,n_prdict=None,plot=False):
        
        #importing required libraries
        from sklearn.preprocessing import MinMaxScaler
        from keras.models import Sequential
        from keras.layers import Dense, Dropout,LSTM
        from keras.models import load_model
        ###############################################
        if symbol==None:
            symbol=self.symbol
        if n_backlook==None :
            n_backlook=60
        if n_prdict==None:
            n_prdict=1
        if time_frame_index==None:
            time_frame_index=self.timeframe_index
        n_forward=1
        plot_from=50
        ################################################
        Repeat_check=True
        while Repeat_check==True :
            #creating dataframe
            df = self.Get_rates(symbol,time_frame_index)
            data = df.sort_index(ascending=True, axis=0)
            new_data = pd.DataFrame(index=range(0,len(df)),columns=['index','time', 'close'])
            for i in range(0,len(data)):
                new_data['time'][i] = data['time'][i]
                new_data['close'][i] = data['close'][i]
                new_data['index'][i] = i
            ################################################
            #setting index
            new_data.index = new_data.index
            new_data.drop('time', axis=1, inplace=True)
            new_data.drop('index', axis=1, inplace=True)
            ################################################
            #creating train and test sets
            dataset = new_data.values
            train = dataset
            #################################################
            #converting dataset into x_train and y_train
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(train)
            #################################################
            #Finding last Trained index to continue training
            second=0
            file_name=None
            index_last_train=0
            must_compile=False
            dir_file_names=os.listdir()
            for name in dir_file_names:
                if name.find(symbol+"_"+time_frame_index)==0:
                    file_name=name
                    print(name)
                    
            if file_name==None:
                print("File name None")
                must_compile=True
            elif not file_name==None:
                second=file_name[len(symbol+"_"+time_frame_index+"_"):-3]
                print(second)
                if second==df.iloc[-2]['second']:
                    print("new data not found")
                    continue
                elif not second==df.iloc[-2]['second']:
                        for i in range(len(df.iloc[:]['second'])-1):
                            if df.iloc[-2-i]['second']==int(second) :
                                location=np.where(df.isin([int(second)])==True)
                                index_last_train=location[0][0]
                            
            if index_last_train<n_backlook:
                print(index_last_train)
                index_last_train=n_backlook
                must_compile=True
            
            ######################################################
            #Setting X_train and Y_train
            x_train, y_train = [], []
            for i in range(index_last_train+n_forward,len(train)):
                x_train.append(scaled_data[i-n_backlook:i,0])
                y_train.append(scaled_data[i,0])
            #print("start candle",df.iloc[index_last_train+n_forward]['second'])
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
            ######################################################################################
            # create and fit the LSTM network
            if must_compile==True:
                model = Sequential()
                model.add(LSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1],1)))
                model.add(LSTM(units=60))
                model.add(Dense(1))
                model.compile(loss='mean_squared_error', optimizer='adam')
                print("Model Compiled")
            elif must_compile==False : 
                model = load_model(file_name)
                print("Model Loaded")
            
            model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
            model.save(symbol+"_"+time_frame_index+"_"+str(df.iloc[-2]["second"])+'.h5')
            
            del model
            #return_filename=symbol+"_"+time_frame_index+"_"+df.iloc[-2]["second"]+'.h5'
            if file_name!=file_name:
                os.remove(file_name)
            Repeat_check=self.Repeat_Dpl_Train
        return None

    def Dpl_LSTM_predicte(self,symbol=None,time_frame_index=None,n_backlook=None,n_prdict=None,plot=False):

        #importing required libraries

        from sklearn.preprocessing import MinMaxScaler
        from keras.models import Sequential
        from keras.layers import Dense, Dropout,LSTM
        from keras.models import load_model
        ###############################################
        if symbol==None:
            symbol=self.symbol
        if n_backlook==None :
            n_backlook=60
        if n_prdict==None:
            n_prdict=1
        if time_frame_index==None:
            time_frame_index=self.timeframe_index
        n_forward=1

        ################################################
        #creating dataframe
        df = self.Get_rates(symbol=symbol,tmf_index=time_frame_index,st_cndl=1,tm_spn=n_backlook)
        data = df.sort_index(ascending=True, axis=0)
        new_data = pd.DataFrame(index=range(0,len(df)),columns=['index','time', 'close'])
        for i in range(0,len(data)):
            new_data['time'][i] = data['time'][i]
            new_data['close'][i] = data['close'][i]
            new_data['index'][i] = i
        print(new_data)
        #setting index
        new_data.index = new_data.index
        new_data.drop('time', axis=1, inplace=True)
        new_data.drop('index', axis=1, inplace=True)
      
        #creating train and test sets
        dataset = new_data.values
        train = dataset

        #converting dataset into x_train and y_train
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(train)

        file_name=None        
        dir_file_names=os.listdir()
        for name in dir_file_names:
            if name.find(symbol+"_"+time_frame_index)==0:
                file_name=name
                print(file_name)
        
        '''                   
        if file_name==None:
            self.Dpl_LSTM_train()
        '''
        ##########################################
        # Load the LSTM network
        model = load_model(file_name)
        
        ##########################################
        #predicting values using past 60 from the train data
        inputs = new_data[len(new_data)  - n_backlook:].values
        inputs = inputs.reshape(-1,1)
        inputs  = scaler.transform(inputs)

        X_test = []
        X_test.append(inputs[:,0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        return closing_price
    
    def Dpl_Multiprocess_traning(self):
        from multiprocessing import Process
        if __name__ == '__main__':
            p = Process(target=self.Dpl_LSTM_train)
            p.start()
            print("Process PID is ",str(os.getpid()))
        return None

#################################################
class1=trade()
class1.User=50896888
class1.Password="aaug7dkp"
class1.server="Alpari-MT5-Demo"
class1.Repeat_Dpl_Train=False
#class1.Dpl_LSTM_train()
a=class1.Dpl_LSTM_predicte()
print("learning starter",a)

