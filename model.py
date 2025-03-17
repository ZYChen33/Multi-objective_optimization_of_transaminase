import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor as GBRT
from skopt.learning import RandomForestRegressor as RF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error
import os

#normalization
def fscaler(Xtrain,Xtest):
    scaler=MinMaxScaler()
    scaler=scaler.fit(Xtrain)
    Xtr_scale=scaler.transform(Xtrain)
    Xte_scale=scaler.transform(Xtest)

    return Xtr_scale,Xte_scale

#select hyperparameters
def select_best_param(X_tr,y_tr,model='gbrt'):
    if model=='gbrt':
        reg=GBRT(random_state=0)
        param_grid={'n_estimators':[400] #np.arange(50,500,50)
                    ,'learning_rate':[0.10]  #np.arange(0.05,0.4,0.05)
                    ,'max_depth':[10] #np.arange(3,10,1)
                    ,'loss':['quantile']
                    ,'min_samples_leaf':[2] #np.arange(1,20,1)
                    ,'min_samples_split':[2]
                    ,'subsample':[0.6] #np.arange(0.5,0.8,0.05)
                    }

    elif model=='rf':
        reg=RF(random_state=0)
        param_grid={'n_estimators':[200] #np.arange(50,500,50)
                    ,'max_depth':[12] #np.arange(3,15,1)
                    ,'min_samples_split':[4] #np.arange(2,20,1)
                    ,'min_samples_leaf':[2] #np.arange(1,20,1)
                    }
        
    elif model=='gpr':
        X_tr,_=fscaler(X_tr,X_tr)
        kernel = RBF(length_scale=1.0)
        reg=GPR(random_state=0, kernel=kernel)
        param_grid={'alpha':[0.1] #np.arange(0.1,1.0,0.1)
                    ,'n_restarts_optimizer':[10] #np.arange(10,50,10)
                   }

    GS=GridSearchCV(reg,param_grid,cv=5,scoring='r2')
    GS.fit(X_tr,y_tr)
    best_param=GS.best_params_
    print(best_param)

    return best_param

#print score
def print_score(X_tr,X_te,y_tr,y_te,best_param,y_min,y_max,current_target,model='gbrt'):
    if model=='gbrt':
        reg=GBRT(random_state=0,**best_param)
    elif model=='rf':
        reg=RF(random_state=0,**best_param)
    elif model=='gpr':
        X_tr,X_te=fscaler(X_tr,X_te)
        kernel = RBF(length_scale=1.0)
        reg=GPR(random_state=0, kernel=kernel,**best_param)

    reg_fitted=reg.fit(X_tr,y_tr)
    train_score=np.around(reg_fitted.score(X_tr,y_tr),3)
    test_score=np.around(reg_fitted.score(X_te,y_te),3)
    print('training set r2: %3f'% train_score)
    print('test set r2: %3f'% test_score)
    ytr_pred=reg_fitted.predict(X_tr)
    yte_pred=reg_fitted.predict(X_te)
    mae_tr=np.around(MAE(y_tr,ytr_pred),3)
    mae_te=np.around(MAE(y_te,yte_pred),3)
    print('training set mae: %3f' % mae_tr)
    print('test set mae: %3f' % mae_te)
    rmse_tr=np.around(np.sqrt(mean_squared_error(y_tr,ytr_pred)),3)
    rmse_te=np.around(np.sqrt(mean_squared_error(y_te,yte_pred)),3)
    print('training set rmse: %3f' % rmse_tr)
    print('test set rmse: %3f' % rmse_te)
    
    os.makedirs('output', exist_ok=True)
    plt_title = f"{model}-{current_target}"
    save_path = os.path.join('output', f'{plt_title}.jpg')
        
    fig=plt.figure(figsize=(8,8))
    plt.scatter(y_tr,ytr_pred,s=30,c="b",alpha=0.8)
    plt.scatter(y_te,yte_pred,s=30,c="r",alpha=0.8)
    plt.xlim((y_min,y_max))
    plt.ylim((y_min,y_max))
    x_=np.array([y_min,y_max])
    y_=x_
    plt.plot(x_,y_,c="k")
    plt.xlabel('True Values',fontsize=20)
    plt.ylabel('Predictions',fontsize=20)
    plt.title(plt_title,fontsize=22)
    t='training set R\u00b2=%s\ntest set R\u00b2=%s\ntraining set RMSE=%s\ntest set RMSE=%s' % (train_score,test_score,rmse_tr,rmse_te)
    plt.text(y_min+0.5,y_max-0.5,t,fontsize=20,verticalalignment="top",horizontalalignment="left")

    plt.savefig(save_path,dpi=140)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
