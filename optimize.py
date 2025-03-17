import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor as GBRT
from skopt.learning import GradientBoostingQuantileRegressor as BOGBRT
from skopt.learning import RandomForestRegressor as RF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
import pareto
import model

def get_predicted(data_df, explore, best_params_dict, num_objective=2, modelname='rf', acq=None):
    regs_fitted = {}

    if modelname=='gbrt':
        for i in range(num_objective):
            A_X = np.array(data_df.iloc[:,:-num_objective])
            A_y = np.around(np.array(data_df.iloc[:,-i-1]),6)

            X_shuffle,y_shuffle = shuffle(A_X,A_y,random_state=0)
            X_tr,X_te,y_tr,y_te = train_test_split(X_shuffle,y_shuffle,test_size=0.15,random_state=0)

            if acq:
                acq_df = pd.read_csv('acq.csv').iloc[:,0:]
                X_acq = np.array(acq_df.iloc[:,:-num_objective])
                y_acq = np.around(np.array(acq_df.iloc[:,-i-1]),6)
                X_tr = np.concatenate((X_tr,X_acq))
                y_tr = np.concatenate((y_tr,y_acq))

            best_param = best_params_dict[f'objective_{i+1}']
            reg = GBRT(random_state=0,**best_param)
            reg_bo = BOGBRT(quantiles=[0.16, 0.5, 0.84],base_estimator=reg)
            reg_fitted = reg_bo.fit(X_tr,y_tr)
            regs_fitted[f'objective_{i+1}'] = reg_fitted

    elif modelname=='rf':
        for i in range(num_objective):
            A_X = np.array(data_df.iloc[:,:-num_objective])
            A_y = np.around(np.array(data_df.iloc[:,-i-1]),6)

            X_shuffle,y_shuffle = shuffle(A_X,A_y,random_state=0)
            X_tr,X_te,y_tr,y_te = train_test_split(X_shuffle,y_shuffle,test_size=0.15,random_state=0)

            if acq:
                acq_df = pd.read_csv('acq.csv').iloc[:,0:]
                X_acq = np.array(acq_df.iloc[:,:-num_objective])
                y_acq = np.around(np.array(acq_df.iloc[:,-i-1]),6)
                X_tr = np.concatenate((X_tr,X_acq))
                y_tr = np.concatenate((y_tr,y_acq))

            best_param = best_params_dict[f'objective_{i+1}']
            reg = RF(random_state=0,**best_param)
            reg_fitted = reg.fit(X_tr,y_tr)
            regs_fitted[f'objective_{i+1}'] = reg_fitted

    elif modelname=='gpr':
        for i in range(num_objective):
            A_X = np.array(data_df.iloc[:,:-num_objective])
            A_y = np.around(np.array(data_df.iloc[:,-i-1]),6)

            X_shuffle,y_shuffle = shuffle(A_X,A_y,random_state=0)
            X_tr,X_te,y_tr,y_te = train_test_split(X_shuffle,y_shuffle,test_size=0.15,random_state=0)

            if acq:
                acq_df = pd.read_csv('acq.csv').iloc[:,0:]
                X_acq = np.array(acq_df.iloc[:,:-num_objective])
                y_acq = np.around(np.array(acq_df.iloc[:,-i-1]),6)
                X_tr = np.concatenate((X_tr,X_acq))
                y_tr = np.concatenate((y_tr,y_acq))

            X_tr,X_te = model.fscaler(X_tr,X_te)
            X_tr,explore = model.fscaler(X_tr,explore)
            best_param = best_params_dict[f'objective_{i+1}']
            kernel = RBF(length_scale=1.0)
            reg=GPR(random_state=0, kernel=kernel,**best_param)
            reg_fitted = reg.fit(X_tr,y_tr)
            regs_fitted[f'objective_{i+1}'] = reg_fitted

    means_all = []
    variances_all = []
    for i in range(num_objective):
        mean, variance = regs_fitted[f'objective_{i+1}'].predict(explore, return_std=True)
        means_all.append(mean)
        variances_all.append(variance)

    means_all = np.array(means_all)
    variances_all = np.array(variances_all)
    means_all_abs = np.abs(means_all)
    means = np.column_stack(means_all_abs)
    variances = np.column_stack(variances_all)

    return means, means_all, variances, variances_all

def get_pareto(data_df, num_objective, pareto_plt, acq):
    if acq:
        train_y = np.around(np.array(data_df.iloc[:,-num_objective:]),6)
        acq_df = pd.read_csv('acq.csv').iloc[:,0:]
        train_y_acq = np.around(np.array(acq_df.iloc[:,-num_objective:]),6)
        train_y = np.concatenate((train_y,train_y_acq))
    else:
        train_y = np.around(np.array(data_df.iloc[:,-num_objective:]),6)

    train_y_abs = np.abs(train_y)

    pareto_train_y = pareto.eps_sort(tables=train_y_abs, maximize_all=True)
    pareto_train_y = np.reshape(pareto_train_y, (-1, num_objective))

    if pareto_plt:
        print("Pareto frontier:")
        print(pareto_train_y)
        np.savetxt(f'.\output\pareto_front.csv', pareto_train_y, delimiter=",")

        if num_objective == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(train_y_abs[:,0], train_y_abs[:,1],
                        c='gray', alpha=0.6, label='All Solutions')

            pareto_sorted = pareto_train_y[np.argsort(pareto_train_y[:, 0])]
            plt.plot(pareto_sorted[:,0], pareto_sorted[:,1],
                     'r--', lw=2, label='Pareto Front')
            plt.scatter(pareto_train_y[:,0], pareto_train_y[:,1],
                        c='red', s=80, edgecolors='k', label='Pareto Solutions')

            plt.xlabel(data_df.columns[-2], fontsize=12)
            plt.ylabel(data_df.columns[-1], fontsize=12)
            plt.title('Pareto Front Visualization', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(f'.\output\pareto_front.jpg', dpi=300, bbox_inches='tight')
            plt.show()

    return pareto_train_y

def score_function(means, kappa, variances, pareto_train_y):
    score = []
    for i in range(0, len(means)):
        diff = (means[i] + kappa * variances[i]) - pareto_train_y
        diff_min = np.min(diff, axis=0)
        score_i = np.max(diff_min)
        score.append(score_i)

    return score

def screening(data_df, explore, best_params_dict, num_objective=2, modelname='rf', num_recommend=1, kappa=1, acq=None, pareto_plt=True):
    means_abs, means_all, variances, variances_all = get_predicted(data_df, explore, best_params_dict, num_objective, modelname, acq)
    pareto_train_y = get_pareto(data_df, num_objective, pareto_plt, acq)
    score = score_function(means_abs, kappa, variances, pareto_train_y)

    top_n_indices = np.argsort(score)[-num_recommend:][::-1]
    recommend_samples = explore[top_n_indices].tolist()
    indices = (top_n_indices + 1).tolist()

    column_names1 = [f'mean_{i+1}' for i in range(num_objective)]
    column_names2 = [f'variance_{i+1}' for i in range(num_objective)]
    column_names3 = ['score']
    column_name = column_names1+column_names2+column_names3

    explore_result = np.column_stack((means_all.T,variances_all.T,score))
    explore_result_df = pd.DataFrame(explore_result,columns=column_name)
    explore_result_df.to_csv(f'.\output\screening.csv')

    return recommend_samples, indices


