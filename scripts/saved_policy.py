import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

n = 30
# sigma=.00001
sigma=.1

sparse_vecs = set()

for i in range(n):
    sparse_vec = np.zeros(n)
    sparse_vec[i] = 1
    sparse_vecs.add(tuple(sparse_vec))

# for i in range(n):
#     for j in range(n):
#         sparse_vec = np.zeros(n)
#         sparse_vec[i] = 1
#         sparse_vec[j] = 1
#         sparse_vecs.add(tuple(sparse_vec))    

sparse_vecs = np.asarray(list(sparse_vecs))

indices = []

for sparse_vec in sparse_vecs:
    indices.append(list(sparse_vec).index(1))

print(indices)

def gauss_pdf(x):
    return np.exp(-0.5*((x/sigma)**2))

data = torch.load('/Users/conor/Documents/PHD_RESEARCH/ACTIVE_SEARCH_AS_RL/rlkit/data/tabular-active-search-k1-low-combo-if-0-01-coeff/tabular_active_search_k1_low_combo_if_0_01_coeff_2020_11_17_09_48_51_0000--s-0/params.pkl')

expl_policy = data['exploration/policy']
eval_policy = data['evaluation/policy']

NUM_OBSERVATIONS = 100
NUM_TRIALS = 1000

entropy_results = np.zeros((NUM_TRIALS, 4, NUM_OBSERVATIONS))
recovery_results = np.zeros((NUM_TRIALS, 4, NUM_OBSERVATIONS))

for trial in range(NUM_TRIALS):
    print(trial)
    beta = np.asarray(sparse_vecs[np.random.choice(len(sparse_vecs))])
    for algo_type in [0,1,2,3]:
        # print(algo_type)
        prior = np.ones(len(sparse_vecs))/len(sparse_vecs)

        for i in beta:
            if i == 0:
                print('.', end="")
            else:
                print('*', end="")

        print('\n')

        entropies = []
        recoveries = []        
        for i in range(NUM_OBSERVATIONS):
            if algo_type == 0:
                interval_start = np.random.randint(n)
                interval_length = np.random.randint(n-interval_start)
            elif algo_type == 1:
                interval_start = np.random.randint(n)
                interval_length = 1
            elif algo_type == 2:
                with torch.no_grad():
                	interval_start = int((expl_policy.get_action(torch.from_numpy(prior).float())[0][1] + 1)*15)
                	interval_length = int((expl_policy.get_action(torch.from_numpy(prior).float())[0][0] + 1)*15)
                	interval_length = min( max(int(abs(interval_length)),1), 30-interval_start)
            elif algo_type == 3:
                with torch.no_grad():                    
                    interval_start = int((eval_policy.get_action(torch.from_numpy(prior).float())[0][1] + 1)*15)
                    interval_length = int((eval_policy.get_action(torch.from_numpy(prior).float())[0][0] + 1)*15)
                    interval_length = min( max(int(abs(interval_length)),1), 30-interval_start)
            # plt.plot( [np.log(prior[indices.index(index)]) for index in range(30)] )
            # plt.ylim([-1000,0])
            # plt.show()            
            x = np.zeros(n)
            interval = np.ones(interval_length) / interval_length
            x[interval_start:interval_start+interval_length] = interval
            y = beta @ x + np.random.normal()*sigma
            posterior = np.zeros(len(sparse_vecs))
            for j in range(len(posterior)):
                posterior[j] =  gauss_pdf( y - sparse_vecs[j].T @ x )*prior[j]
            posterior = posterior/np.sum(posterior)
            prior = posterior
            entropy = 0
            for index, sparse_vec in enumerate(sparse_vecs):
                if prior[index] > 0:
                    entropy -= prior[index] * np.log(prior[index])            
            entropies.append(entropy)
            recovery = int(np.all(sparse_vecs[np.argmax(prior)] == beta))
            recoveries.append(recovery)
            print('\n')
            for i in x:
                if i == 0:
                    print('.', end="")
                else:
                    print(1, end="")
        print(int(np.sum(recoveries)), '\r')            
            
        entropy_results[trial,algo_type,:] = entropies
        recovery_results[trial,algo_type,:] = recoveries

pickle.dump(entropy_results, open('entropy_results_low_k1_combo_if_0_01.p', "wb" ))
pickle.dump(recovery_results, open('recovery_results_low_k1_combo_if_0_01.p', "wb" ))
