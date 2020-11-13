import torch
import numpy as np
import pickle

n = 30
sigma=.00001

sparse_vecs = set()

for i in range(n):
    sparse_vec = np.zeros(n)
    sparse_vec[i] = 1
    sparse_vecs.add(tuple(sparse_vec))

sparse_vecs = np.asarray(list(sparse_vecs))

def gauss_pdf(x):
    return np.exp(-0.5*((x/sigma)**2))

data = torch.load('/Users/conor/Documents/PHD_RESEARCH/ACTIVE_SEARCH_AS_RL/rlkit/data/tabular-active-search-k1-det/tabular_active_search_k1_det_2020_11_12_00_31_19_0000--s-0/params.pkl')
expl_policy = data['exploration/policy']
eval_policy = data['evaluation/policy']

NUM_TRIALS = 1
NUM_OBSERVATIONS = 50
entropy_results = np.zeros((NUM_TRIALS, 4, NUM_OBSERVATIONS))
recovery_results = np.zeros((NUM_TRIALS, 4, NUM_OBSERVATIONS))

beta = np.asarray(sparse_vecs[np.random.choice(len(sparse_vecs))])
for beta in sparse_vecs:
    if beta[9] == 1:
        for algo_type in [0,2,3]:
            print(algo_type)
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
                	interval_start = int((expl_policy.get_action(torch.from_numpy(prior).float())[0][1] + 1)*15)
                	interval_length = int((expl_policy.get_action(torch.from_numpy(prior).float())[0][0] + 1)*15)
                	interval_length = min( max(int(abs(interval_length)),1), 30-interval_start)
                elif algo_type == 3:
                    interval_start = int((eval_policy.get_action(torch.from_numpy(prior).float())[0][1] + 1)*15)
                    interval_length = int((eval_policy.get_action(torch.from_numpy(prior).float())[0][0] + 1)*15)
                    interval_length = min( max(int(abs(interval_length)),1), 30-interval_start)

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

                for i in x:
                    if i == 0:
                        print('.', end="")
                    else:
                        print(1, end="")
                print(recovery, '\r')            
                
            # entropy_results[trial,algo_type,:] = entropies
            # recovery_results[trial,algo_type,:] = recoveries

    # pickle.dump(entropy_results, open('entropy_results_det_short.p', "wb" ))
    # pickle.dump(recovery_results, open('recovery_results_det_short.p', "wb" ))
