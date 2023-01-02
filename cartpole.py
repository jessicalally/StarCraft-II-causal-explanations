import gym
import tensorflow.compat.v1 as tf
import networkx as nx
import numpy as np
import pandas as pd
import structeral_causal_modeling as scm
import explanation_templates as explanations

env = gym.make('CartPole-v1')

causal_graph = np.array([
    [0, 0, 0, 0], # 0 = cart position
    [1, 0, 0, 1], # 1 = cart velocity
    [0, 0, 0, 0], # 2 = pole angle
    [0, 0, 1, 0], # 3 = pole angular velocity
])

action_set = (0, 1) # 0 = push cart to left, 1 = push cart to right

equation_predictions = {}

causal_graph = nx.from_numpy_matrix(causal_graph, create_using=nx.MultiDiGraph())

for edge in causal_graph.edges():
    causal_graph.remove_edge(edge[0], edge[1])
    causal_graph.add_edge(edge[0], edge[1], action=0) # Push cart to left
    causal_graph.add_edge(edge[0], edge[1], action=1) # Push cart to right

num_episodes = 300
time_frame = 500 # Max number of steps per episodes

state_space = 4
action_space = 2

# Simplify by converting to a discrete state space
bin_size = 30
bins = [
    np.linspace(-4.8, 4.8, bin_size),
    np.linspace(-4, 4, bin_size),
    np.linspace(-0.418, 0.418, bin_size),
    np.linspace(-4, 4, bin_size)
]

q_table = np.random.uniform(low=-1,high=1,size=([bin_size] * state_space + [action_space]))
data_set = []

def Discrete(state, bins):
    index = []

    for i in range(len(state)):
        index.append(np.digitize(state[i],bins[i]) - 1)
    return tuple(index)

def Q_learning(q_table, bins, episodes = 2000, gamma = 0.95, lr = 0.1, timestep = 100, epsilon = 0.2):
    print('Performing Q-learning...')
    rewards = 0
    steps = 0

    for episode in range(episodes):
        steps += 1 
        # env.reset() => initial observation
        current_observation, info = env.reset()
        current_state = Discrete(current_observation, bins)
      
        score = 0
        done = False
        while not done: 
            if episode % timestep == 0: 
                env.render()
            
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[current_state])

            next_observation, reward, done, terminated, info = env.step(action)
            next_state = Discrete(next_observation,bins)
            score += reward
            
            if not done:
                max_future_q = np.max(q_table[next_state])
                current_q = q_table[current_state+(action,)]
                new_q = (1-lr)*current_q + lr*(reward + gamma*max_future_q)
                q_table[current_state+(action,)] = new_q

            data_set.append((current_observation, action, reward, next_observation))
            current_state = next_state
            
        # End of the loop update
        else:
            rewards += score
            if score > 195 and steps >= 100: 
                print('Solved')

        if episode % timestep == 0: print(reward / timestep)
    
    print('Finished Q-learning...')

def train_scm():
    state_set = []
    action_set = []
    next_state_set = []

    for (s, a, r, s_next) in data_set: 
        state_set.append(s)
        action_set.append(a)
        next_state_set.append(s_next)
    
    process_explanations(state_set, action_set, next_state_set)

def predict_from_scm(structural_equations):
    predict_y = {}

    for key in structural_equations:
        training_pred = structural_equations[key]['function'].predict(input_fn=get_input_fn(structural_equations[key],                          
                num_epochs=1,                          
                n_batch = 128,                          
                shuffle=False))
        predict_y[key] = np.array([item['predictions'][0] for item in training_pred])
        print(predict_y[key])

    return predict_y


def action_prediction(structural_equations, current_state):
    predicted_difference = {}
    predict_y = predict_from_scm(structural_equations)

    for key in structural_equations:
        predicted_difference[key] = abs(Sy - predict_y[key])


def process_explanations(state_set, action_set, next_state_set):
    print("Starting SCM training...")
    action_influence_dataset = {}

    for idx, action in enumerate(action_set):
        if action in action_influence_dataset:
            action_influence_dataset[action]['state'].append(state_set[idx])
            action_influence_dataset[action]['next_state'].append(next_state_set[idx])
        else:
            action_influence_dataset[action] = {'state' : [], 'next_state': []}
            action_influence_dataset[action]['state'].append(state_set[idx])
            action_influence_dataset[action]['next_state'].append(next_state_set[idx])

    structural_equations = initialize_structural_equations(causal_graph, action_influence_dataset)
    structural_equations = scm.train_structeral_equations(structural_equations)
    print('Ending SCM training...')
    predict_from_scm(structural_equations)

    # print('Processing explanations...')

    # for agent_step in range(1, len(state_set) + 1, 1000):
    #     print(str(agent_step) + "/" + str(len(state_set)))
    #     why_explanations = {}
    #     why_not_explanations = {}

    #     why_explanations[(agent_step, action)] = {'state': state_set[idx], 'why_exps': generate_why_explanations(state_set[agent_step], action, agent_step, causal_graph, structural_equations)}

    #     # poss_counter_actions = set(action_set).difference({action})
    #     # for counter_action in poss_counter_actions:
    #     #     why_not_explanations[(agent_step, action, counter_action)] = {'state': state_set[agent_step], 
    #     #                                             'why_not_exps': generate_counterfactual_explanations(state_set[agent_step], action, counter_action, agent_step, causal_graph, structural_equations)}

    #     pd.DataFrame.from_dict(data=why_explanations, orient='index').to_csv('why_explanations_cartpole.csv', mode='a', header=False)
    #     # pd.DataFrame.from_dict(data=why_not_explanations, orient='index').to_csv('why_not_explanations_cartpole.csv', mode='a', header=False)


def initialize_structural_equations(causal_graph, action_influence_dataset):
    structural_equations = {}
    unique_functions = {}

    for edge in causal_graph.edges():
        for preds in causal_graph.predecessors(edge[1]):
            node = edge[1]
            if (node, 0) not in unique_functions:
                unique_functions[(node, 0)] = set()
            
            if (node, 1) not in unique_functions:
                unique_functions[(node, 1)] = set()

            # Between every node and its predecessor we have L and R actions
            unique_functions[(node, 0)].add(preds)
            unique_functions[(node, 1)].add(preds)

    for key in unique_functions:
        if key[1] in action_influence_dataset:
            x_data = []
            for x_feature in unique_functions[key]:
                x_data.append(np.array(action_influence_dataset[key[1]]['state'])[:,x_feature])

            x_feature_cols = [tf.feature_column.numeric_column(str(i)) for i in range(len(x_data))]  
            y_data = np.array(action_influence_dataset[key[1]]['next_state'])[:,key[0]]
            lr = tf.estimator.LinearRegressor(feature_columns=x_feature_cols, model_dir='scm_models/linear_regressor/'+str(key[0])+'_'+str(key[1]))
            structural_equations[key] = {'X': x_data,'Y': y_data,'function': lr,}
    
    return structural_equations

def generate_why_explanations(actual_state, actual_action, state_num_in_batch, causal_graph, structural_equations):
    optimal_state_set = []
    actual_state = {k: actual_state[k] for k in range(len(actual_state))}
    sink_nodes = get_sink_nodes(causal_graph)
    actual_action_edge_list = get_edges_of_actions(actual_action, causal_graph)
    all_actual_causal_chains_from_action = get_causal_chains(sink_nodes, actual_action_edge_list, causal_graph)
    # print(all_actual_causal_chains_from_action)
    action_chain_list = get_action_chains(actual_action, all_actual_causal_chains_from_action, causal_graph)
    # print(action_chain_list)

    why_exps = set()
    for i in range(len(all_actual_causal_chains_from_action)):
        optimal_state = dict(actual_state)
        print(optimal_state)
        for j in range(len(all_actual_causal_chains_from_action[i])):
            for k in range(len(all_actual_causal_chains_from_action[i][j])):
                optimal_state[all_actual_causal_chains_from_action[i][j][k]] = predict_node_scm(
                    all_actual_causal_chains_from_action[i][j][k], action_chain_list[i][j][k], structural_equations)[state_num_in_batch]

        optimal_state_set.append(optimal_state)
        min_tuple_actual_state = get_minimally_complete_tuples(all_actual_causal_chains_from_action[i], actual_state)
        min_tuple_optimal_state = get_minimally_complete_tuples(all_actual_causal_chains_from_action[i], optimal_state)
        explanation = explanations.cartpole_generate_why_text_explanations(min_tuple_actual_state, min_tuple_optimal_state, actual_state, actual_action)
        print(explanation)
        print("\n")
        why_exps.add(explanation)

    return why_exps


def generate_counterfactual_explanations(actual_state, actual_action, counterfactual_action, state_num_in_batch, causal_graph, structural_equations):
    counterfactual_state_set = []
    actual_state = {k: actual_state[k] for k in range(len(actual_state))}
    sink_nodes = get_sink_nodes(causal_graph)
    counter_action_edge_list = get_edges_of_actions(counterfactual_action, causal_graph)
    actual_action_edge_list = get_edges_of_actions(actual_action, causal_graph)
    
    all_counter_causal_chains_from_action = get_causal_chains(sink_nodes, counter_action_edge_list, causal_graph)
    all_actual_causal_chains_from_action = get_causal_chains(sink_nodes, actual_action_edge_list, causal_graph)
    action_chain_list = get_action_chains(counterfactual_action, all_counter_causal_chains_from_action, causal_graph)
    
    for i in range(len(all_counter_causal_chains_from_action)):
        counterfactual_state = dict(actual_state)
        for j in range(len(all_counter_causal_chains_from_action[i])):
            for k in range(len(all_counter_causal_chains_from_action[i][j])):
                counterfactual_state[all_counter_causal_chains_from_action[i][j][k]] = predict_node_scm(
                    all_counter_causal_chains_from_action[i][j][k], action_chain_list[i][j][k], structural_equations)[state_num_in_batch]
        counterfactual_state_set.append(counterfactual_state)    
    
    contrastive_exp = set()
    for actual_chains in all_actual_causal_chains_from_action:
        for counter_chains in all_counter_causal_chains_from_action:
            for counter_states in counterfactual_state_set:
                contrast_tuple = get_minimal_contrastive_tuples(actual_chains, counter_chains, actual_state, counter_states)
                contrastive_exp.add(explanations.cartpole_generate_contrastive_text_explanations(contrast_tuple, actual_action))    

    for exp in contrastive_exp:
        print(exp)
        print("\n")
    # unique contrastive explanations
    return contrastive_exp    


def predict_node_scm (node, action, structural_equations):
    key = (node, action)
    # print(structural_equations[key]['function'])
    pred = structural_equations[key]['function'].predict(input_fn=get_input_fn(structural_equations[key],                          
                num_epochs=1,                          
                n_batch = 128,                          
                shuffle=False))
    # print(pred)
    result = np.array([item['predictions'][0] for item in pred])
    # print(key)
    # print(result)

    return result
    
    
"""minimally complete tuple = (head node of action, immediate pred of sink nodes, sink nodes)"""
def get_minimally_complete_tuples(chains, state):
    head = set()
    immediate = set()
    reward = set()
    for chain in chains:
        if len(chain) == 1:
            reward.add((chain[0], state[chain[0]]))
        if len(chain) == 2:
            head.add((chain[0], state[chain[0]]))
            reward.add((chain[-1], state[chain[-1]]))
        if len(chain) > 2:    
            head.add((chain[0], state[chain[0]]))
            immediate.add((chain[-2], state[chain[-2]]))
            reward.add((chain[-1], state[chain[-1]]))
    minimally_complete_tuple = {
        'head': head,
        'immediate': immediate,
        'reward': reward
    }
    return minimally_complete_tuple    

def get_minimal_contrastive_tuples(actual_chain, counterfactual_chain, actual_state, counterfactual_state):

    actual_minimally_complete_tuple = get_minimally_complete_tuples(actual_chain, actual_state)
    counterfactual_minimally_complete_tuple = get_minimally_complete_tuples(counterfactual_chain, counterfactual_state)
    min_tuples = np.sum(np.array([list(k) for k in list(actual_minimally_complete_tuple.values())]))
    tuple_states = set([k[0] for k in min_tuples])

    counter_min_tuples = np.sum(np.array([list(k) for k in list(counterfactual_minimally_complete_tuple.values())]))
    counter_tuple_states = set([k[0] for k in counter_min_tuples])
    counter_tuple_states.difference_update(tuple_states)

    contrastive_tuple = {
                        'actual': {k: actual_state[k] for k in counter_tuple_states},
                        'counterfactual': {k: counterfactual_state[k] for k in counter_tuple_states},
                        'reward': {k[0]: k[1] for k in actual_minimally_complete_tuple['reward']}
                        }
    return contrastive_tuple


def get_causal_chains(sink_nodes, action_edge_list, causal_graph):
    # Action edge list contains all the edges corresponding to the action

    counter_action_head_set = set(np.array(action_edge_list)[:,1]) 
    all_causal_chains_from_action = []

    for action_head in counter_action_head_set:
        chains_to_sink_nodes = []
        for snode in sink_nodes:
            if action_head == snode:
                chains_to_sink_nodes.append([snode])
            else:
                chains_to_sink_nodes.extend((nx.all_simple_paths(causal_graph, source=action_head, target=snode)))
        all_causal_chains_from_action.append(chains_to_sink_nodes)

    return all_causal_chains_from_action    


def get_action_chains(action, chain_lists_of_action, causal_graph):
    action_chain_list = []
    for chain_list in chain_lists_of_action:
        action_chains = []
        for chain in chain_list:
            action_chain = []
            for i in range(len(chain)):
                if i == 0:
                    action_chain.append(action)  
                else:
                    action_chain.append(causal_graph.get_edge_data(chain[i-1], chain[i])[0]['action'])
            action_chains.append(action_chain)
        action_chain_list.append(action_chains)          
    return action_chain_list        


def get_edges_of_actions(action, causal_graph):
    return list(edge for edge in causal_graph.edges(data=True) if edge[2]['action'] == action)
   
def get_sink_nodes(causal_graph):
    return list((node for node, out_degree in causal_graph.out_degree_iter() if out_degree == 0 and causal_graph.in_degree(node) > 0 ))

def get_input_fn(data_set, num_epochs=None, n_batch = 128, shuffle=False):
    # print(data_set)
    x_data = {str(k): data_set['X'][k] for k in range(len(data_set['X']))}
    # print(x_data)
    return tf.estimator.inputs.pandas_input_fn(       
            x=pd.DataFrame(x_data),
            y = pd.Series(data_set['Y']),       
            batch_size=n_batch,          
            num_epochs=num_epochs,       
            shuffle=shuffle)

Q_learning(q_table, bins)
train_scm()

# for episode in range(num_episodes):
#     env.reset()

#     for t in range(time_frame):
#         # Get random action (for now)
#         action = env.action_space.sample()

#         next_state, reward, done, terminated, info = env.step(action)
#         print(t, next_state, reward, done, info, action)
        
#         if done:
#             break