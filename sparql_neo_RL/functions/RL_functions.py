import numpy as np, pandas as pd, json, itertools, ast, sys, csv, random
from functions.tree_format import IterateBuildTree, InnerJoinsIntraBGPS, IterateBuildTreeBetweenBGPS, TreeFormat
symbol = "ᶲ"

def GetTriplesSubtree(subtree_as_str):
    code_tpf = ['VAR_VAR_VAR', 'VAR_VAR_URI', 'VAR_URI_VAR', 'VAR_URI_URI', 'VAR_URI_LITERAL', 'VAR_VAR_LITERAL',
                'URI_URI_LITERAL', 'URI_URI_VAR', 'URI_URI_URI', 'URI_VAR_VAR', 'URI_VAR_URI', 'URI_VAR_LITERAL',
                'LITERAL_URI_VAR', 'LITERAL_URI_URI', 'LITERAL_URI_LITERAL']
    total_triples = 0
    for i in code_tpf:
        total_triples += subtree_as_str.count(i)
    #if subtree_as_str in code_tpf:
    #    total_triples += 1
    return total_triples

def GetTreeSize(subtrees, treesize):
    if len(subtrees) == 1:
        return treesize
    else:
        treesize += 1
        left_treesize = GetTreeSize(subtrees[1], treesize)
        right_treesize = GetTreeSize(subtrees[2], treesize)
        treesize = max(left_treesize,right_treesize)
    return treesize

def GetAllJoins(subtree_as_str):
    join = subtree_as_str.count('JOIN')
    left_join = subtree_as_str.count('LEFT_JOIN')
    return join-left_join, left_join

def GetIter(subtree_as_str):
    if "iter" in subtree_as_str:
        return 1
    else:
        return 0
    
def GetTotalBgp(state):
    bgp_list = []
    for s in state:
        bgp_list.append(s[1])
    bgp_list = set(bgp_list)
    return len(bgp_list)

def GetDataframe(subtree, state, columns):
    subtree_str = str(subtree)
    subtree_list = ast.literal_eval(subtree[0].tolist())
    total_bgps = GetTotalBgp(state)
    treesize = GetTreeSize(subtree_list, 1)
    triples = GetTriplesSubtree(subtree_str)
    join, left_join = GetAllJoins(subtree_str)
    iters = GetIter(subtree_str)
    
    values = [total_bgps, triples, treesize, join, left_join]
    x_rl_query = pd.DataFrame([values], columns = columns)
    return x_rl_query
    #print(x_rl_query)
    #scalerx = StandardScaler()
    #values_scaled = scalerx.fit_transform(x_rl_query)
    #x_rl_query = pd.DataFrame(values_scaled, columns = columns)
    #print(x_rl_query)
    
def RL_Actions(bgp):
    actions = []
    idx = 0
    for k,v in bgp.items():
        bgp_list = v['bgp_list']
        for b in v['bgp_list']:
            actions.append((idx,k,b['P'],b['triple_type']))
            idx += 1
    return actions


def RL_Initial_Step(actions):
    initial_state = []
    random_index = random.randint(0, len(actions)-1)
    random_action = actions[random_index]
    initial_state.append(random_action)
    return initial_state

def RL_available_actions(actions, current_state):
    if not current_state:
        return actions
    available_actions = sorted(list(set(actions) - set(current_state)))
    same_bgp_actions = []
    other_bgp_actions = []
    for i in available_actions:
        if current_state[-1][1] == i[1]:
            same_bgp_actions.append(i)
        else:
            other_bgp_actions.append(i)
    if same_bgp_actions:
        available_actions = same_bgp_actions
    else:
        available_actions = other_bgp_actions
    return available_actions


def RL_Argmax(array):
    argmax_list = np.argwhere(array == np.amax(array))
    argmax_list_flatten = [item for sublist in argmax_list for item in sublist]
    return int(random.choice(argmax_list_flatten))

def RL_Argmax_available(q_value,available_actions):
    available_idx = [i[0] for i in available_actions]
    #print("available_actions",available_actions)
    #print("available_idx",available_idx)
    q_value_available = [q_value[i] for i in available_idx]
    #print("q_values",q_value, type(q_value))
    #print("q_values_available",q_value_available, type(q_value_available))
    
    argmax_list = np.argwhere(q_value_available == np.amax(q_value_available))
    argmax_list_flatten = [item for sublist in argmax_list for item in sublist]
    #print("argmax_list",argmax_list, type(argmax_list))
    #print("argmax_list_flatten",argmax_list_flatten, type(argmax_list_flatten))
    value = int(random.choice(argmax_list_flatten))
    #print("value",value)
    
    #print("..........................")
    return value


def RL_Next_step(actions,
                current_state,
                reward,
                q_value,
                x_rl_tree_ind,
                x_rl_query_ind,
                y_rl_ind,
                bgp_ind,
                epsilon
                ):
    random_num = np.random.random()
    available_actions = RL_available_actions(actions, current_state)
    actual_state = len(current_state) - 1
    
    if random_num > epsilon:
        chosen_action_available_idx = random.randint(0, len(available_actions)-1)
        chosen_action = available_actions[chosen_action_available_idx]
        chosen_action_idx = chosen_action[0]
        #print(chosen_action_idx)
        current_state.append(chosen_action)
        reward, terminal = RL_Reward(actions,
                                     available_actions,
                                     current_state,
                                     chosen_action,
                                     reward,
                                     x_rl_tree_ind,
                                     x_rl_query_ind,
                                     y_rl_ind,
                                     bgp_ind)
    else:
        #print(current_state)
        chosen_action_available_idx = RL_Argmax_available(q_value[actual_state],available_actions)
        chosen_action = available_actions[chosen_action_available_idx]
        chosen_action_idx = chosen_action[0]
        current_state.append(chosen_action)
        reward, terminal = RL_Reward(actions,
                                     available_actions,
                                     current_state,
                                     chosen_action,
                                     reward,
                                     x_rl_tree_ind,
                                     x_rl_query_ind,
                                     y_rl_ind,
                                     bgp_ind)
    return current_state, reward, terminal, chosen_action, chosen_action_idx

def RL_Reward(actions,
              available_actions,
              current_state,
              chosen_action,
              reward,
              x_rl_tree_ind,
              x_rl_query_ind,
              y_rl_ind,
              bgp_ind):
    terminal = False
    #print(available_actions)
    #if chosen_action not in available_actions:
    #    reward -= 100000
    if len(current_state) == len(actions):
        terminal = True
    new_dicto = RL_Rebuild_Dictionary(bgp_ind, current_state)
    tree_format = TreeFormat(new_dicto,symbol)
    #tree_format = ast.literal_eval(tree_format)
    new_tree = np.array([str(tree_format).replace('"', ';').replace("'", '"')])
    x_rl_query = GetDataframe(new_tree, current_state, x_rl_query_ind.columns)
    pred = getpredictions_info_nojc(new_tree, x_rl_query, y_rl_ind)['pred']
    reward -= pred[0][0]
    return reward, terminal

def RL_Rebuild_Dictionary(bgp, final_state):
    new_dicto = {}
    bgp_names = list(set([i[1] for i in final_state]))
    ### Keys
    for i in bgp_names:
        new_dicto[i] = {"bgp_list" : [], "opt" : bgp[i]['opt']}
    for i in final_state:
        new_dicto[i[1]]["bgp_list"].append({'P' : i[2], 'triple_type' : i[3]})

    return new_dicto


def RL_First_Policy(actions):
    actions_length = len(actions)
    return np.zeros((actions_length,actions_length))

def RL_bgp_format(ds_rl):
    bgps_rl = ds_rl['bgps'].values
    bgps_rl = [ast.literal_eval(bgp) for bgp in bgps_rl]
    return bgps_rl

def RL_get_data(x_rl_tree, x_rl_query, y_rl, bgps_rl,idx):
    x_rl_tree_c = x_rl_tree.copy()
    x_rl_query_c = x_rl_query.copy()
    y_rl_c = y_rl.copy()
    bgps_rl_c = bgps_rl.copy()
    x_rl_query_c = x_rl_query_c.reset_index(drop=True)
    columns = x_rl_query_c.columns
    values = x_rl_query_c.values[idx]    
    x_rl_tree_test = np.array([x_rl_tree_c[idx]])
    x_rl_query_test = pd.DataFrame([values],columns=columns)
    y_rl_test = np.array([y_rl_c[idx]])
    bgps_test = bgps_rl_c[idx]
    return x_rl_tree_test, x_rl_query_test, y_rl_test, bgps_test


def RL_get_final_state_bgp_tree(q_value,actions,bgp,symbol):
    q_value = q_value.tolist()
    best_state = []
    arg_max = RL_Argmax(q_value[0])
    best_state.append(actions[arg_max])
        
    for q_val in q_value[1:]:
        available_actions = RL_available_actions(actions, best_state)
        chosen_action_available_idx = RL_Argmax_available(q_val,available_actions)
        chosen_action = available_actions[chosen_action_available_idx]
        chosen_action_idx = chosen_action[0]
        best_state.append(chosen_action) 
    new_dicto = RL_Rebuild_Dictionary(bgp, best_state)
    new_tree = np.array([str(TreeFormat(new_dicto,symbol)).replace('"', ';').replace("'", '"')])
    
    return best_state, new_dicto, new_tree


def RL_results_functions(idx,pred_old, pred_new,prl_tol1,prl_tol2):
    difference = pred_new - pred_old
    abs_diff = np.abs(difference)
    best_of_the_pred = []
    similar_of_the_pred = []
    worst_of_the_pred = []
    if abs_diff < prl_tol1:
        best_of_the_pred.append(idx)
    elif abs_diff < prl_tol2:
        similar_of_the_pred.append(idx)
    else:
        worst_of_the_pred.append(idx)
    return best_of_the_pred, similar_of_the_pred, worst_of_the_pred
    
def RLNeo_Next_step(actions,
                current_state,
                reward,
                q_value,
                x_rl_tree_ind,
                x_rl_query_ind,
                y_rl_ind,
                bgp_ind,
                epsilon
                ):
    random_num = np.random.random()
    available_actions = RL_available_actions(actions, current_state)
    actual_state = len(current_state) - 1
    
    if random_num > epsilon:
        chosen_action_available_idx = random.randint(0, len(available_actions)-1)
        chosen_action = available_actions[chosen_action_available_idx]
        chosen_action_idx = chosen_action[0]
        #print(chosen_action_idx)
        current_state.append(chosen_action)
        reward, terminal = RLNeo_Reward(actions,
                                     available_actions,
                                     current_state,
                                     chosen_action,
                                     reward,
                                     x_rl_tree_ind,
                                     x_rl_query_ind,
                                     y_rl_ind,
                                     bgp_ind)
    else:
        #print(current_state)
        chosen_action_available_idx = RL_Argmax_available(q_value[actual_state],available_actions)
        chosen_action = available_actions[chosen_action_available_idx]
        chosen_action_idx = chosen_action[0]
        current_state.append(chosen_action)
        reward, terminal = RLNeo_Reward(actions,
                                     available_actions,
                                     current_state,
                                     chosen_action,
                                     reward,
                                     x_rl_tree_ind,
                                     x_rl_query_ind,
                                     y_rl_ind,
                                     bgp_ind)
    return current_state, reward, terminal, chosen_action, chosen_action_idx

def RLNeo_Reward(actions,
              available_actions,
              current_state,
              chosen_action,
              reward,
              x_rl_tree_ind,
              x_rl_query_ind,
              y_rl_ind,
              bgp_ind):
    terminal = False
    reward = 0
    #print(available_actions)
    #if chosen_action not in available_actions:
    #    reward -= 100000
    if len(current_state) == len(actions):
        terminal = True
        new_dicto = RL_Rebuild_Dictionary(bgp_ind, current_state)
        tree_format = TreeFormat(new_dicto,symbol)
        #tree_format = ast.literal_eval(tree_format)
        new_tree = np.array([str(tree_format).replace('"', ';').replace("'", '"')])
        x_rl_query = GetDataframe(new_tree, current_state, x_rl_query_ind.columns)
        pred = getpredictions_info_nojc(new_tree, x_rl_query, y_rl_ind)['pred']
        reward -= pred[0][0]
    return reward, terminal
    
    
def RLNeo_q_values(q_value_neo,current_state, reward):
    for q_val, act in zip(q_value_neo,current_state):
        q_val[act[0]] = max(q_val[act[0]],reward)
    return q_value_neo