import pandas as pd
import numpy as np
import json

################## BUILD A RANDOM TREE

##### FIND DECISION THRESHOLD AND GROW BRANCH
def fork_tree(data, subspace_list, target_var, target_value, slices, minimum_slice_points, tree_depth, branch_depth):    
    #create empty dataframe to store slice results
    slice_columns = ["subspace","slice_value","below_class","below_count","below_misclass_count","below_target_prob","above_class","above_count","above_misclass_count","above_target_prob","total_misclass_count"]
    slice_record = pd.DataFrame(columns=slice_columns)
    
    ##### GREEDY METHOD
    for subspace in subspace_list:
    
        for slice_counter in xrange(slices):
            
            spread = data[subspace].max() - data[subspace].min()
            slice_value = spread/slices * slice_counter + data[subspace].min()
            
            points_below = data[data[subspace] < slice_value]
            points_above = data[data[subspace] >= slice_value]
        
            # determine side with highest target saturation
            below_target_count = points_below[points_below[target_var] == target_value].shape[0]
            below_nontarget_count = points_below[points_below[target_var] != target_value].shape[0]
            above_target_count = points_above[points_above[target_var] == target_value].shape[0]
            above_nontarget_count = points_above[points_above[target_var] != target_value].shape[0]    
                
            below_total_count = below_target_count + below_nontarget_count
            above_total_count = above_target_count + above_nontarget_count
            
            if below_total_count <= minimum_slice_points or above_total_count <= minimum_slice_points:
                continue    # not enough points, skip to the next iteration
            
            below_target_prob = below_target_count/below_total_count
            above_target_prob = above_target_count/above_total_count
        
            if below_target_prob > above_target_prob:
                below_class_assignment = "target"
                above_class_assignment = "nontarget"
                
                below_misclassification_count = below_nontarget_count
                above_misclassification_count = above_target_count
                
            elif below_target_prob < above_target_prob:
                below_class_assignment = "nontarget"
                above_class_assignment = "target"
                
                below_misclassification_count = below_target_count
                above_misclassification_count = above_nontarget_count
                
            else:
                below_class_assignment = "tie"
                above_class_assignment = "tie"
                
                below_misclassification_count = below_target_count
                above_misclassification_count = above_target_count
        
            # squared error, which in this case is same as misclassification count (because 1**2 == 1)
            total_misclassification_count = below_misclassification_count + above_misclassification_count
        
            # store results in slice_record dataframe
            new_slice = pd.DataFrame(
                [[subspace,slice_value,
                     below_class_assignment, below_total_count, below_misclassification_count, below_target_prob,
                     above_class_assignment, above_total_count, above_misclassification_count, above_target_prob,
                     total_misclassification_count]],   #coerce into creating a new row of data
                columns=slice_columns)
            slice_record = pd.concat([slice_record, new_slice], ignore_index=True)
    
    # check if slice_record is blank (not enough data points to branch)
    if slice_record.shape[0] == 0:
        target_count = data[data[target_var] == target_value].shape[0]
        nontarget_count = data[data[target_var] != target_value].shape[0]          
        total_count = target_count + nontarget_count
        target_prob = target_count/total_count
        return target_prob   # return overall probability for this division
    
    ##### MINIMIZE MISCLASSIFICATION ERROR
    # find row(s) with the minimum misclassification
    lowest_misclassification_count = slice_record.total_misclass_count.min()
    slice_record_minimized = slice_record[slice_record.total_misclass_count == lowest_misclassification_count]
    
    # in case multiple rows exist, pick one at random
    slice_record_minimized_rows = slice_record_minimized.shape[0]
    slice_record_minimized_randrow = np.random.randint(0, slice_record_minimized_rows)
    slice_record_minimized_randrow_index = slice_record_minimized.index[slice_record_minimized_randrow]
    slice_record_minimized = slice_record_minimized[slice_record_minimized.index == slice_record_minimized_randrow_index]
    
    ##### RECURSE OR END RECURSION
    # this is the decision threshold for this node in the tree
    optimal_slice_value = slice_record_minimized.slice_value[slice_record_minimized_randrow_index]
    optimal_subspace = slice_record_minimized.subspace[slice_record_minimized_randrow_index]
    
    # check to see if we've reached the maximum depth of the tree
    if branch_depth < tree_depth:
        # if no, fork the tree
        below_output = fork_tree(data[data[optimal_subspace] < optimal_slice_value], subspace_list, target_var, target_value, slices, minimum_slice_points, tree_depth, branch_depth + 1)
        above_output = fork_tree(data[data[optimal_subspace] >= optimal_slice_value], subspace_list, target_var, target_value, slices, minimum_slice_points, tree_depth, branch_depth + 1)
    else:
        # if yes, pull the probabilities
        below_output = slice_record_minimized.below_target_prob[slice_record_minimized_randrow_index]
        above_output = slice_record_minimized.above_target_prob[slice_record_minimized_randrow_index]

    ##### PUT TREE INTO NESTED DICT FORMAT
    tree = {"subspace" : optimal_subspace,
            "threshold" : optimal_slice_value,
            "below" : below_output,
            "above" : above_output}
    return tree
    


##### CREATE A BOOTSTRAP AND RANDOM SUBSPACE; BUILD A TREE

def grow_random_tree(df, feature_space, target_var, target_value, bootstrap_size, subspace_size, slices, minimum_slice_points, tree_depth):
    ##### BOOTSTRAP SAMPLE
    bootstrap_sample = df[df.index.isin(np.random.choice(df.shape[0], bootstrap_size, replace=False))]
    
    ##### RANDOM SUBSPACES
    random_subspace = []
    random_subspace_index = np.random.choice(len(feature_space), subspace_size, replace=False)
    for i in random_subspace_index:
        random_subspace.append(feature_space[i])
    
    ##### START A NEW TREE
    print "subspace:",random_subspace
    return fork_tree(bootstrap_sample, random_subspace, target_var, target_value, slices, minimum_slice_points, tree_depth, 1)



################## BUILD ALL RANDOM TREES

def grow_a_forest(df, feature_space, target_var, target_value, tree_count, bootstrap_size, subspace_size, slices=100, minimum_slice_points=20, tree_depth=3):
    for tree_counter in xrange(tree_count):
        if tree_counter == 0:
            # initiate tree_record table
            tree_record = [grow_random_tree(df, feature_space, target_var, target_value, bootstrap_size, subspace_size, slices, minimum_slice_points, tree_depth)]
        else:
            tree_record.append(grow_random_tree(df, feature_space, target_var, target_value, bootstrap_size, subspace_size, slices, minimum_slice_points, tree_depth))
        
        print "tree", tree_counter+1, "of", tree_count, "--", tree_record[tree_counter]
        
    return tree_record
        


################## SAVE FOREST AS JSON FILE

# note, set working directory to location where json will be saved/loaded

##### SAVE FOREST AS JSON
def save_forest_as_json(forest, json_filename):
    with open(json_filename, "w") as f:
        json.dump(forest, f)

##### LOAD A FOREST
def load_forest_json(json_filename):
    with open(json_filename) as f:
        forest = json.load(f)
    return forest


################## SCORE DATA WITH FOREST

##### RECURSIVELY FOLLOW TREE
def follow_tree(current_tree, df, current_datapoint):
    current_subspace = current_tree["subspace"]
    current_threshold = current_tree["threshold"]

    # which side of the decision threshold is the data
    if df[current_subspace][current_datapoint] < current_threshold:
        
        # is the node the final probability or another tree; recurse if another tree
        if type(current_tree["below"]) != dict:
            return current_tree["below"]
        else:
            subtree = current_tree["below"]
            return follow_tree(subtree, df, current_datapoint)  #recurse
    else:
        if type(current_tree["above"]) != dict:
            return current_tree["above"]
        else:
            subtree = current_tree["above"]
            return follow_tree(subtree, df, current_datapoint)  #recurse



##### LEAF SCORE AGGREGRATION
def score_aggregation(df, forest):
    df_scored = df
    df_scored["target_score"] = np.NaN
    
    total_datapoints = df.shape[0]
    tree_count = len(forest)
    datapoint_counter = 0
    
    # loop through all the points
    for datapoint_i in df.index:
        
        # sum of scores from all the trees; initiate
        target_score_sum = 0
        
        # loop through the trees
        for tree_counter in xrange(tree_count):       
            target_score_sum += follow_tree(forest[tree_counter], df, datapoint_i)
            
        df_scored.target_score[datapoint_i] = target_score_sum/tree_count
        
        if datapoint_counter % 200 == 0:
            print "% complete:", np.round(datapoint_counter/total_datapoints * 100, 1)
        
        datapoint_counter += 1
    
    print "% complete: 100.0"
    
    df["rank"] = df.target_score.rank(method="first", ascending=False)
    return df_scored



##### ESTIMATE PROBABILITIES FROM SCORE
# Summary: Estimate the probability of an observation belonging to class, based on localized target rates within the ntile in which a observation falls

# General Method: Determine an ntile for every observation, based on the score output of the random forest. Then map each observation to a probability based on its ntile.

# E.g., A score between .5 and .6 may represent a top ntile based on the forest prediction. This ntile has a 90% target rate in actual data; thus the target probability of observations within this score range is 90%
 
def create_probability_map(df, target_var, target_value, ntiles=250):

    total_points = df.shape[0]
    count_per_ntile = total_points/ntiles
    
    df_ntile = pd.DataFrame(range(ntiles), columns=["ntile"])
    
    df_ntile["rank_min"] = df_ntile["ntile"] * count_per_ntile
    df_ntile["rank_max"] = (df_ntile["ntile"] + 1) * count_per_ntile
    
    df_ntile["score_min"] = np.NaN
    df_ntile["score_max"] = np.NaN
    df_ntile["target_probability"] = np.NaN
    
    for ntile_number in range(ntiles):
        rank_min = df_ntile.rank_min[ntile_number]
        rank_max = df_ntile.rank_max[ntile_number]
        
        df_subset = df[(df["rank"] >= rank_min) & (df["rank"] < rank_max)]
        
        df_ntile.score_min[ntile_number] = df_subset["target_score"].min()
        df_ntile.score_max[ntile_number] = df_subset["target_score"].max()
        df_ntile.target_probability[ntile_number] = np.where(df_subset[target_var] == target_value, 1, 0).sum() / count_per_ntile
    
    # if target_score is higher, use that as the probability to smooth ripples; this is ok because target_score is meant to be a probability anyway (but can be underestimated if trees are not deep enough)
    probability_adjust_records = df_ntile.score_max > df_ntile.target_probability
    df_ntile.target_probability[probability_adjust_records] = df_ntile.score_max[probability_adjust_records]    
    
    return df_ntile