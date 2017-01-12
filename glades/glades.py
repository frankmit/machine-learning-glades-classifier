"""Glades is a full suite of tools related to the random forest machine learning algorithm, including model training,
model scoring, conversion of model scores to probabilities, analysis of model features, and easy saving and loading of
trained models in json format for easy portability. All core functionality should be accessed by instantiating the
GladesClassifier object.

Author: Frank Lo
Contact: franklo@alum.mit.edu
Created: December 2013
"""

from __future__ import division, print_function
import pandas as pd
import numpy as np
import random
import json
import time
import logging

logging.basicConfig(level=logging.INFO)


class GladesClassifier(object):
    """This class contains a full suite of tools related to the random forest machine learning algorithm. Overview of
    key methods below:

    `grow_forest` -- train a random forest model given a data set and turning parameters
    `return_forest` -- return trained forest in dict format
    `display_forest` -- prints trained forest
    `save_forest_as_json` -- saves forest as a json file
    `load_forest_json` -- loads a forest from json file
    `analyze_forest` -- determines feature importance within a trained model
    `score_data_set` -- score a data set through a forest
    `score_data_point` -- score a single data point through a forest
    `create_probability_map` -- create mapping between model scores and target probabilities
    `return_probability_map` -- returns a previously generated probability map
    `display_probability_map` --  prints probability map
    `save_probability_map_as_csv` -- saves probability map as a csv file
    `load_probability_map_csv` -- loads a probability map from csv file
    `apply_probability_map_to_data_set` -- calculates target probabilities for all model scores in a data set
    `apply_probability_map_to_score` -- calculates target probability given a single model score
    """
    def __init__(self):
        # initialize instance variables: random forest training parameters
        self.df = None
        self.feature_space = None
        self.target_var = None
        self.target_value = None
        self.tree_count = None
        self.bootstrap_size = None
        self.subspace_size = None
        self.scan_granularity = None
        self.min_points_per_leaf = None
        self.max_tree_depth = None

        # initialize instance variables: trained model, metadata, and probability map
        self.forest = None
        self.forest_with_metadata = None
        self.features_in_forest = None
        self.probability_map = None

        # set lambda functions to simplify calculating target and nontarget counts
        self._get_target_counts = lambda df: df[df[self.target_var] == self.target_value].shape[0]
        self._get_nontarget_counts = lambda df: df[df[self.target_var] != self.target_value].shape[0]

    def grow_forest(self, df, feature_space, target_var, **kwargs):
        """Trains a random forest based on input data set and a variety of tuning parameters. See the glades readme for
        more detailed overview of all the args that can be used to tune the training of the forest.
        """
        # store params as instance variables
        # set default param values if kwargs are not specified
        self.df = df
        self.feature_space = feature_space
        self.target_var = target_var
        self.target_value = kwargs.get('target_value', 1)
        self.tree_count = kwargs.get('tree_count', 50)
        self.bootstrap_size = kwargs.get('bootstrap_size', 500)
        self.subspace_size = kwargs.get('subspace_size', 5)
        self.scan_granularity = kwargs.get('scan_granularity', 100)
        self.min_points_per_leaf = kwargs.get('min_points_per_leaf', 20)
        self.max_tree_depth = kwargs.get('max_tree_depth', 3)

        # set index to incremental integers from 0
        self.df.reset_index(drop=True, inplace=True)

        # forest is represented list of trees, to be populated below
        self.forest = []
        for tree_counter in range(self.tree_count):
            logging.info('currently building tree {} of {}'.format(tree_counter + 1, self.tree_count))

            tree = self._grow_random_tree()
            self.forest.append(tree)

            logging.info(json.dumps(self.forest[tree_counter], indent=4, sort_keys=True))

        # post-build operations
        self.display_forest()
        self.analyze_forest()
        self._set_forest_with_metadata()

    def _grow_random_tree(self):
        """Builds a single tree using a random selection of the data and features."""
        # selected the bootstrap sample and random subspace
        bootstrap_sample = self.df.sample(n=self.bootstrap_size, replace=False)
        random_subspace = random.sample(self.feature_space, self.subspace_size)

        # build tree based on the bootstrap sample and random subspace
        # top-level branch depth is set at 1
        logging.info('current training with subspace: {}'.format(json.dumps(random_subspace, indent=4)))
        return self._fork_tree(data=bootstrap_sample, subspace=random_subspace, current_branch_depth=1)

    def _fork_tree(self, data, subspace, current_branch_depth):
        """At every node in the tree, finds optimal way to further branch out in a way that minimizes misclassification
        error. This function recursively calls itself in order to branch as many times as needed to grow a full tree.
        """
        # every fork represent a potential split of data into two partitions; track every potential fork in this list
        potential_fork_list = []

        # check every feature in subspace for optimal fork
        for feature in subspace:
            # within each feature, scan the range of values within the feature
            # the granularity of this scan is determined by spread and scan_granularity
            spread = data[feature].max() - data[feature].min()
            fork_value_increment = spread / self.scan_granularity

            # list of values to potentially fork this feature by
            potential_fork_value_list = [x * fork_value_increment + data[feature].min() for x in range(self.scan_granularity)]

            for potential_fork_value in potential_fork_value_list:
                # each potential fork splits data into two partitions: data above the fork value, and data below it
                # isolate data above and below the potential_fork_value for this feature
                data_above = data[data[feature] >= potential_fork_value]
                data_below = data[data[feature] < potential_fork_value]

                # count targets and nontargets above and below
                above_target_count = self._get_target_counts(data_above)
                above_nontarget_count = self._get_nontarget_counts(data_above)

                below_target_count = self._get_target_counts(data_below)
                below_nontarget_count = self._get_nontarget_counts(data_below)

                above_total_count = above_target_count + above_nontarget_count
                below_total_count = below_target_count + below_nontarget_count

                # if not enough points below or above, skip to the next iteration
                if above_total_count <= self.min_points_per_leaf or below_total_count <= self.min_points_per_leaf:
                    continue

                # calculate target rates for above and below
                above_target_prob = above_target_count / above_total_count
                below_target_prob = below_target_count / below_total_count

                # determine side of fork with highest target saturation
                # assign target and nontarget labels to each side and determine misclassification rates
                if above_target_prob > below_target_prob:
                    above_class_assignment = 'target'
                    below_class_assignment = 'nontarget'

                    above_misclassification_count = above_nontarget_count
                    below_misclassification_count = below_target_count

                elif above_target_prob < below_target_prob:
                    above_class_assignment = 'nontarget'
                    below_class_assignment = 'target'

                    above_misclassification_count = above_target_count
                    below_misclassification_count = below_nontarget_count

                else:
                    above_class_assignment = 'tie'
                    below_class_assignment = 'tie'

                    above_misclassification_count = above_target_count
                    below_misclassification_count = below_target_count

                # total_misclassification_count represents the error
                # this fits the concept of sum of squared errors, because the error of each misclassification is 1, and 1**2 is 1
                total_misclassification_count = above_misclassification_count + below_misclassification_count

                # store data of potential fork in a dict object and append to potential_fork_list
                potential_fork = {
                    'feature': feature,
                    'fork_value': potential_fork_value,
                    'above_class': above_class_assignment,
                    'above_count': above_total_count,
                    'above_misclassification_count': above_misclassification_count,
                    'above_target_prob': above_target_prob,
                    'below_class': below_class_assignment,
                    'below_count': below_total_count,
                    'below_misclassification_count': below_misclassification_count,
                    'below_target_prob': below_target_prob,
                    'total_misclassification_count': total_misclassification_count
                }
                potential_fork_list.append(potential_fork)

        # check if potential_fork_list is blank (no potential_fork across any feature had enough data points to branch)
        # if that is the case, no more branching; return a score for the data without branching
        if len(potential_fork_list) == 0:
            target_count = self._get_target_counts(data)
            nontarget_count = self._get_nontarget_counts(data)

            total_count = target_count + nontarget_count
            target_prob = target_count / total_count
            return target_prob

        # find the optimal fork to branch, i.e. the fork with the lowest misclassification error
        optimal_fork = min(potential_fork_list, key=lambda x: x['total_misclassification_count'])
        optimal_feature = optimal_fork['feature']
        optimal_fork_value = optimal_fork['fork_value']

        # check to see if there is room to branch (not passed the maximum depth allowed)
        if current_branch_depth < self.max_tree_depth:
            # if eligible for branching, partition data into two sets based on the optimal fork
            data_above_optimal_fork = data[data[optimal_feature] >= optimal_fork_value]
            data_below_optimal_fork = data[data[optimal_feature] < optimal_fork_value]

            # for each data partition, branch further by recursively calling self._fork_tree again
            # note that current_branch_depth is increased by 1 in this downward branch
            above_output = self._fork_tree(data_above_optimal_fork, subspace, current_branch_depth + 1)
            below_output = self._fork_tree(data_below_optimal_fork, subspace, current_branch_depth + 1)

        else:
            # if no room for branching, return target probabilities in the two partitions
            above_output = optimal_fork['above_target_prob']
            below_output = optimal_fork['below_target_prob']

        # put this branch in dict format; due to recursion this becomes a nested dict
        tree = {
            'feature': optimal_feature,
            'threshold': optimal_fork_value,
            'value_above': above_output,
            'value_below': below_output
        }
        return tree

    def _set_forest_with_metadata(self):
        """Include training metadata with forest. This necessary in order to save forest in a file for portability, and
        load it back later without losing context information.
        """
        self.forest_with_metadata = {
            'forest': self.forest,
            'training_params': {
                'feature_space': self.feature_space,
                'target_var': self.target_var,
                'target_value': self.target_value,
                'tree_count': self.tree_count,
                'bootstrap_size': self.bootstrap_size,
                'subspace_size': self.subspace_size,
                'min_points_per_leaf': self.min_points_per_leaf,
                'scan_granularity': self.scan_granularity,
                'max_tree_depth': self.max_tree_depth,
                'features_in_forest': self.features_in_forest
            }
        }

    def return_forest(self):
        """Return forest_with_metadata, which can be used for custom scripting and operations."""
        self._check_prerequisites(prereq_list=['forest_with_metadata_exists'])
        return self.forest_with_metadata

    def display_forest(self, **kwargs):
        """Clean print of forest with indents. Does not print metadata unless include_metadata=True arg is present."""
        # sleep a small amount of time so that the print does not mash with log outputs that are also written to console
        time.sleep(0.1)

        if kwargs.get('include_metadata') == True:
            self._check_prerequisites(prereq_list=['forest_with_metadata_exists'])
            print(json.dumps(self.forest_with_metadata, indent=4, sort_keys=True))
        else:
            self._check_prerequisites(prereq_list=['forest_exists'])
            print(json.dumps(self.forest, indent=4, sort_keys=True))

    def save_forest_as_json(self, json_file_location):
        """Saves forest_with_metadata in json format for portability. Can be loaded back later with load_forest_json."""
        self._check_prerequisites(prereq_list=['forest_with_metadata_exists'])
        with open(json_file_location, 'w') as f:
            json.dump(self.forest_with_metadata, f)
        logging.info('forest saved to {}'.format(json_file_location))

    def load_forest_json(self, json_file_location):
        """Loads a json file of forest_with_metadata that was previously saved using save_forest_as_json method, and
        resets all instance variables back to appropriate values.
        """
        with open(json_file_location) as f:
            self.forest_with_metadata = json.load(f)

        # reset instance variables
        self.forest = self.forest_with_metadata['forest']
        self.feature_space = self.forest_with_metadata['training_params']['feature_space']
        self.target_var = self.forest_with_metadata['training_params']['target_var']
        self.target_value = self.forest_with_metadata['training_params']['target_value']
        self.tree_count = self.forest_with_metadata['training_params']['tree_count']
        self.bootstrap_size = self.forest_with_metadata['training_params']['bootstrap_size']
        self.subspace_size = self.forest_with_metadata['training_params']['subspace_size']
        self.min_points_per_leaf = self.forest_with_metadata['training_params']['min_points_per_leaf']
        self.scan_granularity = self.forest_with_metadata['training_params']['scan_granularity']
        self.max_tree_depth = self.forest_with_metadata['training_params']['max_tree_depth']
        self.features_in_forest = self.forest_with_metadata['training_params']['features_in_forest']

        logging.info('forest loaded from {}'.format(json_file_location))
        self.display_forest()

    def analyze_forest(self, **kwargs):
        """Analyzes forest by following down each of the branches, and prints out feature importance ratings. This
        feature importance is based on frequency of feature being utilized to branch in the forest, as well as branch
        depth when utilized (more top level branches carry more weight, while lower branches carry less weight).
        Optionally add the arg return_data=True to return the feature importance table as dataframe.
        """
        # a forest must already exist in order to run analyze_forest
        self._check_prerequisites(prereq_list=['forest_exists'])

        # collect list of every branch in every tree in the forest
        branch_list = []
        for tree in self.forest:
            branch_list += self._follow_tree_analyze(current_tree=tree, current_branch_depth=1)

        # get unique list of features in branch_list
        # also store this as an instance variable to be saved in forest_with_metadata
        unique_features = set([x['feature'] for x in branch_list])
        self.features_in_forest = list(unique_features)

        # set up structure to store feature scores and initiate scores at 0
        feature_scores_raw = dict()
        feature_scores_relative = dict()
        for feature in self.features_in_forest:
            feature_scores_raw[feature] = 0
            feature_scores_relative[feature] = 0

        # iterate through branch_list to populate feature_scores with raw scores
        cumulative_raw_score = 0
        for branch in branch_list:
            feature = branch['feature']
            branch_depth = branch['branch_depth']

            # basic method to weight higher branches more than lower branches
            raw_score = 1 / branch_depth

            feature_scores_raw[feature] += raw_score
            cumulative_raw_score += raw_score

        # populate relative_score (relative scores sum to 1)
        for feature in self.features_in_forest:
            feature_scores_relative[feature] = feature_scores_raw[feature] / cumulative_raw_score

        # rank by relative score
        feature_scores_ranked = sorted(feature_scores_relative.items(), key=lambda x: x[1], reverse=True)

        # for clean display, represent it as a dataframe and print it
        feature_scores_ranked_df = pd.DataFrame(feature_scores_ranked, columns=['feature', 'importance'])
        feature_scores_ranked_df.index = [x + 1 for x in range(len(feature_scores_ranked))]
        feature_scores_ranked_df.index.name = 'rank'
        print('--------\nAnalysis\n--------\n', feature_scores_ranked_df.to_string())

        # return data if requested via return_data arg
        if kwargs.get('return_data') == True:
            return feature_scores_relative

    def _follow_tree_analyze(self, current_tree, current_branch_depth):
        """Create list of every branch in a tree by recursively calling itself to follow down every branch."""
        # ever branch is represented as a dict with feature and branch_depth
        current_branch = {
            'feature': current_tree['feature'],
            'branch_depth': current_branch_depth
        }

        if type(current_tree['value_above']) is dict:
            # if above path is another branch, recursively follow it down
            above_branch_list = self._follow_tree_analyze(current_tree['value_above'], current_branch_depth + 1)
        else:
            # if above path is a leaf node, return the equivalent of null list (no more features down this path)
            above_branch_list = []

        # same logic for below path as the above path
        if type(current_tree['value_below']) is dict:
            below_branch_list = self._follow_tree_analyze(current_tree['value_below'], current_branch_depth + 1)
        else:
            below_branch_list = []

        # combine together the lists that have been populated through recursion
        branch_list = [current_branch] + below_branch_list + above_branch_list
        return branch_list

    def score_data_set(self, input_df):
        """Scores data in pandas dataframe using forest. Features in forest model must match with column names in the
        dataframe; common case is scoring a training set or validation set. Returns a dataframe with rf_score (random
        forest score) and rf_rank (random forest rank) added as columns.
        """
        # verify that features in data set match features in forest
        self._check_prerequisites(prereq_list=['forest_exists', 'features_in_input_data'],
                                  input_features=input_df.columns)

        # initiate rf_score column, to be populated
        total_rows = input_df.shape[0]
        scored_df = input_df
        scored_df.loc[:, 'rf_score'] = np.nan

        # loop through all the rows in dataframe, which each row represents a data point
        row_counter = 0
        for input_df_index, input_df_row in input_df.iterrows():
            # represent input_df_row as dict, and score using score_data_point
            data_point = input_df_row.to_dict()
            scored_df.loc[input_df_index, 'rf_score'] = self.score_data_point(data_point)

            # progress tracker
            if row_counter % 200 == 0:
                pct_complete = round(row_counter / total_rows * 100, 1)
                logging.info('data set scoring {}% complete'.format(pct_complete))
            row_counter += 1
        logging.info('data set scoring complete')

        # set up rf_rank values, sorted based on rf_score
        scored_df.loc[:, 'rf_rank'] = scored_df.loc[:, 'rf_score'].rank(method='first', ascending=False)
        return scored_df

    def score_data_point(self, data_point):
        """Score a single data point using forest. Data point format can be dict or pandas series (i.e. a dataframe row)."""
        # verify data_point is a valid format
        try:
            assert(type(data_point) is dict or type(data_point) is pd.core.series.Series)
        except AssertionError:
            error_message = 'wrong data type - data point must be dict or pandas series'
            logging.error(error_message)
            raise AssertionError(error_message)

        # if data point is series (i.e. a dataframe row), convert it to dict
        if type(data_point) is pd.core.series.Series:
            data_point = data_point.to_dict()

        # verify that features in data poin match features in forest
        self._check_prerequisites(prereq_list=['forest_exists', 'features_in_input_data'],
                                  input_features=data_point.keys())

        # this represents the sum of scores from all the trees, to be incremented
        rf_score_sum = 0
        
        # iterate through the trees, calling _follow_tree_score to calculate score for every tree in forest
        for tree in self.forest:
            rf_score_sum += self._follow_tree_score(tree, data_point)

        # rf_score averages scores from all the trees
        rf_score = rf_score_sum / len(self.forest)
        return rf_score

    def _follow_tree_score(self, current_tree, data_point):
        """Follows forest down specific branch based on data_point, and returns the leaf value at the end. This method
        calls itself recursively in order to follow tree of any depth.
        """
        # retrieve characteristics of the current branch
        current_feature = current_tree['feature']
        current_threshold = current_tree['threshold']

        # determine which side of the decision threshold is the data - above or below the threshold
        if data_point[current_feature] >= current_threshold:
            branch_label = 'value_above'
        else:
            branch_label = 'value_below'

        # check is the node the final probability (i.e. a leaf) or another tree
        # recursively follow if it is another tree, otherwise return target probability
        if type(current_tree[branch_label]) is dict:
            subtree = current_tree[branch_label]
            return self._follow_tree_score(subtree, data_point)
        else:
            return current_tree[branch_label]

    def create_probability_map(self, scored_df, ntiles):
        """Create a mapping between random forest model scores (rf_score) and actual probability of a data point
        belonging to a class. Overall approach is to extract actual target rates from a scored data set based on
        rf_score ranges. Thus, this method requires a data set that (1) has been scored using score_data_set method
        and (2) has a target variable present. Generally, a scored training set or validation set is ideal. The
        resultant probability map is generated as a pandas dataframe.
        """
        # scored_df needs to contain rf_score and rf_rank (generated by score_data_set method) and target variable
        self._check_prerequisites(prereq_list=['forest_with_metadata_exists', 'dataframe_is_scored', 'dataframe_has_target'],
                                  df_columns=scored_df.columns)

        # create shell for probability map, to be populated
        probability_map_columns = ['ntile', 'rf_rank_min', 'rf_rank_max', 'rf_score_min', 'rf_score_max', 'rf_probability']
        self.probability_map = pd.DataFrame(index=range(ntiles), columns=probability_map_columns)

        # +1 because index starts at 0 and ntile starts at 1 (there is no zero-th ntile)
        self.probability_map.loc[:, 'ntile'] = self.probability_map.index + 1

        # set the min and max ranks within each ntile
        total_points = scored_df.shape[0]
        points_per_ntile = total_points / ntiles
        self.probability_map.loc[:, 'rf_rank_min'] = (self.probability_map.loc[:, 'ntile'] - 1) * points_per_ntile
        self.probability_map.loc[:, 'rf_rank_max'] = self.probability_map.loc[:, 'ntile'] * points_per_ntile

        for i in range(ntiles):
            rank_min = self.probability_map.loc[i, 'rf_rank_min']
            rank_max = self.probability_map.loc[i, 'rf_rank_max']

            # for each ntile, filter scored_df to only rows within the min and max ranks for each ntile
            scored_df_subset = scored_df.query('rf_rank >= @rank_min and rf_rank < @rank_max')

            # user the filtered data set to determine min and max scores for this ntile
            self.probability_map.loc[i, 'rf_score_min'] = scored_df_subset.loc[:, 'rf_score'].min()
            self.probability_map.loc[i, 'rf_score_max'] = scored_df_subset.loc[:, 'rf_score'].max()

            # the true probability for the ntila is the actual target rate found in the ntila
            probability = self._get_target_counts(scored_df_subset) / points_per_ntile

            # if rf_score is higher, use that as the probability to smooth ripples in the curve
            # this is reasonable because rf_score is meant to be a probability anyway (but can be underestimated if trees are not deep enough)

            adjusted_probability = max(probability, self.probability_map.loc[i, 'rf_score_max'])

            # score cannot be higher than 1
            adjusted_probability = min(adjusted_probability, 1)

            self.probability_map.loc[i, 'rf_probability'] = adjusted_probability

        # set ntile as the index of the completed probability map
        self.probability_map.set_index('ntile', inplace=True)
        self.display_probability_map()

    def return_probability_map(self):
        """Return probability_map, which can be used for custom scripting and operations."""
        self._check_prerequisites(prereq_list=['probability_map_exists'])
        return self.probability_map

    def display_probability_map(self):
        """Print of probability_map to console."""
        # sleep a small amount of time so that the print does not mash with log outputs that are also written to console
        time.sleep(0.1)

        self._check_prerequisites(prereq_list=['probability_map_exists'])
        print(self.probability_map.to_string())

    def save_probability_map_as_csv(self, csv_file_location):
        """Saves probability_map in csv format for portability. Can be loaded back later with load_probability_map_csv."""
        self._check_prerequisites(prereq_list=['probability_map_exists'])
        self.probability_map.to_csv(csv_file_location)
        logging.info('probability map saved to {}'.format(csv_file_location))

    def load_probability_map_csv(self, csv_file_location):
        """Loads a csv file of probability_map that was previously saved using save_probability_map_as_csv method."""
        self.probability_map = pd.read_csv(csv_file_location)
        self.probability_map.set_index('ntile', inplace=True)

        logging.info('probability map loaded from {}'.format(csv_file_location))
        self.display_probability_map()

    def apply_probability_map_to_data_set(self, scored_df):
        """Adds probability score to any data set that was generated by score_data_set, by applying the probability
        map to rf_score. Returns a dataframe with rf_probability appended as a column.
        """
        self._check_prerequisites(prereq_list=['dataframe_is_scored', 'probability_map_exists'],
                                  df_columns=scored_df.columns)

        # add rf_probability column to scored_df
        scored_df.loc[:, 'rf_probability'] = np.nan

        total_rows = scored_df.shape[0]
        row_counter = 0

        # iterate through dataframe, executing apply_probability_map_to_score against every row to populate rf_probability
        for scored_df_index, scored_df_row in scored_df.iterrows():
            rf_probability = self.apply_probability_map_to_score(scored_df_row.loc['rf_score'])
            scored_df.loc[scored_df_index, 'rf_probability'] = rf_probability

            # progress tracker
            if row_counter % 200 == 0:
                pct_complete = round(row_counter / total_rows * 100, 1)
                logging.info('data set probability mapping {}% complete'.format(pct_complete))
            row_counter += 1

        return scored_df

    def apply_probability_map_to_score(self, rf_score):
        """Retrieves probability from probability_map, given an rf_score."""
        self._check_prerequisites(prereq_list=['probability_map_exists'])

        # filter down the probability_map, based on the input value
        probability_map_filtered = self.probability_map.query('rf_score_min <= {0} and rf_score_max >= {0}'.format(rf_score))

        # check row count of filtered map to see if rf_score is within range between rf_score_min and rf_score_max
        if probability_map_filtered.shape[0] > 0:
            # if within range, map rf_score to rf_probability
            rf_probability = probability_map_filtered.loc[:, 'rf_probability'].max()
        else:
            # if score is above the max probability, set rf_probability to 1;
            # if below 0, set to 0
            # otherwise, the score itself represents the probability
            rf_probability = max(min(rf_score, 1), 0)

        return rf_probability

    def _check_prerequisites(self, prereq_list, **kwargs):
        """Centralized method to check requirements before running certain operations. E.g. a forest needs to have been
        created or loaded before analyze_forest run. Many of the methods in this class have prerequisites. The kwargs
        are reserved for additional inputs that need to be provided for some of the prerequisite checks.
        """
        # prerequisite provided as a list to allow for multiple prereqs
        # all prereqs throw assertion errors if condition is not properly met
        for prereq in prereq_list:

            # prereqs for operations that require a forest has already been created or loaded
            if prereq in ['forest_exists', 'forest_with_metadata_exists']:
                try:
                    if prereq == 'forest_exists':
                        assert(self.forest is not None)
                    elif prereq == 'forest_with_metadata_exists':
                        assert(self.forest_with_metadata is not None)
                except AssertionError:
                    error_message = 'No forest exists yet. First create a new forest (using grow_forest method) or ' \
                                    'load an existing forest (using load_forest_json method).'
                    logging.error(error_message)
                    raise AssertionError(error_message)

            # prereqs for scoring operations that require model features exist in the inputs
            # expects input_features in kwargs
            if prereq == 'features_in_input_data':
                input_features = kwargs.get('input_features')
                for feature in self.features_in_forest:
                    try:
                        assert(feature in input_features)
                    except AssertionError:
                        error_message = '\'{}\' is a required feature for scoring but is not in the submitted data.'.format(feature)
                        logging.error(error_message)
                        raise AssertionError(error_message)

            # prereqs for probability_map operations that require input dataframe to have been scored
            # expects df_columns in kwargs
            if prereq == 'dataframe_is_scored':
                df_columns = kwargs.get('df_columns')
                try:
                    assert('rf_score' in df_columns and 'rf_rank' in df_columns)
                except AssertionError:
                    error_message = 'Input dataframe is missing columns rf_score and rf_rank. These fields are ' \
                                    'generated by processing the dataframe using the score_data_set method first.'
                    logging.error(error_message)
                    raise AssertionError(error_message)

            # prereqs for probability_map operations that require input dataframe to contain a target variable
            # expects df_columns in kwargs
            if prereq == 'dataframe_has_target':
                df_columns = kwargs.get('df_columns')
                try:
                    assert(self.target_var in df_columns)
                except AssertionError:
                    error_message = 'Input dataframe is missing a target variable.'
                    logging.error(error_message)
                    raise AssertionError(error_message)

            # prereqs for operations that require that a probability_map has been created or loaded
            if prereq == 'probability_map_exists':
                try:
                    assert(self.probability_map is not None)
                except AssertionError:
                    error_message = 'Probability map does not exist yet. First generate probabilility map (using create_probability_map method) ' \
                                    'or load an existing probability map (using load_probability_map_csv method).'
                    logging.error(error_message)
                    raise AssertionError(error_message)
