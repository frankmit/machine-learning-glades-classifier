# Glades (Random Forest Classifier)

Author: Frank Lo  
Contact: franklo@alum.mit.edu  
Created: December 2013

## What is glades?
Glades is a random forest (RF) algorithm written in python and designed with the following in mind:

* Easy to control RF tuning parameters
* Easy to visually inspect trees
* Forests represented and stored as JSON (portable and easy to parse)

This suite of random forest tools enables the following capabilities:

* Train RF models
* Inspect, save, and load trained RF models
* Analyze RF models for feature importance
* Score data against RF models
* Transform RF model scores into predicted target probabilities

## GladesClassifier reference
All functionality is built into the GladesClassifier class. Below is an overview of methods:

###grow\_forest(df, feature\_space, target\_var, **kwargs)

>####Description:
>Trains a random forest based on input data set and a variety of tuning parameters.

>####Parameters:

>* **df** (*pandas dataframe*) -- [required]  
Input data set - includes dependent variable (a.k.a. target) and all features as columns.

>* **feature\_space** (*list*) [required]  
List of names of independent variables in data to be considered in forest.

>* **target\_var** (*str*) -- [required]  
Name of dependent variable in data.

>* **target\_value** (*int, float, or str*)  
Value to look for in target\_var to indicate True.
E.g. if target\_var is a binary variable 0 or 1, then target_value would be 1.  
Default value is 1.

>* **tree\_count** (*int*) -- [optional]  
Number of trees to grow in the forest.  
Default value is 50.

>* **bootstrap\_size** (*int*) -- [optional]  
For each tree, what size subset of data should be taken for each bootstrap sample.  
Default value is 500.

>* **subspace\_size** (*int*) -- [optional]  
For each tree, the number of features that should be randomly selected from the feature_space to form a subspace to build the tree.  
Default value is 5.

>* **scan\_granularity** (*int*) -- [optional]  
At each node, every feature is scanned through from its min to max, looking for an optimal cut. This setting indicates the number of 'slices' to iterate through. Higher means more granular.  
E.g. if a feature X has range 0-100, and scan\_granularity=100, then algorithm will consider branching at X at 1,2,3 ... 97,98,99. If scan\_granularity is set to 1000, then algorithm will consider branching at X at 0.1,0.2,0.3 ... 0.97,0.98,0.99.  
Default value is 100.

>* **min\_points\_per\_leaf** (*int*) -- [optional]  
At each potential fork, algorithm will consider branching, or skip to next potential fork if not enough points exist. min\_points\_per\_leaf indicates the minimum number of points needed to consider branching.  
E.g. if min\_points\_per\_leaf=20, then if a potential fork has less than 20 points either above or below, it is forced to be terminated as a leaf. Otherwise, it is possible to split to deeper nodes.  
Default value is 20.

>* **max\_tree\_depth** (*int*) -- [optional]  
Maximum depth that a tree can reach. Higher depth means more complex trees.  
Default value is 3.

>####Return:
>No return value. Trained tree is stored as an instance variable which can be returned by running **return\_forest** or saved by running **save\_forest\_as\_json**.





###return\_forest()

>####Description:
>Return forest\_with\_metadata, which can be used for custom scripting and operations.

>####Prerequisites:

>* Forest must first already exist in the class instance, either created with **grow\_forest** or loaded with **load\_forest\_json**.

>####Return:
>Returns forest with metadata in dict format.




###display\_forest(**kwargs)

>####Description:
>Clean visual print of forest for easy inspection. View branches at different depth levels with clean indentation. At every node, view features, decision thresholds, and scores.

>####Prerequisites:

>* Forest must first already exist in the class instance, either created with **grow\_forest** or loaded with **load\_forest\_json**.

>####Parameters:

>* **include_metadata** (*bool*) -- [optional]  
If true, also prints metadata around what tuning params were used.

>####Return:
>No return value.




###save\_forest\_as\_json(json\_file\_location)

>####Description:
>Saves forest (with metadata) in json format for portability. Can be loaded back later with **load\_forest\_json**.

>####Prerequisites:

>* Forest must first already exist in the class instance, either created with **grow\_forest** or loaded with **load\_forest\_json**.

>####Parameters:

>* **json\_file\_location** (*str*) -- [required]  
File location (with .json extension) to save the forest.

>####Return:
>No return value.




###load\_forest\_json(json\_file\_location)

>####Description:
>Loads a json file of forest\_with\_metadata that was previously saved using save\_forest\_as\_json method, and resets all instance variables back to appropriate values.

>####Parameters:

>* **json\_file\_location** (*str*) -- [required]  
File location (with .json extension) of saved forest to load.

>####Return:
>No return value.




###analyze\_forest(**kwargs)

>####Description:
>Analyzes forest and prints out feature importance ratings. This feature importance is based on frequency of feature being utilized to branch in the forest, as well as branch depth when utilized (more top level branches carry more weight, while lower branches carry less weight).

>####Prerequisites:

>* Forest must first already exist in the class instance, either created with **grow\_forest** or loaded with **load\_forest\_json**.

>####Parameters:

>* **return\_data** (*bool*) -- [optional]  
>If true, returns feature importance ratings.

>####Return:
>If *return\_data==true*, returns feature importance ratings in dict format. Otherwise, no return value.



###score\_data\_set(input\_df)

>####Description:
>Scores a data set with random forest model.

>####Prerequisites:

>* Forest must first already exist in the class instance, either created with **grow\_forest** or loaded with **load\_forest\_json**.

>* Features in random forest model must match with column names in the input dataframe. Common case is scoring the training set used to build the model, or validation set or other data set associated with the model.

>####Parameters:

>* **input\_df** (*pandas dataframe*) -- [required]  
>Expects a pandas dataframe - see prerequisites note above.

>####Return:
>Returns a copy of input\_df (pandas dataframe) with **rf\_score** (random forest score) and **rf\_rank** (random forest rank) added as columns and populated with values.




###score\_data\_point(data\_point)

>####Description:
>Scores a single data point with random forest model.

>####Prerequisites:

>* Forest must first already exist in the class instance, either created with **grow\_forest** or loaded with **load\_forest\_json**.

>* Features in random forest model must match with keys/columns in the data point.

>####Parameters:

>* **data\_point** (*dict or pandas series*) -- [required]  
>Data point represented as a dict or pandas series (i.e. a dataframe row) - see the prerequisites note above.

>####Return:
>Returns a float which represents the random forest score for the data point.




###create\_probability\_map(scored\_df, ntiles)

>####Description:
>Create a mapping between random forest model scores (rf\_score) and actual probability of a data point belonging to a class. Overall approach is to extract actual target rates from a scored data set based on rf\_score ranges. Generally, this step should run to build the probability map right after the **score\_data\_set** method is run on a training or validation set.

>####Prerequisites:

>* Forest must first already exist in the class instance, either created with **grow\_forest** or loaded with **load\_forest\_json**.

>* Input dataframe is expected to be a dataframe that has been scored by running the **score\_data\_set** method. Specifically, the fields **rf\_score** and **rf\_rank**, which are columns generated by scoring the data set, must be present in the input data.

>* Input dataframe must contain the same target variable field that was used to train the forest. Hence, a scored training set or validation set is ideal, because it will already contain the target variable.

>####Parameters:

>* **scored\_df** (*pandas dataframe*) -- [required]  
>Expect a pandas dataframe, used to generate the probability map - see prerequisites note above.

>* **ntiles** (*int*) -- [required]  
>Specifies granularity of probability map that is returned. E.g. if *ntile=100*, then every centile is assigned its own probability; if ntile=10, then every decile is assigned its own probability.

>####Return:
>Returns a pandas dataframe which represents the newly created probability map - i.e. a score mapping between score calculated by random forest, and target probability




###return\_probability\_map()

>####Description:
>Returns probability\_map, which can be used for custom scripting and operations.

>####Prerequisites:

>* Probability map must first already exist in the class instance either created with **create\_probability\_map** or loaded with **load\_probability\_map\_csv**.

>####Return:
>Returns probability map as a pandas dataframe.




###display\_probability\_map()

>####Description:
>Print of the probability map for easy inspection.

>####Prerequisites:

>* Probability map must first already exist in the class instance either created with **create\_probability\_map** or loaded with **load\_probability\_map\_csv**.

>####Return:
>No return value.



###save\_probability\_map\_as\_csv(csv\_file\_location)

>####Description:
>Saves probability map in csv format for portability. Can be loaded back later with **load\_probability\_map\_csv**.

>####Prerequisites:

>* Probability map must first already exist in the class instance either created with **create\_probability\_map** or loaded with **load\_probability\_map\_csv**.

>####Parameters:

>* **csv\_file\_location** (*str*) -- [required]  
File location (with .csv extension) to save the probability map.

>####Return:
>No return value.



###load\_probability\_map\_csv(csv\_file\_location)

>####Description:
>Loads a csv file of probability\_map that was previously saved using save\_probability\_map\_as\_csv method.

>####Parameters:

>* **csv\_file\_location** (*str*) -- [required]  
File location (with .csv extension) of saved probability map to load.

>####Return:
>No return value.




###apply\_probability\_map\_to\_data\_set(scored\_df)

>####Description:
>Maps RF score to actual target probabilities for all data in the input data set, based on probability map.

>####Prerequisites:

>* Probability map must first already exist in the class instance either created with **create\_probability\_map** or loaded with **load\_probability\_map\_csv**.

>* Input dataframe is expected to be a dataframe that has been scored by running the **score\_data\_set** method. Specifically, the fields **rf\_score** and **rf\_rank**, which are columns generated by scoring the data set, must be present in the input data.

>####Parameters:

>* **input\_df** (*pandas dataframe*) -- [required]  
>Expects a pandas dataframe - see prerequisites note above.

>####Return:
>Returns a copy of scored\_df (pandas dataframe) with **rf\_probability** (target probability) added as a column and populated with values.




###apply\_probability\_map\_to\_score(rf\_score)

>####Description:
>Maps a single RF score to actual target probability, based on probability map.

>####Prerequisites:

>* Probability map must first already exist in the class instance either created with **create\_probability\_map** or loaded with **load\_probability\_map\_csv**.

>####Parameters:

>* **rf\_score** (*float*) -- [required]  
>A float value, representing RF score from a scored data point.

>####Return:
>Returns a float which represents a target probability.



##Example Code
The following is a small-scale demo of the functionality provided by glades, using the famous iris data set. This first part demonstrates model training and data set scoring, then concludes by saving random forest model the probability map as files.

    import glades
    import pandas
    
    # load the iris data set
    iris_data_location = \
        "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
    iris_data = pandas.read_csv(iris_data_location)
    
    # instantiate the glades class
    rf = glades.GladesClassifier()
    
    # train a forest with custom tuning parameters;
    # predicts whether iris species is versicolor,
    # based on sepal and petal measurements
    rf.grow_forest(
        df=iris_data,
        feature_space=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        target_var='species',
        target_value='versicolor',
        tree_count=25,           # forest will have 25 trees
        bootstrap_size=50,       # each tree will be built from 50 random observations
        subspace_size=3,         # each tree will be grown using 3 random features
        min_points_per_leaf=10,  # stop branching if a leaf would have less than 10 obs
        scan_granularity=100,    # scan 100 potential forks for each feature
        max_tree_depth=3         # no tree can be more than 3 trees deep
    )
    
    # inspect the forest
    rf.display_forest()
    
    # show feature importance ratings from the forest
    rf.analyze_forest()
    
    # score the data set using the forest, and save as iris_data_scored
    # generates columns rf_score and rf_rank
    iris_rf_scores = rf.score_data_set(iris_data)
    
    # create the probability map at decile granularity,
    # based on target probabilities from iris_data_scored
    rf.create_probability_map(scored_df=iris_rf_scores, ntiles=10)
    
    # get target probabilities for every data point
    # generates column rf_probability
    iris_probability_scores = rf.apply_probability_map_to_data_set(iris_rf_scores)
    
    # save the forest and probability map into portable files
    rf.save_forest_as_json('iris_rf_model.json')
    rf.save_probability_map_as_csv('iris_probability_map.csv')

The following demonstrates loading a previously saved forest and probability map, along with some additional methods in glades.

    # load the previously saved the forest and probability map from files
    rf.load_forest_json('iris_rf_model.json')
    rf.load_probability_map_csv('iris_probability_map.csv')
    
    # return feature importance values
    feature_importance = rf.analyze_forest(return_data=True)
    
    # return forest and probability map
    my_forest = rf.return_forest()
    my_probmap = rf.return_probability_map()
    
    # display forest with metadata; display probability map
    rf.display_forest(include_metadata=True)
    rf.display_probability_map()
    
    # for a new observation, get model score and target probabilities
    # that species is versicolor
    new_obs = {
        'sepal_length': 6.2,
        'sepal_width': 2.2,
        'petal_length': 1.5,
        'petal_width': 1.9
    }
    new_obs_score = rf.score_data_point(new_obs)
    new_obs_probability = rf.apply_probability_map_to_score(new_obs_score)
