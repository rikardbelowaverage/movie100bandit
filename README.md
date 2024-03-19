# Contextual Multi-armed Bandit for personalized recommendations
#### By Rikard Johansson & Hannes Nilsson
###### Tags: `MAB, decision under uncertainty, reinforcement learning, recommendation systems`

## :memo: Summarized Project Overview
This project will provide personalized recommendation systems based on the bandit framework. More precisely, the agent is presented with a user at each timestep.
The agent's job is to predict the most suitable movie to recommend. The agent decides to present a movie and receives the reward.

## :cd: The dataset
The dataset contains approximately 100,000 ratings on 1682 movies by 943 different users.
The users' data contains contextual demographical features such as age, gender, occupation, and zip code.
Additionally, each user has rated at least 20 movies.
The movie data contains the release date, URL, and one-hot encoded.

#### Reference
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent
Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
DOI=http://dx.doi.org/10.1145/2827872

## Our approach
At each timestep t, the agent is presented with the contextual features of the user which we use as the environment. 
The agent is also presented with the movie history by the user excluding the rating.
All movies in the view history of the user are seen as actions or arms. Thus, the action space will be dynamic and the agent must learn to generalize.

This allows us to encode the contextual data as a vector x = [User data, movie data] and try to predict the rating.
We do this for all ratings providing us a matrix X of size (rated_movies x contextual_features).

The agent's job is then to predict which movie that had the highest rating from the user history delicately balancing the explore-exploit trade-off.
Once the agent has presented the movie to the user, the agent receives the user's true rating only for the recommended movie as bandit feedback.
Then, we store the data and the reward of the recommended movie as X and y. The agent history is available for the agent to gain knowledge and improve their predictions.
When a movie has been recommended to a specific user, the movie is removed from the view history as we do not want to keep recommending the same movie even if the user appreciated it.

Initially, we will employ TETS and TEUCB to evaluate how well tree ensemble methods perform on the MovieLens100k dataset. Full algorithm specifications and results for other datasets are available at https://arxiv.org/abs/2402.06963.




