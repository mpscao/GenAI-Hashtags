Goal of model is to identify similarities between hashtags generated by LLM Models such as Llama for different "players" when given a prompt and to connect players with similar hashtags to their neighbors. 
Each "player" will have multiple rounds to generate a hashtag associated with an event and in each round, they are randomly assigned to a neighbor. If neighbors choose the same hashtag, they both get a point. 

"Players" can use their previous neighbor's selected hashtag to influence their decisions. Players will also be asked to generate a tweet associated to the event.

For players in the llamallm_effect_last_neighbor.py file, they will be also asked to generate 5 effects they believe will be a result of the event described in the file. 

