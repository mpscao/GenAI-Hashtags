Goal of model is to identify similarities between hashtags generated by Google Gemini for different "players" when given a prompt and to connect players with similar hashtags to their neighbors. 
Each "player" will have multiple rounds to generate a hashtag associated with an event and in each round, they are randomly assigned to a neighbor. If neighbors choose the same hashtag, they both get a point. "Players" can use their previous neighbor's selected hashtag to influence their decisions.


NOTE: If you are using the "free tier" of Gemini, you may get a resources exhausted error. I included a delay between API requests to attempt to alleviate the issue.
