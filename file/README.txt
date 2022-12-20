#### HowTo100M dataset file description ####

We provide the following files:

####CSVs:

----HowTo100M_v1.csv----: 
This CSV contains all of the HowTo100M YouTube video urls.
Column description: 
    --video_id: YouTube video id
    --task_id: task id (see task_ids.csv for more details)
    --rank: YouTube search result rank of the video when querying the task
    --category_1: Highest level task category from WikiHow
    --category_2: Second Highest level task category from WikiHow

----task_ids.csv----:
This CSV contains the mapping between the task_id and the task description / name
we have scrapped from WikiHow.

####JSON:

----caption.json----:
This json file contains the captions for all the HowTo100M videos.
It is stored as a dictionary where each key is a video_id.
Each value of the dictionary is another dictionary with the keys ['text', 'start', 'end']
where the value of 'text' is a list of all the captions from the given video_id,
and 'start' and 'end' are arrays correspondings to the start and end time timestamp of the captions
(in second).

Note that the narrated captions have been processed.
In fact, we have removed a significant number of stop words
which are not relevant for the learning of the text-video joint embedding.
The list of stop words can be found here: https://github.com/antoine77340/howto100m/blob/master/stop_words.py

You can find the unprocessed caption file (i.e. with stop words) here: https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/raw_caption.zip
