---
layout: post
title: PIC16B Project Reflection
image: benbrill.github.io\images\ucla-math.png
---
I’ve always viewed music as a connective medium; it’s something that people across all different cultures and languages can sit down and admire together. With today’s advancements in streaming services, music has never been more accessible.

However, I often find that people -- including myself -- have a difficult time fully verbalizing their music taste. Especially when music taste’s are so diverse, it can be difficult to express that variety of music you listen to into a couple of sentences. That’s why Spotify rewind is so popular on social media; it does all the summarizing for you.

Since discovering the Spotify API, a database that contains statistics on the musicality, energy, and even “wordiness” of hundreds of millions of songs, I saw the potential to use this data to aid in communicating music tastes between people in an automated fashion.

Apparently, this was not an uncommon thought. Kendall and Michael had similar thoughts on how we can best leverage this unique dataset. Kendall came up with the idea of trying to match different feelings and emotions to songs that you commonly listen to. That way, when comparing music tastes with another person, you would be able to see what you and your partner view as a “sad” or “exciting” song, making sharing music tastes more streamlined. And what better way to convey emotion through movies?

Our final product, Moodspace, is an app that lets you choose from 10 famous movie scenes. Once you select a movie scene, our algorithm outputs three songs that lie in your Spotify library that match the mood of the scene. You can then compare these songs to other people to see how they could best score this movie scene given their music taste.

### What am I proud of?

I’m especially proud of how nice and polished the final product came out to be. The front-end (mostly thanks to the aesthetic eye of Kendall) was inspired by the bright colors and gradients present in the Spotify UI. Our app is based on their API after all. The app is easy to navigate, and has nice interactive features that make it feel as though this is a semi-professional service, not something thrown together by some college students in seven weeks. 

In addition, I am also proud of how the algorithm that selects songs turned out working, and specifically the amount of time and work I put in personally to make its performance the best that it can be. A problem throughout the entire process was how we would convert a script (which is text) to have the same Spotify metrics that a song might have (which actually has musical qualities you can quantitatively measure). We originally tried to cluster song lyrics based on these traits, and train a neural network to classify the cluster based off of the lyrics. We would then apply the same model on the text of a given movie scene to bridge that divide.

This model worked, for a bit. But we realized as we were finalizing our project that it produced similar outputs for every movie scene. It appeared as though the clustering foundation the model was built on was unsustainable for making predictions, basically ensuring that all songs would have distinct weights. This was not good because this was the key feature of our project.
So I spent about 7 hours straight two days before the project was good revamping the model. Actually, I wouldn’t say revamping. I wrote an entirely new model based on directly predicting the Spotify metrics instead of predicting a cluster derived from the Spotify metrics. This proved much more effective, producing different songs for each movie that appeared to match the context of the scene, like “What a man gotta do” for Syndrome’s villainous monologue in “The Incredibles”  or “New Beginnings” for the escape scene from “Shawshank Redemption.”

### What could we improve on?

Though we successfully implemented the algorithm, we did not have time to implement a feature that would allow users to give feedback on how the algorithm is performing. The neural network performs well on a loss standpoint, but we have no way of knowing if actual humans think these predictions are valid aside from ourselves. Implementing some sort of Google Form like the Puppy Team implemented, which automatically retrains the model according to the feedback could make the model more effective in the eyes of the user.

In addition, it would be nice to also try training our model on languages other than english, opening up our model’s usage to hundreds of thousands of more songs. Say, for instance, you listened to Bad Bunny or BTS enough to make those artists comprise your top songs. Since our model is not trained on languages other than English, it would simply eliminate these songs from its analysis. Millions of people listen to songs in Spanish, Korean, and hundreds of other languages, so it would be nice to incorporate those songs into the model in the future.

### What didn't we end up doing?

Surprisingly, we ended up completing most of what we had outlined in our proposal. We had a nice UI, a good backend, a decent list of notable movie scenes to choose from, and an algorithm that outputted song suggestions. However, instead of creating a playlist for the user to immediately access on their Spotify account, we simply embedded songs for the user to play right then. Initially, our team was called “Team Playlist,” so this would have been a good goal to accomplish. However, the logistics were too difficult and in fact too invasive to make this a reality currently. 


### What did I learn?

Perhaps my biggest technical takeaway from this project was developing a backend in Python, specifically in Flask. All credit goes to Michael for discovering Flask and outlining its basic usage to us. I now feel comfortable knowing how to organize and write a backend and how to integrate it with the front-end. 

In addition, I feel a lot more comfortable using Git and Github after this project. We had over 100 commits total, so none of the collaboration we did would have been possible without Github. I learned the importance of making timely commits to avoid merge errors, and even learned how to make new branches when experimenting with new code to make sure we don’t break anything.

Finally, I furthered my understanding of the structure of Neural Networks. This was mostly done through my seven hour code sprint to redo our entire model, but I learned about different structures of Neural Networks created using Tensorflow, experimented with different layers to determine what worked and what didn’t, and tried different losses and compilation setups to yield the best results. 

### How does this help my future?

Completing this project has already given me something to discuss in job/internship interviews and opportunities. When I presented this project, people were impressed with the professional look of the website, the Flask backend, as well as the method used to generate the neural network and get song predictions. It’s a very nice thing to stick into my portfolio.

However, I think even more valuable are the collaboration skills I learned from completing this long term project with two other people. We learned to determine what we should work on as a team and what to keep separate to take advantage of our strengths and weaknesses as programmers and data scientists. In my academic and professional career, I expect to be working in teams very often, so it was a pleasure creating a finished product with my two partners that I can carry with me to my future in teamwork.
