# Topic Modeling with Emails

## Background
In the modern age of information about 56.5% of emails sent in 2022 were spam. This presents a problem for the average person since they are made to see and read spam and unimportant emails that they don't personally care about. This detracts attention away from actual important mail. Our goal in this project is to create an email clustering system that accurately clusters emails based on specific topics a user may want to see in their inbox with an eventual hope of using it for filtration as a whole.

## Problem Definition
Researchers, students, and professionals all use their emails as a central point of communication. However, because of how universally used emailing is, it often becomes subject to cluttering and a lack of organization with incoming messages of varying importance to receivers. This can lead to a backlog of important messages being obscured by less relevant ones, causing very urgent circumstances and concerns to be unresolved and both incoherence as well as reduced productivity for thousands of individuals—the scale of this problem and ubiquity of email is what makes these consequences even more alarming.

## Methods
We plan to approach solving this problem using 3 main ML pipelines...
1. TF-IDF, PCA, and PAM Pipeline:
   - TF-IDF: It assesses the relevance of frequency of phrases and keywords across documents. By evaluating a phrase's significance across a document corpus, we can pinpoint crucial keywords within emails.
   - PCA: This technique reduces the dimensionality of the keywords extracted via TF-IDF, emphasizing those that yield substantial importance for clustering.
   - Pachinko Allocation Model (PAM): A hierarchical topic modeling approach for captures correlations between vectorized representations of emails for clustering them into topics.

2. Doc2Vec and Lbl2Vec Pipeline:
   - Doc2Vec: An extension of word2vec, it allows for capturing the semantic essence of entire documents. Using this, each email is converted into fixed-size vector embeddings.
   - kNN Model: Post vectorization, emails are clustered where centroids represent semantic essence.
   - Lbl2Vec: This model deciphers the vector embeddings’ meaning by associating them with labels.

3. Non-Negative Matrix Clustering Pipeline:
   - Document Term Matrix (DTM): Documents are transformed into a DTM wherein each row represents an email and columns correspond to words. 
   - Non-Negative Matrix Clustering: This algorithm factors the DTM into non-negative matrices of 'k' different topics giving us a way to find the given topic of the email.

## Dataset
As we’ll need an extensive training data set of emails to facilitate the above methods, we plan to use the Enron dataset, which contains emails of about 150 users and has been compiled with 500,000 messages. This dataset contains information about sender, receiver, timestamp when it was send, subject lines and the actual body of text. We plan to mainly use the subject lines and body from the emails.

## Potenial Results and Discussion
As this is an unsupervised tasks the metrics we use will be especially important in order to get clear and coherent clusters that accurately model the topic. The metrics we plan to use are related to topic coherence. To give a brief intuition behind the metric, our general goal is to maxmize the similarity between documents/elements in a given cluster and maximize the difference of similarity between documents/elements of different clusters. This can be done quite easily using the topic coherence metrics based around UMass’ formulation and Word2Vec usage.

[Gantt and Contribution Chart](https://docs.google.com/spreadsheets/d/1ZUl8Xywp4VTTNtC-8Wq8ZxpYnzXYNJLe/edit?usp=sharing&ouid=101698207149759013919&rtpof=true&sd=true)
