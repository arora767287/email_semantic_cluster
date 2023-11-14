# Topic Modeling with Emails

## Introduction

### Background

The task of sorting through emails is a tedious and time-consuming activity that impacts numerous users daily. With the staggering statistic that 56.5 percent of all emails in 2022 were categorized as spam, the demand for efficient and intelligent email filtering systems is higher than ever. Despite existing measures, current systems still fall short, frequently permitting spam and irrelevant messages to infiltrate inboxes. This inefficiency not only leads to a loss of productivity as users sift through unwanted emails, but it also increases the likelihood of overlooking crucial correspondence. Our project is dedicated to developing an email clustering system, leveraging machine learning to organize emails into user-defined topics, ensuring that recipients are presented with emails that align with their interests and priorities.

### Dataset
We are using the [Enron Dataset](https://www.cs.cmu.edu/~enron/) to gain an extensive set of emails for use in our methods. This dataset contains emails of about 150 users and has been compiled with 500,000 messages. This dataset contains information about sender, receiver, timestamp when it was sent, subject and body of the email. For the purposes of this project we used solely the email information in the dataset.

To understand our dataset, we can add some visualizations. 

<img width="1185" alt="image" src="https://github.com/arora767287/email_semantic_cluster/assets/44822455/b38e8faf-fb7c-4aea-93ac-33fa4dff81d8">

This shows the people who sent the most emails during this time and how many emails they sent. 

We also have a word cloud that shows which words were sent the most number of times during the history of the company. 

![image](https://github.com/arora767287/email_semantic_cluster/assets/44822455/fb53838a-ed5b-4444-9fc4-03995a79115a)

We also have the number of emails sent over time: 

<img width="660" alt="image" src="https://github.com/arora767287/email_semantic_cluster/assets/44822455/35dbe04e-28bb-4f03-9445-90baf4811855">





## Problem Definition

Email serves as a pivotal communication hub for a wide array of individuals including researchers, students, and professionals. The ubiquity of its use, however, renders it susceptible to congestion and disarray due to the volume of incoming messages that vary in significance. Such disorder can result in vital communications being overshadowed by those of lesser relevance, potentially leaving pressing issues unaddressed. This leads to disorganization and a decline in efficiency for countless users. Given the extensive reliance on email for daily exchanges, the ramifications of this disorganization are particularly concerning, highlighting the critical need for more refined management solutions.

## Data Preprocessing

In our project, preprocessing the Enron emails dataset was a fundamental step to ensure the data's integrity and usability for machine learning analysis. We started by loading the dataset into a pandas DataFrame, focusing primarily on the email body content. The initial stage of preprocessing involved a comprehensive cleaning process where we removed HTML tags, special characters, and unnecessary whitespace from the emails. This cleaning was crucial to eliminate noise and standardize the text. In addition to cleaning, we also normalized the text by converting it to lowercase, ensuring uniformity and preventing case sensitivity from affecting our analysis.

After cleaning and normalizing the text, we addressed any missing values in the dataset. Handling these missing values was essential to maintain the integrity of our analysis. This step was followed by the tokenization of the email content, where we broke down the text into individual words. Tokenization is a key process in text mining as it prepares the data for feature extraction and modeling.

The next significant step in our preprocessing was the removal of common English stop words. Eliminating these words allowed us to focus on the more meaningful content of the emails. Once the dataset was cleaned, normalized, tokenized, and freed of stop words, we vectorized it. This technique transformed the text into a numerical format, emphasizing the importance of specific terms within the corpus.

This thorough preprocessing of the Enron emails dataset was a critical foundation for our project. It enabled us to effectively apply our models which were pivotal in clustering the emails and achieving our project's goal of creating an efficient email categorization system.

## Methods

In this midterm report, we have implemented and compared two distinct models designed to organize emails into coherent clusters, facilitating easier navigation and management for users based on their topical preferences.

### Model 1: Doc2Vec+K-means

Given the raw text data from each of the emails of the processed Enron dataset, we decided to sample 20% of the total emails due to computational complexity related to Doc2Vec. Note one thing that should be mentioned is that this sampling is different from Model 2's (which sampled 80% of the total emails). The reason for this was that we ran into issues related to training Doc2Vec taking far too long to train to convergence if we used all the emails. One thing that should be noted however is that as we sample 20% which is greater than 100 / 10 =  10% of the total population of the enron dataset (by the 10% condition) this sample is generalizable to the complete population.

Using this subset of the data, we trained an unsupervised Doc2Vec model to extract a semantically significant 256-dimensional vector of each document. Doc2Vec is an extension of the Word2Vec model, which increases the scope to include document-level context, rather than just word-level. This approach allows the model to learn not only the significance of individual words, but also how these words come together to convey meaning in a larger text. This model was trained for 5 epochs to achieve convergence for the meaning in these vectors.

Given these semantically significant embeddings, we ran PCA, an unsupervised dimensionality reduction algorithm, to “narrow” and “focus” the embeddings on the most important semantic differences between the document. This allowed for clearer and more significant clustering in the last step (k-means) while also keeping the computational complexity of the model pipeline as a whole under control.

These reduced embeddings were then passed into the k-means algorithm in order to cluster the documents into “topics.” The k-means algorithm is a centroid-based clustering method that partitions the data into K distinct, non-overlapping subgroups, or clusters. This model is straightforward and allows for quick retrieval of similar emails, although it requires a careful choice of 'k' will significantly impact the granularity of the clustering, requiring careful tuning to strike a balance between overgeneralization and fragmentation of topics.

We ran KMeans with target cluster number inputs from 2 to 50 in order to find the best clustering (most separable) to use as the final model for this pipeline. 

<img width="396" alt="image" src="https://github.com/arora767287/email_semantic_cluster/assets/44822455/5ecccff4-5f36-4dbe-8409-9afb84d2dce3">

We found that the optimal number of clusters is 9. We examined a number of different clusters to reach this conclusion. 

<img width="537" alt="image" src="https://github.com/arora767287/email_semantic_cluster/assets/44822455/68c47523-b145-43f0-a072-1bd1b463b68a">
<img width="539" alt="image" src="https://github.com/arora767287/email_semantic_cluster/assets/44822455/7c2395a7-e71a-42b4-af30-83dd4377825b">
<img width="525" alt="image" src="https://github.com/arora767287/email_semantic_cluster/assets/44822455/3911d38c-d8a2-47a6-bbb2-0f4a7e50fc21">
<img width="528" alt="image" src="https://github.com/arora767287/email_semantic_cluster/assets/44822455/02461cf3-1414-4572-86ac-b3a0532e6634">

### Model 2: TF-IDF+LDA

Initially, we employed the TF-IDF technique, a widely-used feature extraction method in text mining, to convert our email dataset into a format suitable for machine learning. This method works by quantifying the importance of a word in a document relative to a collection of documents or corpus. The TF-IDF model was instantiated with parameters designed to optimize performance: it ignores terms appearing in more than 85% of documents (max_df=0.85), disregards terms that appear in less than two documents (min_df=2), and excludes common English stop words to focus on more meaningful terms. After fitting our model to the email contents, we obtained a sparse matrix of TF-IDF vectors, representing the significance of words across the emails.

To further enhance our understanding of the data, we applied Truncated Singular Value Decomposition (SVD) to the TF-IDF vectors. This dimensionality reduction technique was used to condense the information into a manageable number of components, specifically four in our case, while preserving the essential characteristics of the dataset. The resultant lower-dimensional representation was then visualized using a scatter matrix plot, enabling us to observe potential clusters and patterns in the email data.

![truncatedSVD](https://github.com/arora767287/email_semantic_cluster/assets/82481744/20ed10dc-4b44-439d-8237-832c7097e1b2)

Parallel to this, we implemented the LDA model, a probabilistic technique for topic modeling. This method aims to discover the latent topics that pervade a large collection of documents, making it particularly suitable for our goal of clustering emails by topic. To prepare our data for the LDA model, we first tokenized our emails and created a dictionary and corpus. The LDA model was then configured with five topics, reflecting our aim to categorize emails into a manageable number of groups. Upon training the model, each email was represented as a distribution over these topics.

Finally, to visualize the results of the LDA model, we transformed the topic distributions into a DataFrame and plotted them using a scatter matrix. This visualization provided us with a clear view of how the emails were distributed across the different topics, thus fulfilling our objective of clustering emails to enhance inbox management.


![LDA](https://github.com/arora767287/email_semantic_cluster/assets/82481744/dd29c01d-ed74-435f-91d0-ce6e9f70e1ce)



## Results and Discussion

Model 1: Doc2Vec+K-means

Davies-Bouldin Index: 6.040773131121281

Calinski-Harabasz Index: 1479.2366698699566


Model 2: TF-IDF+LDA

Davies-Bouldin Index: 0.3863525829481374

Calinski-Harabasz Index: 1630166.5380492713



<img width="586" alt="image" src="https://github.com/arora767287/email_semantic_cluster/assets/44822455/3f736ac0-4752-4bad-b16d-dbd6d3386464">

**Calinski-Harabasz Score**

<img width="545" alt="image" src="https://github.com/arora767287/email_semantic_cluster/assets/44822455/8d62af1f-2a47-4efc-8357-c570a23344f3">

**Davies-Bouldin Index**

**Images: We see that the elbow for both the Calinski-Harabasz Score and Davies Bouldin Score is around 9 clusters.**

![DBI CHI](https://github.com/arora767287/email_semantic_cluster/assets/82481744/fe7d5b7b-8bd0-47b4-838c-8810015e8ad3)



For the model 1 pipeline, we were tasked with finding the best number of clusters to evaluate our model upon. As such, we plotted an elbow curve across k=2 to k=50 (inclusive) used as the number of clusters we ran KMeans on the output of our PCA for. The elbow curve plotted above shows that the best value for the number of clusters occurs at k=9, where the plot of within cluster squared sums reaches a point after which decreases in the WCSS (a metric of the variance within the points of each cluster) are not substantial for increases in the number of clusters. Using this clustering of k=9 clusters, the Davies-Bouldin Index calculated is 6.04, which means that the average distance between document representations in each cluster was greater than the distance between the different identified 9 “clusters” or topics. This presents a significant amount of variance within each grouping of emails, indicative of poor separation boundaries between the emails.

Upon deeper analysis with our model 2 pipeline, the Davies-Bouldin Index is 0.39, which showcases better clustering performance. This lower index indicates improved separation between clusters compared to model 1. Although caution is warranted in direct comparison due to the different scales of Davies-Bouldin Index values, the model displays more distinct topics. 

In addition, the Calinski Harabasz index for the model 1 pipeline was 1479.23, which is a ratio of the between cluster sum of squares to the within cluster sum of squares, with larger indices being significantly better, presenting the k=9 clustering as a good candidate. The Calinski-Harabasz Index for model 2 yields a score of 1,630,166.54. This substantial improvement over model 1's index underscores the effectiveness of our model 2 approach, suggesting better-defined clusters and enhanced separation between them.


# Next Steps
Going forward we hope to do one more model pipeline for topic modeling on Emails. In addition, we hope to broaden the metrics we use to include the topic coherence score to gain additional valuable insight into the clusters we’re creating. One stretch goal we have for the final submission is to use investigative techniques to determine what the model pipelines’ clusters topics are and whether or not they truly represent something meaningful that a user would want to see.

# Contribution Table For Proposal
| Member | Job | Description |
| --- | --- | --- |
| Katherine & Samarth | Model 2 | Preprocessing, Creation, Training, Metrics and Analysis of Model 2 |
| Tanush, Nitya | Model 1 | Preprocessing, Creation, Training, Metrics and Analysis of Model 1 |
| Katherine & Ajay | Visualization | Visualizations of Data and Models |
| All | GitHub Pages | Making and formatting of GitHub pages website |

[Gantt Chart](https://docs.google.com/spreadsheets/d/1ZUl8Xywp4VTTNtC-8Wq8ZxpYnzXYNJLe/edit?usp=sharing&ouid=101698207149759013919&rtpof=true&sd=true)
---
# References
- Schopf, T., Braun, D., Matthes, F. (2023). Semantic Label Representations with Lbl2Vec: A Similarity-Based Approach for Unsupervised Text Classification. In: Marchiori, M., Domínguez Mayo, F.J., Filipe, J. (eds) Web Information Systems and Technologies. WEBIST WEBIST 2020 2021. Lecture Notes in Business Information Processing, vol 469. Springer, Cham. https://doi.org/10.1007/978-3-031-24197-0_4
- Quoc Le, Tomas Mikolov Proceedings of the 31st International Conference on Machine Learning, PMLR 32(2):1188-1196, 2014.
- Schopf, T., Braun, D., Matthes, F. (2023). Semantic Label Representations with Lbl2Vec: A Similarity-Based Approach for Unsupervised Text Classification. In: Marchiori, M., Domínguez Mayo, F.J., Filipe, J. (eds) Web Information Systems and Technologies. WEBIST WEBIST 2020 2021. Lecture Notes in Business Information Processing, vol 469. Springer, Cham. https://doi.org/10.1007/978-3-031-24197-0_4
- Li, Wei & Mccallum, Andrew. (2006). Pachinko allocation: DAG-structured mixture models of topic correlations. 577-584. 10.1145/1143844.1143917. 
- Sharaff, A., & Nagwani, N. K. (2016). Email thread identification using latent Dirichlet allocation and non-negative matrix factorization based clustering techniques. Journal of Information Science, 42(2), 200-212. https://doi.org/10.1177/0165551515587854
- Klimt, B., Yang, Y. (2004). The Enron Corpus: A New Dataset for Email Classification Research. In: Boulicaut, JF., Esposito, F., Giannotti, F., Pedreschi, D. (eds) Machine Learning: ECML 2004. ECML 2004. Lecture Notes in Computer Science(), vol 3201. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-30115-8_22
- [Topic Modeling Coherence Score](https://www.baeldung.com/cs/topic-modeling-coherence-score)
- Saitta, S., Raphael, B. and Smith, I.F.C. "A comprehensive validity index for clustering", Intelligent Data Analysis, vol. 12, no 6, 2008, pp. 529-548 http://content.iospress.com/articles/intelligent-data-analysis/ida00346  Copyright IOS Press
- D. L. Davies and D. W. Bouldin, "A Cluster Separation Measure," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PAMI-1, no. 2, pp. 224-227, April 1979, doi: 10.1109/TPAMI.1979.4766909.
- [The 2023 Spam Report](https://www.orbitmedia.com/blog/spam-statistics/#:~:text=56.5%25%20of%20all%20email%20is%20spam&text=But%20not%20all%20of%20that,making%20it%20past%20the%20filters.)
