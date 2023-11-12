# Topic Modeling with Emails

## Introduction
The task of sorting through emails is a tedious and time-consuming activity that impacts numerous users daily. With the staggering statistic that 56.5 percent of all emails in 2022 were categorized as spam, the demand for efficient and intelligent email filtering systems is higher than ever. Despite existing measures, current systems still fall short, frequently permitting spam and irrelevant messages to infiltrate inboxes. This inefficiency not only leads to a loss of productivity as users sift through unwanted emails, but it also increases the likelihood of overlooking crucial correspondence. Our project is dedicated to developing an email clustering system, leveraging machine learning to organize emails into user-defined topics, ensuring that recipients are presented with emails that align with their interests and priorities.

## Dataset (Checkpoint)
As we’ll need an extensive training dataset of emails to facilitate the above methods, we plan to use the [Enron Dataset](https://www.cs.cmu.edu/~enron/), which contains emails of about 150 users and has been compiled with 500,000 messages. This dataset contains information about sender, receiver, timestamp when it was sent, subject and body of the email.

## Problem Definition
Email serves as a pivotal communication hub for a wide array of individuals including researchers, students, and professionals. The ubiquity of its use, however, renders it susceptible to congestion and disarray due to the volume of incoming messages that vary in significance. Such disorder can result in vital communications being overshadowed by those of lesser relevance, potentially leaving pressing issues unaddressed. This leads to disorganization and a decline in efficiency for countless users. Given the extensive reliance on email for daily exchanges, the ramifications of this disorganization are particularly concerning, highlighting the critical need for more refined management solutions.

## Methods
In this midterm report, we have implemented and compared two distinct models designed to organize emails into coherent clusters, facilitating easier navigation and management for users based on their topical preferences.

### Model 1: Doc2Vec+K-means

Given the raw text data from each of the emails of the processed Enron dataset, we decided to sample 20% of the total emails due to computational complexity related to Doc2Vec. Note one thing that should be mentioned is that this sampling is different from Model 2's (which sampled 80% of the total emails). The reason for this was that we ran into issues related to training Doc2Vec taking far too long to train to convergence if we used all the emails. One thing that should be noted however is that as we sample 20% which is greater than 100 / 10 =  10% of the total population of the enron dataset (by the 10% condition) this sample is generalizable to the complete population.

Using this subset of the data, we trained an unsupervised Doc2Vec model to extract a semantically significant 256-dimensional vector of each document. Doc2Vec is an extension of the Word2Vec model, which increases the scope to include document-level context, rather than just word-level. This approach allows the model to learn not only the significance of individual words, but also how these words come together to convey meaning in a larger text. This model was trained for 5 epochs to achieve convergence for the meaning in these vectors.

Given these semantically significant embeddings, we ran PCA, an unsupervised dimensionality reduction algorithm, to “narrow” and “focus” the embeddings on the most important semantic differences between the document. This allowed for clearer and more significant clustering in the last step (k-means) while also keeping the computational complexity of the model pipeline as a whole under control.

These reduced embeddings were then passed into the k-means algorithm in order to cluster the documents into “topics.” The k-means algorithm is a centroid-based clustering method that partitions the data into K distinct, non-overlapping subgroups, or clusters. This model is straightforward and allows for quick retrieval of similar emails, although it requires a careful choice of 'k' will significantly impact the granularity of the clustering, requiring careful tuning to strike a balance between overgeneralization and fragmentation of topics.

### Model 2: TF-IDF+LDA

The second model takes a different approach by first applying the Term Frequency-Inverse Document Frequency (TF-IDF) technique. This method transforms the text data into a sparse matrix where each email is represented by a vector indicating the frequency of terms weighted by their importance across the corpus. Given the high dimensionality of the resulting vectors, we then perform PCA dimensionality reduction using the TruncatedSVD tehcnique in scit-kit learn. This seeks to reduce the feature space while maintaining as much variance as possible. 

Subsequently, Latent Semantic Analysis (LSA) is applied, which further reduces dimensions and identifies latent topics by grouping together terms that co-occur across the corpus, thereby uncovering underlying thematic structures. This model is particularly powerful for large datasets, capable of extracting and clustering emails based on the nuanced themes that may not be immediately apparent through direct term comparisons. However, the transformations applied during PCA and LSA may result in a loss of interpretability and require a more complex pipeline to process and cluster the emails effectively.

## Potential Results and Discussion
We plan to use Word2Vec-based topic coherence metrics to score how well reference topics match clustered emails. The Calinski-Harabasz Index will measure the difference between Word2Vec embeddings for each topic and the average embeddings of all other clusters. This will provide a ratio of intra-cluster to inter-cluster variations based on topic names, gauging label accuracy. Lastly, the Davies-Bouldin Index will assess the cosine similarity between a cluster and its topic compared to the next closest cluster, indicating the topic's distinctiveness from its nearest neighbor.

# Contribution Table For Proposal
| Member | Job | Description |
| --- | --- | --- |
| Katherine | Video | Recording and editing of video |
| Tanush, Nitya & Katherine | Slides | Presentation visuals for video |
| Ajay | Introduction & Background | Introduction and background for proposal |
| Nitya | Problem Description | Description of problem for proposal |
| Nitya, Tanush & Ajay | Methods | Methods for proposal |
| Tanush & Nitya | Potential Results and Discussion | Metrics for proposal |
| All | Literature Review | Review of general methods for topic modeling |
| All | GitHub Pages | Making and formatting of GitHub pages website |
| Tanush | Gantt Chart | Formatting and creation of Gantt Chart |

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
