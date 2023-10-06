# Topic Modeling with Emails

## Background
In the modern age of information about 56.5% of emails sent in 2022 were spam. This presents a problem for the average person since this detracts attention away from actual important mail. Our goal in this project is to create an email clustering system that accurately clusters emails based on specific topics a user may want to see in their inbox, using it for inbox filtration in the future.

## Problem Definition
Emails are a key communication tool for researchers, students, and professionals. Due to its widespread use, email inboxes often become cluttered and disorganized, causing vital messages to be overshadowed by less important ones. This leads to unresolved urgent matters and decreased productivity for many. The pervasive use of email amplifies these challenges.

## Methods
We plan to approach solving this problem using 3 main ML pipelines...
1. TF-IDF + PCA + Pachinko Allocation Model (PAM):
   - TF-IDF: It assesses the relevance of frequency of phrases and keywords across documents. By evaluating a phrase's significance across a document corpus, we can pinpoint crucial keywords within emails.
   - PCA: This technique reduces the dimensionality of the keywords extracted via TF-IDF, emphasizing those that yield substantial importance for clustering.
   - PAM: A hierarchical topic modeling approach for captures correlations between vectorized representations of emails for clustering them into topics.

2. Doc2Vec + Lbl2Vec:
   - Doc2Vec: An extension of word2vec, it allows for capturing the semantic essence of entire documents. Using this, each email is converted into fixed-size vector embeddings.
   - kNN Model: Post vectorization, emails are clustered where centroids represent semantic essence.
   - Lbl2Vec: This model deciphers the vector embeddings’ meaning by associating them with labels.

3. Non-Negative Matrix Factorization (NNMF):
   - BERT: A pre-trained neural model, BERT captures contextual relationships bi-directionally within texts. By employing it, we intend to vectorize the email content.
   - Document Term Matrix (DTM): The resultant vector embeddings are transformed into a DTM wherein each row represents an email and columns correspond to words.
   - NNMF: This algorithm's application facilitates clustering of the emails in the DTM by discernible topics.

## Dataset (Checkpoint)
As we’ll need an extensive training dataset of emails to facilitate the above methods, we plan to use the [Enron Dataset](https://www.cs.cmu.edu/~enron/), which contains emails of about 150 users and has been compiled with 500,000 messages. This dataset contains information about sender, receiver, timestamp when it was sent, subject and body of the email.

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
