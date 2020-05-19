import sys
sys.path.insert(0, './')

import argparse
from sklearn.naive_bayes import MultinomialNB
import helper
from pyserini.vectorizer import TfidfVectorizer
from pyserini.search import pysearch
import os



# path variables
train_txt_path = 'nist/data/qrels_train_dev.txt'
test_txt_path = 'nist/data/qrels_test.txt'
rodrigo_txt_path = 'nist/data/rodrigo_judged.txt'
lucene_index_path = 'nist/data/lucene-index-cord19-abstract-2020-05-01'

# get round 1 topics & detail
topics_dict = pysearch.get_topics('covid_round1')


def run(k, R, alpha):
    R_str = ''.join([str(i) for i in R])
    run_path = f'runs/tfidf.k{k}.R{R_str}.A{alpha}.txt'
    print('Outputing:', run_path)

    topics = helper.get_qrels_topics(train_txt_path)
    vectorizer = TfidfVectorizer(lucene_index_path, min_df=5)
    searcher = pysearch.SimpleSearcher(lucene_index_path) if k > 0 else None

    f = open(run_path, 'w+')
    for topic in topics:
        train_docs, train_labels = helper.get_X_Y_from_qrels_by_topic(
            train_txt_path, topic, R)
        print(f'[topic][{topic}] eligible train docs {len(train_docs)}')
        train_vectors = vectorizer.get_vectors(train_docs)

        # classifier training
        clf = MultinomialNB()
        clf.fit(train_vectors, train_labels)

        # search topic question
        test_docs, search_scores, docs, preds = [], [], [], []
        if k > 0:
            question = topics_dict[topic]['question']
            hits = searcher.search(question, k=k)
            eligible_test_docs = helper.get_doc_ids_from_qrels_by_topic(
                test_txt_path, topic)
            hits = [hit for hit in hits if hit.docid in eligible_test_docs]
            test_docs = [hit.docid for hit in hits]
            search_scores = helper.normalize([hit.score for hit in hits])
        else:
            test_docs, search_scores = helper.get_docs_from_qrun_by_topic(            rodrigo_txt_path, topic)
            search_scores = helper.normalize(search_scores)
            # test_docs = helper.get_doc_ids_from_qrels_by_topic(
            #     test_txt_path, topic)
            # search_scores = [0] * len(test_docs)

        # classifier inference
        print(f'[topic][{topic}] eligible test docs {len(test_docs)}')
        test_vectors = vectorizer.get_vectors(test_docs)

        rank_scores = clf.predict_proba(test_vectors)
        # Extract prob of label 1
        rank_scores = [row[1] for row in rank_scores]

        # interpolation
        assert len(test_docs) > 0 and len(search_scores) > 0 and len(
            rank_scores) > 0, "len(test_docs) == 0"
        assert len(test_docs) == len(search_scores) == len(
            rank_scores), "Dimension mismatch"
        preds = [a * alpha + b * (1-alpha)
                 for a, b in zip(rank_scores, search_scores)]
        preds, docs = helper.sort_dual_list(preds, test_docs)

        for index, (score, doc_id) in enumerate(zip(preds, docs)):
            rank = index + 1
            f.write(f'{topic} Q0 {doc_id} {rank} {score} tfidf\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='use tfidf vectorizer on cord-19 dataset with ccrf technique')
    parser.add_argument('--k', type=int, required=True,
                        help='depth of search')
    parser.add_argument('--R', type=int, required=True,
                        nargs='+', help='Use docs with relevance 1s and/or 2s')
    parser.add_argument('--alpha', type=float,
                        help='alpha value for interpolation')
    args = parser.parse_args()

    R = sorted([r for r in args.R if r == 1 or r == 2])
    run(args.k, R, args.alpha)
