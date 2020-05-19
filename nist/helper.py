import os
from scipy.sparse import csr_matrix


def read_qrels(path):
    qrels = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            tokens = line.split(' ')
            topic = int(tokens[0])
            doc_id = tokens[-2]
            relevance = int(tokens[-1])
            qrels.append({
                'topic': topic,
                'doc_id': doc_id,
                'relevance': relevance
            })

    return qrels


def get_qrels_topics(path):
    qrels = read_qrels(path)
    topics = set()
    for pack in qrels:
        topics.add(pack['topic'])

    return topics


def normalize(scores):
    low = min(scores)
    high = max(scores)
    width = high - low

    return [(s-low)/width for s in scores]


def get_X_Y_from_qrels_by_topic(path, topic, R):
    # always include topic 0
    R.append(0)
    qrels = [qrel for qrel in read_qrels(path) if qrel['topic'] == topic]
    qrels = [qrel for qrel in read_qrels(path) if qrel['relevance'] in R]
    x, y = [], []
    for pack in qrels:
        if pack['topic'] == topic:
            x.append(pack['doc_id'])
            label = 0 if pack['relevance'] == 0 else 1
            y.append(label)

    return x, y


def get_docs_from_qrun_by_topic(path, topic):
    x, y = [], []
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            t = int(tokens[0])
            if topic != t:
                continue
            doc_id = tokens[2]
            score = float(tokens[-2])
            x.append(doc_id)
            y.append(score)

    return x, y


def get_doc_ids_from_qrels_by_topic(path, topic):
    qrels = read_qrels(path)
    return [pack['doc_id'] for pack in qrels if pack['topic'] == topic]


def sort_dual_list(pred, docs):
    zipped_lists = zip(pred, docs)
    sorted_pairs = sorted(zipped_lists)

    tuples = zip(*sorted_pairs)
    pred, docs = [list(tuple) for tuple in tuples]

    pred.reverse()
    docs.reverse()
    return pred, docs


specter = None


def get_specter_by_doc_id(doc_id):
    global specter
    path = 'nist/data/specter.csv'
    if os.path.exists(path) is False:
        os.system(
            'wget -nc https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/cord_19_embeddings.tar.gz')
        os.system(
            'tar -xvzf cord_19_embeddings.tar.gz -C nist/data/ --strip-components 1')
        os.system(f'mv nist/data/cord_19_embeddings*.csv {path}')

    if specter is None:
        specter = {}
        print('Reading SPECTER embeddings')
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                tokens = line.split(',')
                cord_uid = tokens[0]
                specter[cord_uid] = tokens[1:]

    if doc_id in specter:
        return specter[doc_id]

    raise Exception(f"SPECTER not found {doc_id}")


def get_specter_by_doc_ids(docs):
    res = [get_specter_by_doc_id(doc)for doc in docs]
    return csr_matrix(res)


if __name__ == '__main__':
    pass
    # path = 'nist/data/qrels_test.txt'
    # qrels = read_qrels(path)
    # with open('nist/data/qrels_test_sorted.txt', 'w+') as f:
    #     for topic in get_qrels_topics(path):
    #         for rele in [2,1,0]:
    #             for qrel in qrels:
    #                 t = qrel['topic']
    #                 r = qrel['relevance']
    #                 doc_id = qrel['doc_id']
    #                 if t != topic or r != rele:
    #                     continue
    #                 f.write(f't{t} {doc_id} {r}\n')
