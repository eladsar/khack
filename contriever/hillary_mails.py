
from contriever_utils import mail_raw_text2paragraphs, get_contriever_embedding, calc_contriever_score
import pandas as pd
from tqdm import trange, tqdm

emails_path = '/home/hackathon_2023/data/hillary_clinton_emails/data/Emails.csv'
hillary_paragraphs_path = '/home/hackathon_2023/ron/hillary_paragraph_embedded.fea'

queries = ['Benghazi attack']

class HillaryEmails:
    def __init__(self, src_path=None):
        if src_path is None:
            src_path = emails_path
        self.src_path = src_path
        self.emails_raw = pd.read_csv(emails_path)
        self.emails_paragraphs = None

    def get_paragraphs(self):
        self.emails_paragraphs = self.emails_raw[['DocNumber', 'RawText']].copy()
        self.emails_paragraphs['paragraphs'] = self.emails_paragraphs['RawText'].map(mail_raw_text2paragraphs)
        self.emails_paragraphs = self.emails_paragraphs.explode('paragraphs')
        self.emails_paragraphs = self.emails_paragraphs.dropna()
        return self.emails_paragraphs

    def add_embedding(self, batch_size=100):
        batch_size = 100
        embeddings = []
        for i in trange(0, len(self.emails_paragraphs), batch_size):
            embeddings += get_contriever_embedding(self.emails_paragraphs.iloc[i:i + batch_size]['paragraphs'])
        self.emails_paragraphs['embedding'] = embeddings

    def embedded_to_file(self, dst_path):
        if 'embedding' not in self.emails_paragraphs.columns:
            print('Missing embedding column')
            return
        self.emails_paragraphs.reset_index().to_feather(dst_path)

class HillaryQueryScore:
    def __init__(self, src_path=None):
        if src_path is None:
            src_path = hillary_paragraphs_path
        self.src_path = src_path
        self.paragraphs = pd.read_feather(src_path)
        if 'index' in self.paragraphs.columns:
            self.paragraphs = self.paragraphs.drop('index', axis=1)


    def get_query_scores(self, query):
        query_embedding = get_contriever_embedding([query])[0]
        contriever_scores = self.paragraphs['embedding'].map(lambda emb: calc_contriever_score(emb, query_embedding))
        contriever_scores_df = self.paragraphs.copy()
        contriever_scores_df['score'] = contriever_scores
        contriever_scores_df = contriever_scores_df.sort_values(by=['score'], ascending=False)
        return contriever_scores_df




