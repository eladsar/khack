from hillary_mails import HillaryQueryScore, HillaryEmails
from connect import OpenAI
from tqdm import tqdm


class Retrieve(object):
    def __init__(self):
        self.oi = OpenAI()
        self.scores = HillaryQueryScore()
        self.corpus = HillaryEmails()
        self.emails = self.corpus.emails_raw.set_index('DocNumber')

    def query(self, query, top_n=10):
        df = self.scores.get_query_scores(query).drop_duplicates(subset=['score']).head(top_n)

        titles = []
        body = self.emails.loc[df['DocNumber']]['RawText']
        for r in tqdm(body.iteritems(), total=len(body)):
            titles.append(self.oi.title(r[1][:3072]))

        df['title'] = titles

        return df