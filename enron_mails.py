
from contriever_utils import mail_raw_text2paragraphs, get_contriever_embedding, calc_contriever_score
import pandas as pd
from tqdm import trange, tqdm

emails_path = '/home/hackathon_2023/data/hillary_clinton_emails/data/Emails.csv'
enron_paragraphs_path = '/home/hackathon_2023/ron/enron_paragraph_embedded.fea'

queries = ['Benghazi attack']


class HillaryEmails:
    def __init__(self, src_path=None):
        if src_path is None:
            src_path = emails_path
        self.src_path = src_path
        self.emails_raw = pd.read_csv(emails_path)
        self.emails_paragraphs = None