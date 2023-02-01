
import numpy as np

from contriever.src.contriever import Contriever
from transformers import AutoTokenizer

contriever = Contriever.from_pretrained("facebook/contriever")
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever") #Load the associated tokenizer:

model = contriever

end_of_paragraphs_length_th = 60
minimum_paragraph_length = 35

meta_words = ['UNCLASSIFIED', 'U.S. Department of State', 'Case No. ', 'Doc No. ', 'Date: ',
              'STATE DEPT. - ', 'SUBJECT TO AGREEMENT ON', 'RELEASE IN', 'PART B6', 'For:',
              'From:', 'Sent:', 'To:', 'Subject:', 'Attachments:', 'B6', 'CONFIDENTIAL',
              'Original Message', 'Cc: ', 'cc:', 'Subject FW:', 'Importance:',
              'SENSITIVE SOURCE ', 'THE FOLLOWING ', 'Subject RV:']

def calc_contriever_score(emb, query_embedding):
    score = np.dot(emb, query_embedding) / (np.linalg.norm(emb) * np.linalg.norm(query_embedding))
    return score

def get_contriever_embedding(texts):
    texts = [text.replace('\n', ' ') for text in texts]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    embeddings = model(**inputs)
    return embeddings.detach().numpy().tolist()

def mail_raw_text2paragraphs(raw_text):
    clean_text = mail_raw_text_remove_meta(raw_text)
    lines = clean_text.split('\n')
    paragraphs = []
    paragraphs_lines = []
    for line in lines:
        paragraphs_lines.append(line)
        if len(line) < end_of_paragraphs_length_th:
            paragraphs.append(' '.join(paragraphs_lines))
            paragraphs_lines = []
    paragraphs = [parag for parag in paragraphs if len(parag) > minimum_paragraph_length]
    return paragraphs

def mail_raw_text_remove_meta(raw_text):
    lines = raw_text.split('\n')
    lines = [line for line in lines if not is_meta_words_in_line(line)]
    return '\n'.join(lines)

def is_meta_words_in_line(line):
    for word in meta_words:
        if word in line:
            return True
    return False



