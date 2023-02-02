from flow import Flow
from pathlib import Path
import pandas as pd
from questions import QuestionsType, get_question_func
from questions import YesNoNextStep, MailEntitiesNextStep
from connect import OpenAI
import random
ROOT = '/home/hackathon_2023/'


def get_data(path=None, limit=-1):
    if not path:
        path = Path(ROOT) / 'data' / 'enron' / 'emails.parquet'
    df = pd.read_parquet(path=path, engine='pyarrow')
    return df

def add_question_func(list_of_questions):
    for q in list_of_questions:
        q['question_func'] = get_question_func(q=q)


def fake_dataset(df):
    qs1 = ['Can you change the accounting treatment on this transaction? We need to make the earnings look better.',
          "Let's make sure that these transactions don't appear on the books.",
          "Can we make this transaction look more favorable on the books?",
          "Let's make sure that we don't get caught for this accounting fraud.",
           "I don't want these emails to be seen by the authorities. Can you help me delete them?"
          ]
    qs2 = ["I think you will be interested in the stock movement of Enron.",
           "We need to sell all of our Enron stock before it's too late.",
           "I just found out some confidential information about Enron's financials."
           " Let's sell our stock before anyone else finds out.",
           "I just heard some rumors that the stock is going to drop soon."]

    qs3 = ["Can you help me destroy this email? I don't want anyone to see it.",
           "Let's make sure that this information doesn't get out. Can you help me keep it confidential?",
           "Can you delete these emails? I don't want them to be seen by anyone.",
           "Let's make sure that we don't leave any evidence behind. Can you help me destroy this information?"]

    for i in range(len(qs1)):
        df.at[i, "body"] = qs1[i]

    for i in range(6, 6+len(qs2)):
        df.at[i, "body"] = qs2[i-6]

    for i in range(11, 11+len(qs3)):
        df.at[i, "body"] = qs3[i-11]

def som_filtering(df):
    vip = ["kenneth lay", "jeffrey skilling", "andrew fastow"]
    # cond = (df['x_from'].str.contains('kenneth')) | (df['x_from'].str.contains('jeffrey')) |\
    #        (df['x_from'].str.contains('andrew')) | (df['x_to'].str.contains('kenneth')) |\
    #         (df['x_to'].str.contains('jeffrey')) | (df['x_to'].str.contains('andrew'))
    # df = df[cond]
    oi= OpenAI(api_key=Flow.get_token_from_path(Path(ROOT)))
    df_to_sum = df[df['body'].str.len()>2000]
    for i, val in df_to_sum.iterrows():
        oi.summary(val['body'], n_words=100)
    return df

if __name__ == '__main__':
    question1 = {"type": QuestionsType.YESNO, "question": 'Does the text suggest criminal activity?', 'next_step': YesNoNextStep}
    question2 = {"type": QuestionsType.YESNO, "question": 'Does the enron stock mentioned?', 'next_step': YesNoNextStep}
    question3 = {"type": QuestionsType.YESNO, "question": 'Does the text mention hiding or destroying documents?', 'next_step': YesNoNextStep}
    question4 = {"type": QuestionsType.MAIL_ENTITIES, 'next_step': MailEntitiesNextStep}
    questions = [question1, question2, question3, question4]
    add_question_func(questions)
    mails = get_data()
    # test_mail = som_filtering(mails)
    # raise Exception
    # test_mail = pd.read_parquet('/home/hackathon_2023/david/enron_filtered_10000_processed.parquet')
    test_mail = som_filtering(mails)
    # fake_dataset(test_mail)
    my_flow = Flow.from_questions("enron_investigation", questions)
    my_flow.build_flow()
    my_flow.start(test_mail)
