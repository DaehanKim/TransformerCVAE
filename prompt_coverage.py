from nltk.stem import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from rouge import Rouge
from typing import List
import string
import numpy as np
import re
from evaluate import load

STEMMER = snowball.SnowballStemmer("english")
DETOK = TreebankWordDetokenizer()
ENG_STOPWORDS = stopwords.words("english")
BERTSCORE = load("bertscore")

def syntactic_coverage(prompt_lst : List[str], story_lst : List[str]):
    '''story가 prompt 안에 있는 단어들로 얼마나 많이 구성되어 있는지 (rouge-1)
    prompt 안에 있는 내용들이 얼마나 story 안에 담겼는지 (bleu-1)를 활용한 F1 score 계산
    
    - stop words 제외
    - stemming 후 겹치는 단어 비율
    - TODO : synonym 반영하여 해당 단어가 있는지까지 확인

    e.g. The star shines (prompt) The bright star is shining through clouds. (story)
    2 단어(star, shine)가 겹치므로 recall coverage = 2/5 = 0.4
    pricision은 2/2 = 1.0
    따라서 f1은 0.8/1.4 = 0.57
    '''
    refined_prompt_lst = []
    refined_story_lst = []

    for prompt, story in zip(prompt_lst, story_lst):
        prompt_split = [STEMMER.stem(e) for e in word_tokenize(prompt) if e not in ENG_STOPWORDS]
        story_split = [STEMMER.stem(e) for e in word_tokenize(story) if e not in ENG_STOPWORDS]
        refined_prompt = DETOK.detokenize(prompt_split).translate(str.maketrans("","",string.punctuation))
        refined_story = DETOK.detokenize(story_split).translate(str.maketrans("","",string.punctuation))
        refined_story_lst.append(refined_story)
        refined_prompt_lst.append(refined_prompt)

    rouge = Rouge()
    scores = rouge.get_scores(refined_prompt_lst, refined_story_lst, avg=True)
    return scores['rouge-1']
    


def semantic_coverage(prompt_lst : List[str], story_lst : List[str]):
    '''bert score를 이용한 symentic coverage 계산'''
    # predictions = ["hello there", "general kenobi"]
    # references = ["hello there", "general kenobi"]
    results = BERTSCORE.compute(predictions=story_lst, references=prompt_lst, lang="en")
    prec = np.array(results['precision']).mean()
    rec = np.array(results['recall']).mean()
    f1 = 2*prec*rec/(prec+rec)
    return {"p" : prec, "r" : rec, "f" : f1}

def file_based_coverage(filename: str, method='sem'):
    '''cvae format에서 각 스토리 파싱해서 스코어 계산함'''
    reg = re.compile("[=]+ Outlines  [=]+|[=]+ Story [=]+|[=]+ SAMPLE \w [=]+|[=]+ Generated [=]+")
    with open(filename, "rt", encoding='utf8') as fin:
        split = re.split(reg, fin.read())
    split = [e.strip() for e in split if e.strip()] # remove empty strings
    split = [re.sub(r"<\|endoftext\|>|\[ \w+ \]","", e) for e in split] # remove <endoftext> token and WP specific prefix
    prompt_lst, true_story_lst, gen_story_lst = [], [], []
    # print(len(split))
    for i in range(0,len(split),3):
        prompt_lst.append(split[i])
        true_story_lst.append(split[i+1])
        gen_story_lst.append(split[i+2])
    if method == 'syn' :
        metric = syntactic_coverage
    elif method == 'sem' :
        metric = semantic_coverage
    else:
        raise NotImplementedError(f"method {method} is not supported!") 
    gen_story_relevance = metric(prompt_lst, gen_story_lst)
    true_story_relevance = metric(prompt_lst, true_story_lst)

    print("generated story score :", gen_story_relevance)
    print("true story score :", true_story_relevance)

if __name__ == "__main__":
    print("test")
    prompt = "Leonardo DiCaprio in a fit of rage begins to torpedo his own career by deliberately acting poorly and taking on bad films. He finally wins an oscar for starring in Paul Blart : Mall Cop 3."
    story = '''I have found myself at this point with the man I love. The very same man whose work in the art business brought him and his family joy and happiness and all kinds of love, and who now, it seems, has the audacity to run his own film company.

The truth is, he is one of the highest paid actors in the world, and yet he has been saddled with the same question for so long. Because he loves a movie for a reason, he has decided, what do you do with the money? What do you do with the talent that would become a career in the film industry in the first place? How can you take someone whose work has generated acclaim, who created films and has won awards, and give them something that they never had in a career?

The answer lies in his own greed. He is happy to keep those who have paid him this kind of money and to make it go on, but he fears that he would end up being a corrupt crony capitalist, and would get caught after winning Best Director Oscars, or taking on lesser projects. He fears, however, that he will be robbed of some of the joy that he once enjoyed, and that some of it will evaporate into oblivion. And this worries me deeply. That is, until I play, and the film I am making.

As for the plot, I will say that in the book, I am, essentially, trying to find a way to not go into too much detail with the story. I will say something more important, and more important than that, but I will just get out and do it. A story isn't a film unless you want to be inside a giant metal tube that will burst into life every time you enter it. And that's what I intend to do, in order to convince the world that I will be the way to go to see those things.

Well, this is the ending that finally opens up for me, and I have the choice to make it any way I choose to. I can either go into high finance, and take the same chance I take in entertainment, or I can go back to academia and take the same chance that the American people were willing to give me. I can do both. But I don't desire to be the way to go, for I have not been a successful performer, a successful director, a successful actor in a movie. I have no intention of becoming one in the entertainment world. I have not been very successful at a high level.

I don't plan on starting my career in the entertainment business. I have tried. I have gotten along with actors, singers, songwriters, actors, singers, directors, filmmakers, presidents, all of them have had success. And I think that if I can get my hand on this film, I can make it a success. So I will go through a long series of trials and tribulations, and come up with some good, good, good, good things. I will choose one that is better, one that is better, but one that will not be the film, and one that has a certain amount of heart and soul and wit that will satisfy the hunger of every poor soul in the world. 
 And for that, I will give. 
 This is my fourth try at a film, so thank you for the support.'''
    score = syntactic_coverage(prompt, story)
    print(score)