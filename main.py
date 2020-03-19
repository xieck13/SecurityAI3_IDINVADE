# 加载Python自带 或通过pip安装的模块
import jieba
import json

# 加载用户自己的模块
#from example_module import foo

# ----------------------------------------
# # 本地调试时使用的路径配置
inp_path = 'benchmark_texts.txt'
out_path = 'adversarial.txt'
# ----------------------------------------

# ----------------------------------------
# 提交时使用的路径配置（提交时请激活）
# inp_path = '/tcdata/benchmark_texts.txt'
# out_path = 'adversarial.txt'
# ----------------------------------------

print('here is 2020 3 6 12.37')


# ----------------------------------------
# function of self defination
def tokenize(text):
    return ' '.join(jieba.cut(text))
#-----------------------------------------

#-----------------------------------------
import fasttext

def reference_model(model, test_args):
    """
    调用参考模型; 需要fasttext
    """
    if test_args:
        y_label, Fy = model.predict(test_args)
        if(y_label[0] == '__label__0'):
            Fy[0] = 1-Fy[0]
        return float(Fy[0])
    else:
        return 0
#----------------------------------------

#----------------------------------------
with open('./stopwords/中文停用词表.txt','r',encoding='UTF-8') as c:
    stop_words_chinese = c.readlines()
with open('./stopwords/哈工大停用词表.txt', 'r',encoding='UTF-8') as c:
    stop_words_hit = c.readlines()
with open('./stopwords/四川大学机器智能实验室停用词库.txt', 'r',encoding='UTF-8') as c:
    stop_words_scu = c.readlines()
with open('./stopwords/百度停用词表.txt', 'r',encoding='UTF-8') as c:
    stop_words_baidu = c.readlines()


stop_words = []

set_stop_words = [stop_words_chinese, stop_words_hit, stop_words_scu, stop_words_baidu]

for m_stop in set_stop_words:
    for m_s in m_stop:
        m_s = m_s.strip()
        if m_s not in stop_words:
            stop_words.append(m_s)
# print((stop_words))
#-----------------------------------------


#----------------------------------------
#tfidf score of word
#-----------------------------------------



def tfidf_score_of_word(corpus):

    corpus = [tokenize(line) for line in corpus]

    import numpy as np 
    from sklearn import feature_extraction
    from sklearn.feature_extraction.text import CountVectorizer 
    from sklearn.feature_extraction.text import TfidfTransformer


    vectorizer = CountVectorizer(token_pattern='[\u4e00-\u9fa5_a-zA-Z0-9]{1,}')
    X = vectorizer.fit_transform(corpus)
    word_features = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    tfidf_standard = tfidf/np.max(tfidf)


    tf_idf_score = []
    for i, sen in enumerate(corpus):
        word_list = sen.split(' ')
        temp = []
        for word in word_list:
            if(word in word_features):
                temp.append(tfidf_standard[i,word_features.index(word)])
            else:
                temp.append(0)
        tf_idf_score.append(temp)

    return tf_idf_score

# tfidf_score_of_word(corpus)

#-----------------------------------------
#important word

with open('./stopwords/important_words.txt', 'r', encoding='utf-8') as c:
    important_words = c.readlines()[0].strip().split()
#     print(important_words)

#-----------------------------------------

#----------------------------------------
#-----------------------------------------

def calculate_importance_of_word(model, Fy_all, list_of_word, i, weight=0.5):
    temp = list_of_word.copy()

    # if temp[i] in important_words:
    #     return 1

    # if temp[i] in stop_words:
    #     # print(temp[i])
    #     return 0


    if i == 0:
        forward_score = 0
        backward_score = 0
    else:
        forward_score = reference_model(model, ''.join(temp[0:i+1])) - reference_model(model, ''.join(temp[0:i]))
        backward_score = reference_model(model, ''.join(temp[i:])) - reference_model(model, ''.join(temp[i+1:]))
    temp.pop(i)
    delete_score = Fy_all - reference_model(model, ''.join(temp))
    return delete_score*weight + forward_score*(1-weight)/2 + backward_score*(1-weight)/2
    
    
model_path = 'reference_model/mini.ftz'
model = fasttext.load_model(model_path)
# caculate the importance of word to the sentence
def importance(list_of_texts, tf_idf_score):
    tokenized_text = tokenize(list_of_texts)
    list_of_word = tokenized_text.strip().split(' ')
    scores = []
    Fy_all = reference_model(model, [''.join(list_of_word)])
    # print('label------', Fy_all, '----sentence----- ', list_of_word)
    for i in range(len(list_of_word)):
        if len(list_of_word) == 1:
            scores.append(Fy_all)
            continue
        else:
            scores.append(calculate_importance_of_word(model, Fy_all, list_of_word, i)*0.8+tf_idf_score[i]*0.2)
    return scores
#-----------------------------------------

#-----------------------------------------
#caculate similarity between string a and
# string b
#-----------------------------------------


from distance_module import DistanceCalculator


g_dc = DistanceCalculator() # avoid loading WORD2vec many times


def distance_measure(dc ,test_args=(['你好呀'], ['你好呀'])):
    """
    调用距离计算器; 需要gensim, numpy
    """
    similarity_dic = dc(*test_args)

    score_levenshtein = 3/14.0 * ( 1 -  similarity_dic['normalized_levenshtein'][0])
    score_jaccard_word = 1/7.0 * (1 - similarity_dic['jaccard_word'][0])
    score_jaccard_char = 3/14.0 * (1 - similarity_dic['jaccard_char'][0])
    score_embedding_cosine = 3/7.0 * (1 - similarity_dic['embedding_cosine'][0])
    score_similarity = score_levenshtein + score_jaccard_word + score_jaccard_char + score_embedding_cosine

    # print('----Distance measure----')
    # print(*test_args, score_similarity, similarity_dic)
    return score_similarity

#-----------------------------------------

#-----------------------------------------
# use greedy algorithm
#-----------------------------------------
import copy
def calculate_similarity(target_text, index, list_of_texts, hanzi_of_target_test, gedit_text, black_list_word):

    original_word_target = copy.deepcopy(target_text[index])
    m_destination_word = copy.deepcopy(target_text[index])
    m_similarity_max = 0
    for m_hanzi_target in hanzi_of_target_test:
        if (''.join(m_hanzi_target.path) in black_list_word) or ''.join(m_hanzi_target.path) == original_word_target:  # avoid changing the pinyin to the originnal word
            # print(' the two is same')
            continue
        target_text[index] = ''.join(m_hanzi_target.path)
        gedit_text = ''.join(target_text)
        test_args = ([gedit_text], [list_of_texts])
        m_similarity_socre = distance_measure(g_dc, test_args)
        if m_similarity_socre >= m_similarity_max and ''.join(m_hanzi_target.path):
            m_destination_word = ''.join(m_hanzi_target.path)
            m_similarity_max = m_similarity_socre
            # print('score ---------', m_similarity_socre)
    return m_destination_word



#-----------------------------------------
import heapq
from pypinyin import lazy_pinyin
from Pinyin2Hanzi import DefaultHmmParams
from Pinyin2Hanzi import viterbi
import random

def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def top_k_transform(importance_score, list_of_texts, porpotion, new_word_dictionary, black_list_word):

    hmmparams = DefaultHmmParams() # HMM pinyin2hanzi
    

    target_text = list_of_texts
    target_text = tokenize(target_text).split(' ')
    k = int(len(target_text)*porpotion) + 1
    top_k_score = heapq.nlargest(k, importance_score)
    top_k_score_index = [importance_score.index(score) for score in top_k_score]

    for index in top_k_score_index:
        # make a virables repsent modified list_of_text
        gedit_text = copy.deepcopy(list_of_texts)
        if(is_Chinese(target_text[index])):
            pinyin_of_target_text = lazy_pinyin(target_text[index])
            if pinyin_of_target_text == ['ni']:
                pinyin_of_target_text = random.choice([['li'], ['ni']])
            if pinyin_of_target_text == ['ta']:
                pinyin_of_target_text = random.choice([['ta'], ['te']])
            if pinyin_of_target_text == ['cao']:
                pinyin_of_target_text = random.choice([['ca'], ['cao']])
            if pinyin_of_target_text == ['ma']:
                pinyin_of_target_text = random.choice([['me'], ['ma']])
            if pinyin_of_target_text == ['si']:
                pinyin_of_target_text = random.choice([['shi'], ['si']])
            try:
                #pinyin to other Chinese
                hanzi_of_target_test = viterbi(hmm_params=hmmparams, observations=pinyin_of_target_text, path_num = 10)

                # choose a word randly
                # target_text[index] = ''.join(random.choice(hanzi_of_target_test).path)


                # caculate the similarity between original word and transferable word
                # use greedy algorithm
                m_destination_word = calculate_similarity(target_text, index, list_of_texts[i], hanzi_of_target_test, gedit_text, black_list_word)

                target_text[index] = m_destination_word
                list_of_texts = ''.join(target_text)
                # 加入新词字典
                temp = new_word_dictionary.get(m_destination_word,0)
                temp += 1
                # 如果这个新词已经出现了10次，那么把它加到黑名单里
                if(temp < 20):
                    new_word_dictionary[m_destination_word] = temp
                else:
                    new_word_dictionary.pop(m_destination_word)
                    black_list_word.append(m_destination_word)
            except:
                pass
        else:
            continue
    return list_of_texts
#---------------------------------------------------------




with open(inp_path, 'r', encoding='UTF-8') as f:
    inp_lines = f.readlines()


dict_word = {
    'ni': ['你', '祢', '妳', '你', '妮', '你', '鉨', '您', '你', 'ni', '伱'],
    # 'zi': ['仔', '籽', '秄', '耔', '釨', '子', '子'],
    # 'bi': ['比', '必', '毕', '鼻', '逼'],
    # 'biao': ['表', '裱', '俵'],
    # 'gou': ['苟', '枸'],
    # 'si':  ['死', '挂'],
    # 'cao': ['肏', '擦', '艹']
}

def transform(line, tf_idf_score, new_word_dictionary, black_list_word):
    """转换一行文本。

    :param line: 对抗攻击前的输入文本
    :type line: str
    :returns: str -- 对抗攻击后的输出文门
    """
    # 修改以下逻辑
    from preprocessing_module import preprocess_text

    preprocess_text(line)
    # 选择修改文本的比例
    a = random.choice([1, 0, 2, 5, 4])
    if a >= 6:
        return line
    hmmparams = DefaultHmmParams()  # HMM pinyin2hanzi


    #进行重要度排序，得出每个词的辱骂性质的分数
    imp_score = importance(line, tf_idf_score)


    #修改一定比例的词语， 当比例为0时，最低为一个
    out_line = top_k_transform(imp_score, line, 0, new_word_dictionary, black_list_word)
    out_line = "".join(out_line)
    out_line = out_line.replace('\n', '')
    m_line = tokenize(out_line)

    _list_m_line = []
    for _word in m_line:
        _list_m_line.append(_word)

    #将“你”这个字进行替换

    for i, m_word in enumerate(m_line):
        if m_word in important_words:
            hanzi_of_target_test = ''
            pinyin_of_target_text = lazy_pinyin(m_word)
            if pinyin_of_target_text == ['ni']:
                hanzi_of_target_test = dict_word['ni']
            else:
                continue
            m_destination_word = m_word
            # pinyin to other Chinese
            nums_circle = 0
            #选择一个汉字原始汉字不同且不在黑名单里
            while nums_circle <= 50:
                nums_circle += 1
                m_destination_word = random.choice(hanzi_of_target_test)
                if m_destination_word != m_word and m_destination_word not in black_list_word:
                    break
                else:
                    continue
            _list_m_line[i] = m_destination_word

            m_line = ''.join(_list_m_line)

            temp = new_word_dictionary.get(m_destination_word, 0)
            temp += 1
            # 如果这个新词已经出现了30次，那么把它加到黑名单里
            if (temp < 30):
                new_word_dictionary[m_destination_word] = temp
            else:
                new_word_dictionary.pop(m_destination_word)
                black_list_word.append(m_destination_word)
    out_line = m_line.split()
    out_line = ''.join(out_line)
    _line = out_line
    str_dot = ''

    #求出最起始比例
    _ori_pro = reference_model(model, _line)
    _nums = 0

    #在句子末尾加逗号，至多50个，（当前概率-原始概率）/原始概率>0.8时停止
    for i in range(50):
        _line += ','
        _nums += 1
        _pre_pro = reference_model(model, _line)
        if abs(_pre_pro - _ori_pro)/_ori_pro > 0.8:
            break
    out_line = _line + str_dot
    print('outline,', out_line)
    return out_line
import time

start = time.clock()

#当中是你的程序
from preprocessing_module import preprocess_text
benchmark_text = [preprocess_text(_line) for _line in inp_lines]



tf_idf_score = tfidf_score_of_word(benchmark_text)
new_word_dictionary = {}

out_lines = []
black_list_word = []

for i, sen in enumerate(benchmark_text):
    out_lines.append(transform(sen, tf_idf_score[i], new_word_dictionary, black_list_word))

# out_lines = [transform(sen, tf_idf_score, new_word_dictionary) for sen in benchmark_text]
with open('my.txt', 'w', encoding='utf-8') as w:
    for m_line in out_lines:
        w.writelines(m_line + '\n')

target = json.dumps({'text': out_lines}, ensure_ascii=False)

with open(out_path, 'w' ,encoding='UTF-8') as f:
    f.write(target)

elapsed = (time.clock() - start)
print("Time used:", elapsed)