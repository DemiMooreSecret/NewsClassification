import torch
import pandas as pd
import numpy as np

from tqdm import tqdm

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import pymorphy2

import re, hashlib, math, time
from random import randint, seed
seed(1631996)


class hashFamily:
    def __init__(self, i):
        self.resultSize = 8 # размер выходного хэша
        self.maxLen = 20 # how long can our i be (in decimal) максимальное длина
        self.salt = str(i).zfill(self.maxLen)[-self.maxLen:] # добавка нулей в конец

    def get_hash_value(self, el_to_hash):
        return int(hashlib.sha1(str(el_to_hash).encode('utf-8') + self.salt.encode('utf-8')).hexdigest()[-self.resultSize:], 16)


class shingler:
    def __init__(self, k):

        if k > 0:
            self.k = int(k)
        else:
            self.k = 10

    #внутренние классы для обработки документа
    def process_doc(self, document):
        return re.sub("( )+|(\n)+"," ",document).lower()

    def get_shingles(self, document):
        shingles = set()
        document= self.process_doc(document)
        for i in range(0, len(document)-self.k+1 ):
            shingles.add(document[i:i+self.k])
        return shingles

    def get_k(self):
        return self.k

    #возвращает отсортированные хэши для синглеров
    def get_hashed_shingles(self, shingles_set):
        hash_function = hashFamily(0)
        return sorted( {hash_function.get_hash_value(s) for s in shingles_set} )


class minhashSigner:
    def __init__(self, sig_size):
        self.sig_size=sig_size
        self.hash_functions = [hashFamily(randint(0,10000000000)) for i in range(0,sig_size)]

    def compute_set_signature(self, set_):
        set_sig = []
        for h_funct in self.hash_functions:
            min_hash = math.inf
            for el in set_:
                h = h_funct.get_hash_value(el)
                if h < min_hash:
                    min_hash = h

            set_sig.append(min_hash)

        return set_sig

    #return a list of lists that can be seen as the signature matrix
    def compute_signature_matrix(self, set_list):
        signatures = []
        for s in set_list:
            
            signatures.append( self.compute_set_signature(s) )

        return signatures


class lsh:
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def get_signature_matrix_bands(self, sig_matrix, bands_nr, sign_len):
        r = int(sign_len/bands_nr)
        bands = {}
        for i in range(0,bands_nr):
            bands[i] = []

        for signature in sig_matrix:

            for i in range(0, bands_nr):
                idx = i*r
                bands[i].append(' '.join(str(x) for x in signature[idx:idx+r]) )

        return bands

    def get_band_buckets(self, band, hash_funct):
        buckets = {}
        for doc_id in range(0,len(band)):
            value = hash_funct.get_hash_value( band[doc_id] )
            if value not in buckets:
                buckets[value] = [doc_id]
            else:
                 buckets[value].append(doc_id)

        return buckets

    def get_candidates_list(self, buckets):
        candidates = set()
        for bucket,candidate_list in buckets.items():
            if len(candidate_list) > 1:
                for i in range(0,len(candidate_list)-1):
                    for j in range(i+1,len(candidate_list)):
                        pair = tuple(sorted( (candidate_list[i],candidate_list[j]) ))
                        candidates.add(pair)

        return candidates

    def check_candidates(self, candidates_list, threshold, sigs):
        similar_docs = set()
       
        for  similar_pair in candidates_list:
            doc_id_1 = similar_pair[0]
            doc_id_2 = similar_pair[1]
            signature_1 = set(sigs[doc_id_1])
            signature_2 = set(sigs[doc_id_2])
            js = len(signature_1.intersection(signature_2)) /len(signature_1.union(signature_2))

            if js >= threshold:
                similar_docs.add( tuple(sorted((doc_id_1,doc_id_2) )) )


        return similar_docs

    def get_similar_items(self, sig_matrix, bands_nr, sign_len):
        similar_docs = set()
        bands = self.get_signature_matrix_bands(sig_matrix,bands_nr,sign_len)

        for band_id, elements in bands.items():
            buckets = self.get_band_buckets(elements, hash_funct=hashFamily(randint(0,10000000000)))
            
            candidates = self.get_candidates_list(buckets)
            
            for sim_tuple in self.check_candidates(candidates, self.threshold, sig_matrix):
                similar_docs.add( sim_tuple)

        return similar_docs


class DemiMooreDuplicates:
    def __init__(self):
        pass
    def predict(self, data):
        data['nn_text'] = data['text']
        data['nn_text'] = data['nn_text'].fillna('пустая новость')
        data['nn_text'] = data['nn_text'].apply(lambda x: x if isinstance(x, str) else 'пустая новость')

        data['nn_text'] = data['nn_text'].apply(lambda text : re.sub(r'\s+', ' ', text)) #удаляет уникоды 

        data['nn_text'] = data['nn_text'].apply(lambda text : ' '.join(text.split("'"))) # удаляет апострофы

        data['nn_text'] = data['nn_text'].apply(lambda text : re.sub(r'\\[a-z0-9]+', ' ', text)) #удаляет уникоды с двойным обратным слешом


        data['nn_text'] = data['nn_text'].apply(lambda text : re.sub(r'#[\w]+','',text))
        data['nn_text'] = data['nn_text'].apply(lambda text : re.sub(r'@[\w]+','',text))


        data['nn_text'] = data['nn_text'].apply(lambda text : re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text))

        data['nn_text'] = data['nn_text'].apply(lambda text : re.sub('\W+',' ', text))
        data['nn_text'] = data['nn_text'].apply(lambda text : re.sub(' +',' ', text))

        data['nn_text'] = data['nn_text'].apply(lambda text : text.lower())

        stop_words_rus = list(stopwords.words('russian'))
        stop_words_eng = list(stopwords.words('english'))
        remove_stopwords = stop_words_rus+stop_words_eng

        data['nn_text'] = data['nn_text'].apply(lambda text: ' '.join(list(filter(lambda x: x not in remove_stopwords, text.split()))))

        morph = pymorphy2.MorphAnalyzer()
        data['nn_text'] = data['nn_text'].apply(lambda text: morph.parse(text)[0][0])
        
        data['nn_text'] = data['nn_text'].astype(str)
        data['nn_text'] = data['nn_text'].apply(lambda text: 'пустая новость' if not len(text) else text)
        
        data = data.drop_duplicates(subset=['text'])
        
        data['id'] = range(data.shape[0])

        data['doc_id']=data.index
        doc_nr = data.shape[0]
        shingling_list = [None] * doc_nr
        shingling_size = 10
        signature_size = 50
        bands_nr = 10

        shingler_inst = shingler(shingling_size)
        signer = minhashSigner(signature_size)

        for index, row in data.iterrows():
            doc = row['text']
            i = row['id']
            shingler_set = shingler_inst.get_shingles(doc)
            shinglings = shingler_inst.get_hashed_shingles(shingler_set)
            shingling_list[i] = shinglings

        print("Computing signature matrix...")
        #produce a signature for each shingle set
        signature_matrix = signer.compute_signature_matrix( shingling_list )
        
        lsh_instance = lsh(threshold=0.8)
        print("Computing LSH similarity...")
        lsh_similar_itemset = lsh_instance.get_similar_items(signature_matrix, bands_nr, signature_size)


        data['list_similar'] = 0
        
        
        num_similar = 1
        for i in lsh_similar_itemset:
            if data.at[i[0], 'list_similar'] == 0 and data.at[i[1], 'list_similar'] == 0:
                data.at[i[0], 'list_similar'] = num_similar
                data.at[i[1], 'list_similar'] = num_similar
                num_similar +=1

            elif data.at[i[0], 'list_similar'] == 0 and data.at[i[1], 'list_similar'] != 0:
                num_old = data.at[i[1], 'list_similar']
                data.at[i[0], 'list_similar'] = num_old
            elif data.at[i[0], 'list_similar'] != 0 and data.at[i[1], 'list_similar'] == 0:
                num_old = data.at[i[0], 'list_similar']
                data.at[i[1], 'list_similar'] = num_old
            elif data.at[i[0], 'list_similar'] < data.at[i[1], 'list_similar']:
                num_old = data.at[i[0], 'list_similar']
                data.at[i[1], 'list_similar'] = num_old
            elif data.at[i[0], 'list_similar'] > data.at[i[1], 'list_similar']:
                num_old = data.at[i[1], 'list_similar']
                data.at[i[0], 'list_similar'] = num_old
        
        return pd.concat([data.query('list_similar==0'), data.groupby('list_similar').max()[1:]])

import torch
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import re
import pandas as pd
import numpy as np


class NewsClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(312, 128)
        self.linear2 = torch.nn.Linear(128, 128)
        self.linear3 = torch.nn.Linear(128, 128)
        self.linear4 = torch.nn.Linear(128, 29)
        self.sigmoid = torch.nn.SiLU()
        self.sm = torch.nn.Softmax()
        
    
    def forward(self, X):
        X = self.sigmoid(self.linear1(X))
        X = self.sigmoid(self.linear2(X))
        X = self.sigmoid(self.linear3(X))
        return 20*self.sm(self.linear4(X))


class DemiMooreClassification:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        self.model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
        self.nn_model = NewsClassifier()
        self.nn_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    def predict(self, data):
        """
        data : pd.DataFrame
        return : pd.DataFrame
        """

        sentences = list(data['nn_text'])
        result = torch.FloatTensor(0, 312)

        for sent in DataLoader(sentences, batch_size=10):
            t = self.tokenizer(sent, padding=True, truncation=True, max_length = 512, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**{k: v for k, v in t.items()})
            embeddings = model_output.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings)
            result = torch.cat([result, embeddings.cpu()])
        
        data['category'] = (torch.argmax(self.nn_model(result), axis=1)+1).cpu().numpy()
        categ_names = {1: 'Финансы', 2: 'Технологии', 3: 'Политика', 4: 'Шоубиз',
                       5:'Мода и красота', 6:'Криптовалюты', 7:'Путешествия',
                       8:'Образование и познавательное', 9:'Развлечения и Юмор',
                       10:'Другое', 11: 'Блоги', 12: 'Новости и СМИ', 13: 'Экономика', 
                       14:'Бизнес и стартапы', 15:'Маркетинг, PR, реклама', 16:'Психология',
                       17:'Дизайн', 18: 'Искусство', 19: 'Право', 20:'Спорт', 21: 'Здоровье и медицина',
                       22: 'Картинки и фото', 23: 'Софт и приложения', 24:'Видео и фильмы', 25:'Музыка',
                       26:'Игры', 27:'Еда и кулинария', 28:'Цитаты', 29:'Рукоделие'}
        data['category'] = data['category'].map(categ_names)
        return data[['text', 'channel_id', 'category']]



def printApp(app, msg):
    app.status += msg + "\n"

def categories(file_path, app_main):
    
    #---------------------------------
    #Дубликаты
    #---------------------------------

    #Иннициализация
    dm = DemiMooreDuplicates()
    #Предсказание
    df = pd.read_csv(file_path)
    res = dm.predict(df)

    #---------------------------------
    #Классификация
    #---------------------------------

    # Иннициализация
    total_model = DemiMooreClassification('demi-moore-model.pth')
    # Предсказание
    df = total_model.predict(res)
    printApp(app_main, "Completed!")
    return df