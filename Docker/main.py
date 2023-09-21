import nltk
from nltk.corpus import stopwords

def main():
    nltk.download('stopwords')
    print(list(stopwords.words('russian')))
    print('test')

if __name__ == "__main__":
    main()