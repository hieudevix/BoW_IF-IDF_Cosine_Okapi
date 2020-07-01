import os
import re
from bs4 import BeautifulSoup
import nltk
from pyvi import ViTokenizer
import gensim
from newspaper import Article
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize #import hàm xử lý tách từ,tách câu
from nltk.corpus import stopwords #import tập hư từ
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from string import punctuation #import tập dấu câu từ thư viện string
my_stopwords = set(stopwords.words('english') + list(punctuation))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
from gensim.summarization.bm25 import get_bm25_weights
stop_words = open('vietnamese_stopwords.txt',encoding="utf8")
a=stop_words.read()
my_stopwords = a 


#Hàm trả về type của file
def getTypeOfFile(path):
    return path[-3:]
# Hàm đọc đường dẫn file                
def readLinkFile(path):   
    list_path=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            list_path.append(root+"/"+file)
    return list_path
            
# Hàm đọc file
def readFile(list_path):
    read_files = []
    i=0
    for i in range (len(list_path)):
        read_file=open(list_path[i], "r" ,encoding="utf8")
        a=read_file.readlines()
        a = ' '.join(a)
        read_files.append(a)            
    return read_files

#Hàm lấy dữ liệu từ website
def getData(url):
    data_url = []
    article = Article(url)
    article.download()
    article.parse()
    data_url.append(article.text)
    return data_url


#Hàm lấy tên file
def getNameTxtFile(path):
    if getTypeOfFile(path) == "txt":
        arrSplit = path.split('/')
        nameTxtFile = arrSplit[len(arrSplit)-1]
        return nameTxtFile
#Hàm xóa thẻ html
def clean_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()
#Hàm loại bỏ các ký tự đạc biệt
def remove_special_character(text):
     #Thay thế các ký tự đặc biệt bằng ''
    string  = re.sub('[^\w\s]','',text)
    #xóa ký tự \n
    string = string.replace("nn", "")
    #Xử lí các khoảng trắng thừa ở giữa chuỗi
    string = re.sub('\s',' ',string)
    #Xử lý khoảng trắng thừa ở đầu và cuối câu
    string = string.strip() 
    return string
    
    return string
# Hàm xóa trùng
def remove_duplicates(text):
    return list(set(text))
#Tách câu tách từ, loại bỏ hư từ
#Tách câu tách từ, loại bỏ hư từ
def filterTexts(txtArr):
    res = []
    for i in range(len(txtArr)):
        # text_cleaned = clean_html(txtArr[i])
        # #Tách câu
        # sents  = sent_tokenize(text_cleaned)
        # #Loại bỏ ký tự đặc biệt
        # sents_cleaned = [remove_special_character(s) for s in sents]
        # #Nối các câu lại thành text
        # text_sents_join = ''.join(sents_cleaned)
        # #Tách từ
        # words = word_tokenize(text_sents_join)
        # #Đưa về dạng chữ thường
        # words = [word.lower() for word in words]
        # #Loại bỏ hư từ
        # words = [word for word in words if word not in my_stopwords]
        # words = [ps.stem(word) for word in words]
        # #Xóa kí tự trùng
        # # words = remove_duplicates(words)
        # words = ' '.join(words) 
       
        # res.append(words)
        #vietnamese
        #tách từ trong tiếng việt
        lines = ViTokenizer.tokenize(txtArr[i])  
        lines = lines.replace('\ n', ' ')
        #Xư lí các ký tự đặc biệt
        lines = gensim.utils.simple_preprocess(lines)
        #Loại bỏ stop word
        lines = [line for line in lines if line not in my_stopwords]
        #Xóa trùng
        # lines = remove_duplicates(lines)
        lines = '\n'.join(lines)         
        res.append(lines)
    return res
#Hàm trà về ma trận vector
def bagOfWords(txtArr):
    res = CountVectorizer()
    return res.fit_transform(txtArr).todense()

#tf-idf
def tF_IDF(txtArr):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=0, stop_words='english')
    tf_idf_matrix = tf.fit_transform(txtArr)
    dense = tf_idf_matrix.todense()
    return dense

#Ghi ra file txt
def writeFile(txtArrAfter, outputName, path):
    f = open(path + "/" + outputName + "_word.txt", 'w')
    f.write(str(txtArrAfter))
    f.close()
    
    
def writeVectorFite(txtArrAfter, outputName, list_path, path):
    f = open(path + "/" + outputName + ".txt", 'w')
    stt = 1
    for i in range(len(txtArrAfter)): 
        strPath = list_path[i].split('/')
        txtFileName = strPath[len(strPath)-1]
        stt_txtFileName = str(stt) + " " + txtFileName
        stt = stt + 1
        f.write(stt_txtFileName)
        f.write("\n")
        for j in range(len(txtArrAfter[i])):
            f.write(str(txtArrAfter[i][j]))           
        f.write("\n")   
    f.close()
#Ghi cosine ra file
def writeCosineFile(txtArrAfter, outputName, path):
    f = open(path + "\\" + outputName + ".txt", 'w')
    for i in range(len(txtArrAfter)):
        for j in range(len(txtArrAfter)):
            cossim = round(1 - spatial.distance.cosine(txtArrAfter[i], txtArrAfter[j]),3)
            res = cossim
            #write float to file
            if len(str(res)) == 3:
                res = str(res) + "   "
            elif len(str(res)) == 4:
                res = str(res) + "  "
            else:
                res = str(res) + " "
            f.write(res)
        f.write("\n")             
    f.close()


#Ghi okapi ra file text
def writeOkapiFile(txtArrAfter, outputName, path):
    f = open(path + "/" + outputName + ".txt", 'w')
    arr = []
    for i in range(len(txtArrAfter)):
        item  = txtArrAfter[i].split(" ")
        arr.append(item)
    result = get_bm25_weights(arr, n_jobs=-1)
    result = str(result)
    f.write(result)
    f.close()

#main   
def main():
    i_path = input('input path:')
    list_path = readLinkFile(i_path)
    o_path = input('output path:')   
    read_files = readFile(list_path)
    files_filted = filterTexts(read_files)
    BagOfWord = bagOfWords(files_filted)
    TF_IDF = tF_IDF(files_filted)
    while True:
        print("~~MENU~~")
        print(' 1.Bow ')
        print(' 2.TF-IDF ')
        keyStep2 = input('Chon: ')
        keyStep2 = str(keyStep2)
        if keyStep2 == '1':
            writeVectorFite(BagOfWord, "BoW", list_path, o_path)
            print(BagOfWord)
            break
        if keyStep2 == '2':
            print(TF_IDF)
            writeVectorFite(TF_IDF, "TF-IDF", list_path, o_path)
            break
    while True:
        print("~~MENU~~")
        print(' 1.Cosine')
        print(' 2.Okapi ')
        keyStep3 = input('Chon: ')
        if keyStep3 == '1':
            if keyStep2 == '1':
                writeCosineFile(BagOfWord, "BoW_Cosine", o_path)
            else:
                writeCosineFile(TF_IDF, "TF-IDF_Cosine", o_path)
            break
        if keyStep3 == '2':
            if keyStep2 == '1':
                writeOkapiFile(files_filted, "BoW_OkapiBM25", o_path)
            else:
                writeOkapiFile(files_filted, "TF-IDF_OkapiBM25", o_path)
            break
    print('Đã xuất file !')

    
if __name__ == "__main__":
    main()
    
    
