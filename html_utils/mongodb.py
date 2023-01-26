from importlib.resources import path
import pymongo
from pymongo import MongoClient
import pymysql
import certifi
import os

from bson.objectid import ObjectId
from bs4 import BeautifulSoup
import configparser
import json 

#Conexion con la bd
#ca = certifi.where()
#cluster = MongoClient(host="mongodb+srv://read_db:Dak2ZIvwL7ZNqIt6@gcflearnfree.ivza6.azure.mongodb.net/gcfglobal?retryWrites=true&w=majority", tlsCAFile=ca)
#db = cluster["gcfglobal"]
# Donde se va a apuntar 
#collection2= db["tutorial"]
#collection3=db["lesson"]
#Filtro a colocar
#body = {"language":"en"} 


#coleccion
#resultsTutorial =collection2.find(body)


#Obtener el html en texto plano
def remove_tags(html):
    # parse html content
    soup = BeautifulSoup(html, "html.parser")
    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()
    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)


#Get links from html 
def get_links(html):
    soup = BeautifulSoup(html, "lxml")
    links_dictionary ={}
    i=0
    for link in soup.findAll('a'):
        links_dictionary[i]=link.get('href')
        i=i+1
    # Serializing json  
    json_object = json.dumps(links_dictionary, indent = 4) 
    return json_object


#Get youtube's ids 
def get_ids_youtube(result):
    dic_Ids = {}
    try:
        if result["videos"] is not None :
            i=0
            for id in result["videos"]:
                dic_Ids[i]=id["youtube_id"]    
    except:
        pass
    json_object = json.dumps(dic_Ids, indent = 4) 
    return json_object

#Realiza la lista de publishers que se van a ingresar a la bd 
    #En caso de que si se cuemplan con todas las condiciones, se retorna la info a guardar. 
    #En caso de que no se cumplan las condiciones, se retorna ""
def hacerComprobacionData(lesson, path, i ):
    listNotCourses=["Spanish","Basic Spanish Skills","English","Apprendre l'anglais","Englisch lernen","Arabic","Chinese","Aprenda Inglês","Aprenda Inglés","Korean","Simple English","Français","Français","Initiation all'informatique","Pусский","Интернет-безопасность","Интернет-безопасность для детей","русский","中文","中文","计算机基础","计算机基本技能","儿童因特网安全","因特网安全","平面设计入门","谷歌地图","Windows基础","谷歌使用技巧","Norsk","Grunnleggende IT-ferdigheter","Grunnleggende om Datamaskiner","Grunnleggende om Windows","Internett-sikkerhet","Bahasa Indonesia","Dasar-Dasar Komputer","Dasar-Dasar Windows","Excel 2016 (Bahasa Indonesia)","Google Maps (Bahasa Indonesia)","Keamanan Internet","Keamanan Internet Untuk Anak-anak","Kemampuan-Kemampuan Komputer Dasar","Word 2016 (Bahasa Indonesia)","Greek","Χάρτες Google","Μικρές συμβουλές Google","Λογαριασμός Google","Gmail (Greek)","Google Drive (Greek)","Φόρμες Google","Έγγραφα Google","Παρουσιάσεις Google","Παρουσιάσεις Google","Υπολογιστικά Φύλλα Google","YouTube (Greek)","Arabic","أساسيات الكمبيوتر","أكسيل 2016","Kyrgyz","Балдар үчүн интернет коопсуздугу","Интернет коопсуздугу","Google Карталар","Компьютерде иштөө негиздери","Графикалык дизайн негиздери","Google колдонуу кеңештери","Windows негиздери","Компьютер негиздери","PowerPoint 2016 Негиздери","Excel 2016 Негиздери","Word 2016 Негиздери","Dutch","Computer Basiskennis","Computer basisvaardigheden","Polski","Excel (Polski)" ]    
    file = open("/home/profesor/html/noHtml.txt", "r")
    contador=int(file.readlines()[-1].split(",")[0])
    file.close()

    #Revisar si es una lesson o una pagina informativa 
    if (len(lesson["tutorials"])>0 and lesson["published"]=="true"):
            course=lesson["tutorials"][0]["title"]
            if  not(course in listNotCourses ):

                #obtener la info necesaria 
                lesson_id= str(lesson["_id"])
                html= remove_tags( lesson["publish"]["pages"]["1"])
                links=get_links(lesson["publish"]["pages"]["1"]) 
                lista_youtube= get_ids_youtube(lesson)

                #Maneja los htmls que no tienen contenido
                if html =="":
                    contador+=1
                    file = open("/home/profesor/html/noHtml.txt", "a")
                    file.write(str(contador)+","+lesson_id + ","+ path+ os.linesep)
                    file.close()
                else:  
                    i+=1
                    data= (int(i),  lesson_id, path, html, links, lista_youtube, json.dumps({}, indent = 4) ,json.dumps({}, indent = 4) )
                    return data
    return ""


def main_function(resultsTutorial):
    #lista con datos para colocar en la bd 
    list_data=[]
    file = open("/home/profesor/html/noHtml.txt", "w")
    file.write("0"+",publish id, path"+ os.linesep)
    file.close()

    i=0
    for result in resultsTutorial:
        if not (result["path"].__contains__('tr_'))  and not( result["description"].__contains__('your native language')):
            for unidades in result["units"]:
                for idPublish in unidades["ids"].split(","):
                    if (idPublish!=""):
                        lesson = collection3.find_one({"_id": ObjectId(idPublish)})
                        #metodo para hacer data 
                        datoLesson =hacerComprobacionData(lesson, result["path"] , i)
                        if datoLesson != "":
                            list_data.append(datoLesson)
                            i+=1
    return list_data


def export_to_db(list_data, h, u, p, db):
    scrap_db = pymysql.connect(host=h, user=u, password=p, db=db) 
    cursor = scrap_db.cursor()
    query="""INSERT INTO publish  (number, id_lesson, course_path, html, links , id_links, lib1 , lib2 ) 
    VALUES (%s,%s, %s, %s, %s ,%s, %s, %s); """
    cursor.executemany(query, list_data)
    scrap_db.commit()
    scrap_db.close()


def remove_and_create_table(h,u,p,db):
    scrap_db = pymysql.connect(host=h, user=u, password=p, db=db)
    cursor = scrap_db.cursor()
    cursor.execute("DROP TABLE IF EXISTS publish")
    sql="CREATE TABLE publish ( number Int PRIMARY KEY, id_lesson VARCHAR(100), course_path VARCHAR(100), html TEXT, links json, id_links json, lib1 json, lib2 json);"
    cursor.execute(sql)
    alterTable="alter table publish modify html longtext;"
    cursor.execute(alterTable)
    scrap_db.close()

#list_data=main_function(resultsTutorial)
#print (len(list_data) )

#config = configparser.RawConfigParser()
#config.read(filenames = 'access.properties')

#h = config.get('mysql','host')
#u = config.get('mysql','user')
#p = config.get('mysql','password')
#db = config.get('mysql','db')

#Descomentar para correr segun lo que se necesite. 
#remove_and_create_table(h, u, p, db)
#export_to_db(list_data,  h, u, p, db)



