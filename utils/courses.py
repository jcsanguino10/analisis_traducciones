from functools import partial, reduce
from itertools import product, cycle
from bson.objectid import ObjectId
from html_utils.mongodb import remove_tags
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
import numpy as np
import pandas as pd

def find_tutorials(tutorials_collection, path:str): 
    """
    Finds the tutorials whose url path matches the arg provided.
    It can return multiple tutorials if tutorials in different
    languages share the same url path.

    Example: If for both portuguese and spanish the url is www.domain./<language>/excel-2016 then
    find_tutorials("excel-2016") will return [<tutorial1>, <tutorial2>].
    """ 
    ans =  list(tutorials_collection.find({'path': path}))
    print("Found {n} tutorials for course {c} \n".format(n = len(ans), c = path))
    return ans

def find_courses_by_name(tutorials_collection,course_url_path: str | tuple):
    """
    Finds the courses given its url path both in portuguese and spanish.

    If the url path of the course in pt and es is the same then
    the course_url_path argument must be a str with the url.
    
    If the url path is different the function expects a tuple
    in which the first element is the url path in portuguese and
    the second element the url path in spanish.

    It returns a dict with the following structure {<name_of_course_in_pt>: [<course1>, <course2>]}.
    Where course1 and course2 are the retrieved courses in each language.

    """
    #Find the tutorials given a tuple or string indicating the corresponding url paths of the courses.
    f = partial(find_tutorials,tutorials_collection)
    if type(course_url_path) == str:
        courses = [f(course_url_path)]
        key = [course_url_path]
    else:
        tutorials = list(reduce(lambda x, y: x + y, (map(lambda x: f(x),course_url_path)),[])) 
        courses = [tutorials]
        key = [course_url_path[0]]
    return dict(zip(key, courses)) 


def search_lesson(lesson_collection, id:str):
    """
    Receives the lessons collection connection to the db
    and a lesson id.
    
    Returns  a list with the lesson whose ids match. 
    """
    #Create the ObjectId
    _id = ObjectId(id)
    return list(lesson_collection.find({'_id': _id}))

def retrieve_lessons_text(lesson_collection, dict_tutorials: dict):
    """
    Given a connection to the db to find the lessons and a dict of the form:
    {<course_name>: [<course_pt>, <course_es>]}, returns a dict with the following structure:

    {<course_name>: {<course_language>: [lesson_1, ..., lesson_n]}}
    """
    # Partial function used to search the lessons 
    f = partial(search_lesson,lesson_collection) 
    #Accumulator dictionary where we are going to store all of the lessons 
    #sample new_dict = {k: }
    new_dict = {}
    for k,v in dict_tutorials.items():
        #key = name of course, v = list of courses 
        dictionary = new_dict.get(k)
        if not dictionary:
            new_dict[k] = {}
        for course in v:
            language = course.get('language')
            #Each course has multiple units and, in turn, each unit has multiple lessons
            lessons_per_unit = list(map(lambda x: x.get('ids'), course.get('units')))
            #Consolidate the unit ids in a single list
            complete_lessons = reduce(lambda acc,x: acc + x.split(','),lessons_per_unit,list())
            #Retrieve the lesson from the db
            lesson_list = reduce(lambda acc,x: acc + x,(map(f, complete_lessons)),list())
            #Flatten it into a consolidated list of the lesson text
            published_pages = reduce(lambda acc, x: acc + x,(map(lambda x: list(x.get('publish').get('pages').values()),lesson_list)))
            #Clean the html tags from the text
            clean_pub_pages = list(map(lambda x: remove_tags(x), published_pages)) 
            # Add the result to the dictionary
            new_dict[k][language] = clean_pub_pages 

    return new_dict

def concatenate_lessons_text(lessons_text_dict: dict):
    """
    Receives a dict of the form {<course_name>: {<course_language>: [lesson_1, ..., lesson_n]}}, concatenates
    all of the lessons text for each course, for each language and returns a dictionary of the form:

    {<course_name>: {<course_language>: <concatenated_lessons>}}
    """
    final_dict = {}
    #Iterar sobre los cursos
    for k1, v1 in lessons_text_dict.items():
        if not final_dict.get(k1):
            final_dict[k1] = {} 
        #Iterar sobre el diccionario que tiene el contenido
        for k2,v2 in v1.items():
            #Por el momento se decició concatenar todo el texto
            final_dict[k1][k2] = reduce(lambda acc,x: acc + x, v2, "")

    return final_dict


def keys_to_tuple(dict: dict):
    """
    Recieves a dictionary that has nested dictionaries and returns a list of the
    cartesian product of the first level keys with the second level keys.
    """

    first_lvl_keys = list(dict.keys())
    second_lvl_dicts = list(map(lambda x: dict[x], first_lvl_keys))
    second_lvl_keys = list(map(lambda x: list(x.keys()), second_lvl_dicts))
    flattened_keys = set(reduce(lambda acc, x: acc + x, second_lvl_keys, []))
    index_keys = list(product(first_lvl_keys, flattened_keys))

    return index_keys


def encode_course(encoder_name: str, text: dict):
    """
    Receives a SentenceTransformer encoder name and a dictionary with the structure:
    {<course_name>: {<course_language>: <concatenated_lessons_text>}} and returns a dictionary
    encoding all of the lessons with the following structure:

    {(<course_name>, <course_language>): <encoding>}
    """
    index_keys = keys_to_tuple(text)
    encoder = SentenceTransformer(encoder_name)
    encodings = []
    for course, language in index_keys:
        if text.get(course).get(language):
            encoding = encoder.encode(text.get(course).get(language))
            labeled = tuple([(course,language), encoding])
            encodings.append(labeled)

    return dict(encodings)


def PCA_course_encodings(course_encodings: dict):
    """
    Receives a dictionary of the form {(<course_name>, <course_language>): <encoding>}
    and returns a list of the form:
    
    [((<course_name>, <course_language>), <pca_of_embeddings>)]
    """
    pca = PCA(n_components=2)
    text_encodings = list(course_encodings.values())
    pca_encodings = pca.fit_transform(text_encodings)
    labeled_pca_encodings = list(zip(course_encodings.keys(), pca_encodings))
    return labeled_pca_encodings


def plot_course_encodings(labeled_pca_encodings: list):
    """
    Receives a list of the form [((<course_name>, <course_language>), <pca_of_embeddings>)],
    and plots the pca_vectors for each course, language pair.
    """
    x = list(map(lambda x: x[1][0],labeled_pca_encodings))
    y = list(map(lambda x: x[1][1],labeled_pca_encodings))
    keys = list(map(lambda x: x[0], labeled_pca_encodings))
    fig, ax = plt.subplots()
    ax.plot(x,y,ls="", marker ="o")
    for xi, yi, id_ in zip(x,y,keys):
        ax.annotate(str(id_), xy=(xi,yi))

    plt.show()


def cosine_similarity(a,b):
    """
    Utility function to calculate cosine similarity 
    """
    return dot(a,b)/(norm(a)*norm(b))

def get_google_translations_from_txt(path: str, course: str , lessons_dict: dict, acc=[]):
    """
    Receives:

        1. A path to the folder that contains a folder with the name of the course in which the .txt files will be found
        2. The name of the course whose translations are to be retrieved.
        3. A dictionary of the form : {<course_name>: {<course_language>: [lesson_1, ..., lesson_n]}}

    Returns a list with the contents of the read files.
    """
    if len(acc) == len(lessons_dict[course]['es']):
        return acc
    else:
        i = len(acc)
        with open(path + '/' + course +  '/google_translate-' + course + "-" + str(i) + ".txt") as f:
            return get_google_translations_from_txt(path,course, lessons_dict, acc + [f.read()])

def encode_course_by_lesson(lessons_dict: dict, course: str, encoder: str ,google_translations = []):
    """
    Receives:

        1. A dictionary of the form : {<course_name>: {<course_language>: [lesson_1, ..., lesson_n]}}
        2. The name of the course whose lessons are to be encoded into embbeddings
        3. The name of the encoder to use with the SentenceTransformer constructor
        4. An optional list of the lessons translated from spanish to portuguese by google.

    Returns a dictionary of the form: {<course_name>: {<course_language>: [encoding_lesson1, ..., encoding_lesson_n]}}
    """

    lessons = {course: {}}
    result = {course: {}} 
    if len(google_translations) == 0:
        lessons[course] = lessons_dict[course]
    else:
        lessons[course] = lessons_dict[course] | {"google": google_translations}
    encoder = SentenceTransformer(encoder)
    for k, v in lessons[course].items():
        encodings = encoder.encode(v)
        result[course][k] = encodings
    return result


def pca_encodings_by_lesson(encoded_lessons: dict, course:str):
    """
    Receives:

        1. A dictionary of the form {<course_name>: {<course_language>: [encoding_lesson1, ..., encoding_lesson_n]}}
        2. The course name whose encodings are to be reduced in dimensionality.

    Returns a list of vectors of the form [(<language>, <pca_vector>)].
    """
    pca = PCA(n_components=2)
    vectors = []
    for k, v in encoded_lessons[course].items():
        encodings_by_language = list(product([k], v))
        vectors += encodings_by_language

    only_vectors = list(map(lambda x: x[1], vectors))
    pca_vectors = pca.fit_transform(only_vectors)
    labeled = zip(pca_vectors, vectors)
    labeled_only_pca = list(map(lambda x: tuple([x[1][0], x[0]]), labeled))
    return labeled_only_pca 

def plot_labeled_pca_vectors(labeled_vectors: list):
    """
    Receives list of vectors of the form [(<language>, <pca_vector>)] and generates a
    scatter plot coloring the pca_vectors by language.
    """
    groups = set(list(map(lambda x: x[0], labeled_vectors)))
    vectors_list = list(map(lambda x: list(filter(lambda y: y[0] == x,labeled_vectors)), groups))
    vectors_list
    colors = cycle(['r','g','b'])
    for series in vectors_list:
        pca = list(map(lambda x: x[1], series))
        x = list(map(lambda x: x[0], pca))
        y = list(map(lambda x: x[1], pca))
        plt.scatter(x, y, color=next(colors))


def calculate_pairwise_distances_labeled(v1, v2) -> pd.DataFrame:
    """
    Receives two lists of vector embeddings and returns a dataframe
    with their corresponding cosine distances.
    """
    result = []
    for i in range(len(v1)):
        for j in range(len(v2)):
            similarity = cosine_similarity(v1[i][1], v2[j][1])
            result.append(similarity)
    matrix = np.array(result)
    #Resize to have matrix form
    matrix = matrix.reshape(len(v1), len(v2))
    return pd.DataFrame(matrix, columns=list(range(len(v2))))


def calculate_intracluster_pairwise_distances(encodings_dict: dict, course: str):
    """
    Receives:

        1. A dictionary of the form {<course_name>: {<course_language>: [encoding_lesson1, ..., encoding_lesson_n]}}
        2. Name of the course

    Returns a dictionary of the form {<course_name>: <course_language>: <nxn dataframe with cosine similarities of lessons>}}
    """

    keys = list(encodings_dict[course].keys())
    acc = {}
    vectors = (map(lambda x: list(product([x],encodings_dict[course][x])), keys))
    for encoding_list in zip(keys, vectors):
        language_vectors = encoding_list[1]
        language = encoding_list[0]
        if acc.get(language) is None:
            acc[language] = []
        #Pairwise cosine similarity
        acc[language] = calculate_pairwise_distances_labeled(language_vectors,language_vectors)

    return acc


def calculate_intercluster_pairwise_distances(encodings_dict: dict, course: str):
    """
    Receives:

        1. A dictionary of the form {<course_name>: {<course_language>: [encoding_lesson1, ..., encoding_lesson_n]}}
        2. Name of the course

    Returns a dictionary of the form {<course_name>: <language1language2>: <nxn dataframe with cosine similarities of lessons embeddings from language1 w.r.t language2>}}
    """

    keys = list(encodings_dict[course].keys())
    acc = {}
    vectors = (map(lambda x: list(product([x],encodings_dict[course][x])), keys))
    encoding_list = list(zip(keys, vectors))
    for i in range(len(encoding_list)):
        for j in range((len(encoding_list))):
            #Languages compared
            key1 = encoding_list[i][0]
            key2 = encoding_list[j][0]
            #Prevent double count
            if (key1 + key2) in acc.keys() or (key2 + key1) in acc.keys() or key1 == key2:
                continue
            else:
                acc[key1 + key2] = []
            matrix: pd.DataFrame = calculate_pairwise_distances_labeled(encoding_list[i][1], encoding_list[j][1])
            acc[key1 + key2] = matrix

    return acc