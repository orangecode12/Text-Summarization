from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Объявление глобальных параметров
FILE_PATH = 'интервью-павел-дуров.md'
CHUNK_SIZE = 4000

# Чтение транскрипта из файла
def read_transcript_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        transcript = file.read()
    return transcript

# Сегментация текста на основе символов '\n\n'
def segment_text(text, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=chunk_size, chunk_overlap=500)
    segments = text_splitter.create_documents([text])
    return segments


# Инициализация модели и цепочек
model = ChatOpenAI(model="gpt-3.5-turbo-0125")

key_point_prompt = ChatPromptTemplate.from_template("Обобщите документ. Для каждого фрагмента выделите 12 ключевых пунктов, сохранив значимую информацию {summary} Ключевые пункты:")
key_point_chain = key_point_prompt | model | StrOutputParser()

bullet_points_prompt = ChatPromptTemplate.from_template("Выделите основные темы текста. Дайте им значимое название.  Оформите отчет, где за названием темы следует перечисление фактов по этой теме. Постарайтесь сохранить всю существенную информацию и удалить малозначительные пункты. {grouped_topics} Отчет:")
bullet_points_chain = bullet_points_prompt | model | StrOutputParser()


# Обобщение и генерация ключевых пунктов
def get_key_points(text):
    summaries = segment_text(text, CHUNK_SIZE)
    key_points = []
    for summary in summaries:
        key_point = key_point_chain.invoke({ "summary": summary})
        key_points.extend(key_point.split("\n"))
    return key_points

def generate_bullet_points(grouped_topics):
    bullet_points = bullet_points_chain.invoke({ "grouped_topics" : grouped_topics})
    return bullet_points

def generate_followup(text):
    key_points = get_key_points(text)
    bullet_points = generate_bullet_points(key_points)
    return bullet_points

# Основное выполнение
transcript = read_transcript_from_file(FILE_PATH)
output = generate_followup(transcript)

#Вывод результата в файл
file = open("outut.txt", "w")
file.write(str(output))
file.close()