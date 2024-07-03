# Составление протокола встречи

## Задача:
Есть транскрипт встречи, необходимо сгенерировать для него followup (буллет-поинты). Followup -- протокол встречи, в нем должно содержаться самая важная информация (итоги, договоренности, ключевые решения), сгруппированная по темам.

## Решение:
Первое решение было очевидным: я отправила весь текст встречи в едином запросе к GPT-4. В ответ получила довольно неплохое обобщение (код и ответ можно посмотреть в simple_openai.py и respond_gpt4.txt соответственно). Но мне захотелось реализовать решение с использованием LangChain.
В результате я создала приложение openai_langchain.py. Алгоритм работы приложения:
1. Считать текст из файла указанного в FILE_PATH
2. Разбить исходный текст на блоки размера CHUNK_SIZE
3. Найти в каждом блоке 12 ключевых пунктов (первая языковая цепочка)
4. Разбить пункты по темам и сформировать отчет (вторая языковая цепь)
5. Записать результат в output.txt

Такая конструкция программы позволяет легко добавлять новые цепи (предобработки, например), менять размер блока и используемые внутри модели. 

Сейчас я использую внутри GPT-3.5, но хочу попробовать развернуть open source модель и подобрать подходящие промты и параметры.

Спасибо за это задание мне было очень интересно наконец познакомиться с интервью Павла Дурова и поэкспериментировать с языковыми цепями.