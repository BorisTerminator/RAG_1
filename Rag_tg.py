import os
import re
import time
import math
import pandas as pd
from typing import List, Union, Dict, Tuple
from loguru import logger
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

class RAGProcessor:
    """
    Класс для работы с базой знаний и генерации ответов на вопросы с использованием LLM.
    Теперь он сохраняет историю диалога, чтобы при последующих вопросах учитывался предыдущий контекст.
    """

    DB_FILE_NAME = 'db'
    PDF_DIR = 'pdf'
    LOG_DIR = "log"
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 0
    MODEL_ID = 'intfloat/multilingual-e5-large'
    MODEL_KWARGS = {'device': 'cuda'}
    LOCAL_LLM = "owl/t-lite"
    # Возможны и другие модели: "llama3.2", "owl/t-lite" и т.д.

    # Добавлен новый placeholder {conversation_history} для передачи истории диалога
    RAG_PROMPT_GENERATE_ANSWER = """
Ты являешься помощником службы поддержки Unity Investments.
Вот контекст из базы знаний, который нужно использовать для ответа на вопрос: {context}
{conversation_history}
Внимательно проанализируй приведённый контекст.
Теперь просмотри вопрос пользователя: {question}
Дай ответ на этот вопрос, используя только вышеуказанный контекст.
Используй не более трёх предложений и будь лаконичен. Ответ:
    """

    def __init__(self) -> None:
        logger.add(
            os.path.join(self.LOG_DIR, "rag_pdf.log"),
            format="{time} {level} {message}",
            level="DEBUG",
            rotation="100 KB",
            compression="zip",
        )
        self.embeddings = self._initialize_embeddings()
        self.db = self._load_or_create_db()
        # Инициализация истории диалога
        self.conversation_history: List[str] = []

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Создает объект для векторных представлений."""
        logger.debug('Инициализация Embeddings')
        return HuggingFaceEmbeddings(model_name=self.MODEL_ID, model_kwargs=self.MODEL_KWARGS)

    def _load_or_create_db(self) -> FAISS:
        """Загружает или создает векторную базу знаний."""
        file_path = os.path.join(self.DB_FILE_NAME, "index.faiss")

        if os.path.exists(file_path):
            logger.debug('Загрузка существующей векторной базы знаний')
            return FAISS.load_local(
                self.DB_FILE_NAME, self.embeddings, allow_dangerous_deserialization=True
            )

        logger.debug('Создание новой векторной базы знаний')
        documents = self._load_documents(self.PDF_DIR)
        source_chunks = self._split_documents(documents)
        db = FAISS.from_documents(source_chunks, self.embeddings)
        db.save_local(self.DB_FILE_NAME)
        return db

    def _load_documents(self, directory: str) -> List[str]:
        """Читает PDF-файлы из указанной директории."""
        logger.debug(f'Загрузка документов из директории: {directory}')
        documents = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".pdf"):
                    logger.debug(f'Обработка файла: {file}')
                    loader = PyPDFLoader(os.path.join(root, file))
                    documents.extend(loader.load())
        return documents

    def _split_documents(self, documents: List[str]) -> List[str]:
        """Разделяет документы на чанки."""
        logger.debug('Разделение документов на чанки')
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE, chunk_overlap=self.CHUNK_OVERLAP
        )
        return text_splitter.split_documents(documents)

    def get_relevant_chunks(self, question: str, num_chunks: int, return_as_list: bool = False) -> Union[str, List[Dict[str, str]]]:
        """Извлекает релевантные чанки из базы знаний."""
        logger.debug('Поиск релевантных чанков')
        docs = self.db.similarity_search(question, k=num_chunks)

        if return_as_list:
            return [
                {
                    "metadata": doc.metadata,
                    "content": re.sub(r'\n{2}', ' ', doc.page_content)
                } for doc in docs
            ]

        return re.sub(r'\n{2}', ' ', '\n '.join([
            f'\n#### {i+1} Relevant chunk ####\n{doc.metadata}\n{doc.page_content}\n' for i, doc in enumerate(docs)
        ]))

    def generate_answer(self, question: str, message_content: str) -> str:
        """Генерирует ответ модели на основе контекста, истории диалога и вопроса."""
        logger.debug('Генерация ответа модели')
        # Формируем строку истории диалога, если таковая имеется
        conversation_history_str = ""
        if self.conversation_history:
            conversation_history_str = "\n".join(self.conversation_history) + "\n"

        prompt = self.RAG_PROMPT_GENERATE_ANSWER.format(
            context=message_content,
            conversation_history=conversation_history_str,
            question=question
        )

        llm = ChatOllama(model=self.LOCAL_LLM, temperature=0)
        generation = llm.invoke([HumanMessage(content=prompt)])
        return generation.content

    def process_question(self, question: str, num_chunks: int = 3, del_context: bool = False) -> Tuple[str, List[Dict[str, str]]]:
        """Обрабатывает вопрос, получая релевантный контекст, генерируя ответ и обновляя историю диалога."""
        if del_context:
            self.conversation_history = []
        logger.debug(f'Вопрос: {question}')
        relevant_chunks = self.get_relevant_chunks(question, num_chunks, return_as_list=True)
        answer = self.generate_answer(question, relevant_chunks)
        # Обновляем историю диалога, добавляя вопрос и ответ
        self.conversation_history.append(f"Вопрос: {question}")
        self.conversation_history.append(f"Ответ: {answer}")
        return answer, relevant_chunks
    
    def del_context(self, del_context: bool = False):
        "Удалает контекст, истию запросов"
        if del_context:
            self.conversation_history = []



rag = RAGProcessor()

import telebot
from telebot import types

# Замените на ваш токен
TOKEN = "8036620770:AAG3MxO4lv1wU2dcw87qwhvtRxgas9bQPu0"

bot = telebot.TeleBot(TOKEN)

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я бот по поддержке клиентов в Unity.", reply_markup=start_new_chat())

# Функция для создания клавиатуры с кнопкой "Начать новый диалог"
def start_new_chat():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)  
    button1 = types.KeyboardButton("Начать новый диалог ?")
    markup.add(button1)  
    return markup

# Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: message.text is not None)  # Обрабатывает любые текстовые сообщения
def echo_message(message):
    if message.text == 'Начать новый диалог ?':
        # Удаляем контекст (если это необходимо)
        rag.del_context(True)
        # Скрываем клавиатуру
        markup = types.ReplyKeyboardRemove()
        bot.send_message(message.chat.id, "Новый диалог начат. Задайте ваш вопрос.", reply_markup=markup)
    else:
        bot.send_message(message.chat.id, 'Думаю...')
        question = message.text
        model_answer = rag.process_question(question)[0]
        # Отправляем ответ и показываем кнопку "Начать новый диалог"
        bot.send_message(message.chat.id, model_answer, reply_markup=start_new_chat())

# Запуск бота
bot.polling(none_stop=True)