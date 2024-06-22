import cohere
from langchain.vectorstores import FAISS
from langchain.embeddings import CohereEmbeddings
import os
from langchain.memory import StreamlitChatMessageHistory

def get_answer(message, chat_history):
    co = cohere.Client('3Kcl5JLbJPfliBP1zGQLQ1SreitV2ulm9IG9SO3x')
    ctr = 0  

    os.environ["COHERE_API_KEY"] = "3Kcl5JLbJPfliBP1zGQLQ1SreitV2ulm9IG9SO3x"
    embeddings = CohereEmbeddings()

    db = FAISS.load_local("cohere_index", embeddings)


    context = db.similarity_search(message)


    context_str = ""

    for i in context:
        context_str = context_str + i.page_content
        # print(context_str)
        

    temp_prompt = f"You are a chatbot made for the discovery of past projects submitted to an intenational competition caled iGEM. Your job is to answer questions based on the context provided. Only use the following context to answer the question. If you do not know the answer of the the question say I don't know. Try to structure the output as well as possible. \n Context: %s" %(context_str)
    question_prompt = f"\n Question: %s: " %(message)
    final_prompt = temp_prompt + question_prompt
    # generate a response with the current chat history
    print(f"Chat_History: {chat_history}")
    print(f"Question Prompt: {question_prompt}")

    response = co.chat(
        final_prompt,
        temperature=0,
        max_tokens=1000,
        chat_history=chat_history
    )
    answer = response.text
    # print(f"Answer: {answer} \ns")

    # add message and answer to the chat history
    user_message = {"user_name": "User", "text": message}
    bot_message = {"user_name": "Chatbot", "text": answer}

        
    # chat_history.append(user_message)
    # chat_history.append(bot_message)
    
    return answer, user_message, bot_message
