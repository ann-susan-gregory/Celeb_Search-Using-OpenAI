#Integrate OpenAI API
import os
from constants import openai_key
from langchain_openai.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

#streamlit framework
st.title('Indian Celebrity Search')
input_text=st.text_input("Search about your favourite CELEB")

#Prompt Templates
first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

person_memory=ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory=ConversationBufferMemory(input_key='person',memory_key='chat_history')
descr_memory=ConversationBufferMemory(input_key='dob',memory_key='chat_history')


#OpenAI LLMs
llm=OpenAI(temperature=0.8)
chain=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)

#Prompt Templates
second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="When was {person} born"
)

chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)

#Prompt Templates
third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 movies released on {dob} in the world"
)

chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description',memory=descr_memory)

parent_chain=SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],
output_variables=['person','dob','description'],verbose=True)

if input_text:
    person_output = chain.run({'name': input_text})
    dob_output = chain2.run({'person': person_output})
    description_output = chain3.run({'dob': dob_output})

    st.write("Person: ", person_output)
    st.write("Date of Birth: ", dob_output)
    st.write("Movies: ", description_output)

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(descr_memory.buffer)