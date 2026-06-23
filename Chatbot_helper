import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain

load_dotenv()  # Load your GOOGLE_API_KEY from .env

def generate_response(skin_type, color):
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.6
    )

    prompt_template = PromptTemplate(
        input_variables=["skin_type", "color"],
        template="Suggest 5 makeup steps and types of makeup products suitable for a {skin_type} skin type and {color} color girl."
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run({'skin_type': skin_type, 'color': color})
    return response
