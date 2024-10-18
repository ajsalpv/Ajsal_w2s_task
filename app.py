from flask import Flask, render_template, request
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader



app=Flask(__name__)

#loading csv file using CSVLoader from langchain
loader = CSVLoader(file_path=r'A:\brototype\reference\w2s\south_indian_food_questions (1) (1).csv')
data=loader.load()

# cleaning and extracting text from the csv file and creating dictionary and as 
# question as key and answer as value and also create a list of questions to store in vectorstore
result_dict = {}
questions = []
for i in data:
    content = i.page_content.split('\n')
    question = content[0].replace('Question: ', '').strip()
    answer = content[1].replace('Answer: ', '').strip()  
    
    result_dict[question] = answer 
    questions.append(question) 

# use "sentence-transformers/all-MiniLM-L6-v2" model from HuggingFace to create embeddings of questions
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Using FAISS as vectorstore because it is faster for searching and free
db=FAISS.from_texts(questions,embedding)



def find_answer(query):
  """
  function for perform similarity search  with the query and the questions which stored in vectorstore.
  and retrieve one similar question from vectorstore and return the corresponding answer from a results dictionary.
  """
  similarity=db.similarity_search(query,k=1)
  similar_one=similarity[0].page_content
  return result_dict[similar_one]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_answer', methods=['POST'])
def answer():
    """ 
    Calling find_answer function and return the answer
    """
    user_question = request.form['question']
    answer = find_answer(user_question)
    return answer

if __name__ == '__main__':
    app.run(debug=True)