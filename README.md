

### MYBot — AI News Research Toot
Ask questions. Get answers. From any news article on the web.
Powered by Google Gemini + LangChain + FAISS


### What is it ?
MYBot is an AI-powered news research assistant that lets you drop in any news article URLs and instantly ask questions about them. No more reading through entire articles — just ask, and RockyBot answers with sources.
Built with:

🤖 Google Gemini — for intelligent, context-aware answers
🔗 LangChain — for document processing pipeline
🗄️ FAISS — for fast local vector search
🧠 HuggingFace Embeddings — local, free, no API needed
🌐 Streamlit — clean, interactive web UI


### 🎬 Demo
1. Paste up to 3 news article URLs in the sidebar
2. Click "Process URLs"
3. Ask any question about the articles
4. Get an AI-generated answer with sources

### 🛠️ Setup & Installation
1. Clone the repo
bashgit clone https://github.com/KishanRavat2898/News_Research_Ai_Powered.git
cd News_Research_Ai_Powered
2. Create virtual environment
bashpython -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
3. Install dependencies
bashpip install -r requirements.txt
4. Set up your API key
Create a .env file in the project root:
envGOOGLE_API_KEY=your_google_api_key_here
Get your free API key at 👉 Google AI Studio
5. Run the app
bashstreamlit run main.py

### 📦 Requirements
txtstreamlit
langchain
langchain-core
langchain-community
langchain-google-genai
langchain-huggingface
langchain-text-splitters
faiss-cpu
sentence-transformers
beautifulsoup4
requests
python-dotenv

🧠 How It Works
URLs Entered
     ↓
Fetch Article Text (BeautifulSoup)
     ↓
Split into Chunks (RecursiveCharacterTextSplitter)
     ↓
Generate Embeddings (HuggingFace all-MiniLM-L6-v2)
     ↓
Store in FAISS Vector Index
     ↓
User Asks Question
     ↓
Semantic Search → Top 3 Relevant Chunks
     ↓
Send to Gemini LLM with Context
     ↓
Answer + Sources Displayed

💡 Example Usage
URLs:

https://en.wikipedia.org/wiki/Tata_Consultancy_Services
https://en.wikipedia.org/wiki/Infosys
https://en.wikipedia.org/wiki/Wipro

Questions you can ask:

"What is the revenue of TCS?"
"Where is Infosys headquartered?"
"Who founded Wipro?"
"Compare TCS and Infosys employee count"


📁 Project Structure
📦 News_Research_Ai_Powered
 ┣ 📜 main.py              # Main Streamlit app
 ┣ 📜 requirements.txt     # Dependencies
 ┣ 📜 .env                 # API keys (not committed)
 ┣ 📜 .gitignore           # Ignored files
 ┗ 📜 README.md            # You are here
