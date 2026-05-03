# AI Text Detection

## Run Backend

cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload

## Run Frontend

cd frontend
npm install
npm run dev
