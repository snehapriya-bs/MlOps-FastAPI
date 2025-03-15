Here’s a `README.md` draft:

---

# **FastAPI Prediction Service**

This project sets up a FastAPI service for making predictions based on a bike rental dataset.

## 🚀 **Setup**
1. **Create a virtual environment:**
```bash
python -m venv fastapi
```
2. **Activate the environment:**
```bash
fastapi\Scripts\activate
```
3. **Install dependencies:**
```bash
pip install -r requirements/requirements_api.txt
```
4. **Run the FastAPI app:**
```bash
python .\app\main.py
```

## 📡 **Example API Call**
Use `curl` to test the prediction endpoint:
```bash
curl -X POST "http://127.0.0.1:8001/api/v1/predict" \
-H "Content-Type: application/json" \
--data-raw '{
  "inputs":[
    {
      "dteday":"2012-11-05","season":"winter","hr":"6am","holiday":"No",
      "weekday":"Mon","workingday":"Yes","weathersit":"Mist","temp":6.1,
      "atemp":3.0014,"hum":49.0,"windspeed":19.0012,"casual":4,"registered":135
    }
  ]
}'
```

