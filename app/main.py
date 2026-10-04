from fastapi import FastAPI,HTTPException
from app.schema import TransactionInput
from app.model import predict_fraud

app = FastAPI(title='credit card fraud detection API')

@app.get('/health')
def health_check():
    return {'status':'ok'}

@app.post('/predict')
def predict(transaction:TransactionInput):
    try:
        result_prob, result_label = predict_fraud(transaction.model_dump())
        return{
             'fraud_probability' : round(result_prob,4),
            'is fraud': result_label
            
            }
    except ValueError as e:
        raise HTTPException(status_code = 400 , detail=str(e))
@app.get('/')
def root():
    return{
            'message': ' credit card fraud detection api',
            'docs' : '/docs',
            'health' : '/health'
            }
