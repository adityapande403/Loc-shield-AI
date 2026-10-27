# LocalShield AI Deployment Guide

## 1. Local Python SDK Deployment
Install the package locally:
```bash
pip install -e .
```
Initialize and use directly in Python:
```python
from localshield import LocalShield

shield = LocalShield.from_pretrained("./models/localshield_multilingual")
result = shield.redact("Call Rahul at 9876543210")
print(result.redacted_text)
```

## 2. FastAPI Service Deployment
Run the service on localhost:
```bash
localshield serve --host 127.0.0.1 --port 8000
```
Or via uvicorn directly:
```bash
uvicorn localshield.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

### Endpoints
- `GET /health` - Health status and backend details.
- `GET /ready` - Readiness check for model availability.
- `GET /taxonomy` - Supported PII categories.
- `POST /detect` - Detect sensitive entities.
- `POST /redact` - Redact sensitive entities.
- `POST /redact/batch` - Redact batches of texts.
- `POST /analyze` - Leakage risk scoring.

## 3. Node.js Client Deployment
In your Node.js application:
```javascript
const { LocalShieldClient } = require("./sdk_node");

const shield = new LocalShieldClient({ baseUrl: "http://127.0.0.1:8000" });
const res = await shield.redact("Contact Rahul Sharma at 9876543210");
console.log(res.redacted_text);
```
