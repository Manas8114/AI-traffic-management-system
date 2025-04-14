from pymongo import MongoClient
import os

# Load environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB = os.getenv("MONGODB_DB", "traffic")

# Connect to MongoDB
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]

# Test insertion
test_data = {"message": "Test data"}
result = db.test_collection.insert_one(test_data)
print("Inserted ID:", result.inserted_id)
