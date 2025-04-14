import secrets
import base64

# Generate a 32-byte (256-bit) random key
encryption_key = secrets.token_bytes(32)
# Convert to base64 for storage
base64_key = base64.urlsafe_b64encode(encryption_key).decode('utf-8')
print(f"ENCRYPTION_KEY={base64_key}")