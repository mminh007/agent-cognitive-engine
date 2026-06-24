# agent/test_grpc_receipt.py
import sys
import os
import hashlib

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.settings import settings
from app.core.asymmetric_helper import sign_data_es256, verify_data_es256

def test_ai_response_receipt_flow():
    print("=== Testing AI Response Receipt Cryptographic Flow ===")
    
    # 1. Simulate gRPC Stream chunks
    chunks = ["Hello! ", "I am an AI assistant. ", "I have processed your request safely."]
    accumulated_text = "".join(chunks)
    print(f"Accumulated Response Text: '{accumulated_text}'")
    
    # 2. Compute SHA-256 hash of response
    response_hash = hashlib.sha256(accumulated_text.encode('utf-8')).hexdigest()
    print(f"Computed SHA-256 Hash: {response_hash}")
    
    # 3. Simulate Receipt metadata
    session_id = "test-session-123"
    timestamp = 1719224400
    
    # Data structure to sign: session_id + timestamp + response_hash
    data_to_sign = f"{session_id}:{timestamp}:{response_hash}"
    print(f"Data to sign: '{data_to_sign}'")
    
    # 4. Generate Signature (using Private Key from Settings)
    print("Generating signature...")
    signature = sign_data_es256(data_to_sign, settings.security.ai_receipt_private_key)
    print(f"Signature (Base64): {signature}")
    
    # 5. Verify Signature (using Public Key from Settings)
    print("Verifying signature with public key...")
    verified = verify_data_es256(data_to_sign, signature, settings.security.ai_receipt_public_key)
    
    if verified:
        print("✅ SUCCESS: AI Response Receipt verification PASSED!")
    else:
        print("❌ FAILURE: AI Response Receipt verification FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    test_ai_response_receipt_flow()
