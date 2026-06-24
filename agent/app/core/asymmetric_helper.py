# app/core/asymmetric_helper.py
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key

def sign_data_es256(data: str, private_key_pem: str) -> str:
    """
    Signs string data using ECDSA SECP256R1 (NIST P-256) and returns a Base64-encoded signature.
    """
    try:
        private_key = load_pem_private_key(
            private_key_pem.encode('utf-8'),
            password=None
        )
        
        if not isinstance(private_key, ec.EllipticCurvePrivateKey):
            raise ValueError("Provided key is not an Elliptic Curve Private Key.")

        signature = private_key.sign(
            data.encode('utf-8'),
            ec.ECDSA(hashes.SHA256())
        )
        
        return base64.b64encode(signature).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"ECDSA signing failed: {str(e)}")

def verify_data_es256(data: str, signature_b64: str, public_key_pem: str) -> bool:
    """
    Verifies ECDSA signature against the provided data.
    """
    try:
        public_key = load_pem_public_key(public_key_pem.encode('utf-8'))
        
        if not isinstance(public_key, ec.EllipticCurvePublicKey):
            raise ValueError("Provided key is not an Elliptic Curve Public Key.")

        signature = base64.b64decode(signature_b64)
        
        public_key.verify(
            signature,
            data.encode('utf-8'),
            ec.ECDSA(hashes.SHA256())
        )
        return True
    except Exception:
        return False
