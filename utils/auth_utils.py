"""
Authentication and user management utilities
"""
import hashlib
import jwt
import time
from datetime import datetime, timedelta
import secrets
import bcrypt

class AuthManager:
    def __init__(self):
        self.secret_key = "hackathon_churn_prediction_2024"  # In production, use environment variable
        self.users = {
            "admin": {
                "password_hash": self._hash_password("admin123"),
                "role": "admin",
                "permissions": ["read", "write", "admin", "reports", "alerts"]
            },
            "analyst": {
                "password_hash": self._hash_password("analyst123"), 
                "role": "analyst",
                "permissions": ["read", "write", "reports"]
            },
            "viewer": {
                "password_hash": self._hash_password("viewer123"),
                "role": "viewer", 
                "permissions": ["read"]
            }
        }
        
    def _hash_password(self, password):
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def _verify_password(self, password, hashed):
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def authenticate(self, username, password):
        """Authenticate user credentials"""
        if username in self.users:
            user = self.users[username]
            if self._verify_password(password, user["password_hash"]):
                return {
                    "username": username,
                    "role": user["role"],
                    "permissions": user["permissions"],
                    "authenticated": True
                }
        return {"authenticated": False}
    
    def create_token(self, user_data):
        """Create JWT token for authenticated user"""
        payload = {
            "username": user_data["username"],
            "role": user_data["role"],
            "permissions": user_data["permissions"],
            "exp": datetime.utcnow() + timedelta(hours=24),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_token(self, token):
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return {"error": "Token expired"}
        except jwt.InvalidTokenError:
            return {"error": "Invalid token"}
    
    def has_permission(self, user_data, required_permission):
        """Check if user has required permission"""
        if not user_data.get("authenticated"):
            return False
        return required_permission in user_data.get("permissions", [])
    
    def create_user(self, username, password, role="viewer"):
        """Create new user"""
        if username in self.users:
            return {"success": False, "error": "User already exists"}
        
        permissions = {
            "admin": ["read", "write", "admin", "reports", "alerts"],
            "analyst": ["read", "write", "reports"],
            "viewer": ["read"]
        }.get(role, ["read"])
        
        self.users[username] = {
            "password_hash": self._hash_password(password),
            "role": role,
            "permissions": permissions
        }
        
        return {"success": True, "message": f"User {username} created successfully"}

# Global auth manager
auth_manager = AuthManager()