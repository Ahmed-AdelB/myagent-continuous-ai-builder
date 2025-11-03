"""
Authentication module for MyAgent API
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets
import hashlib
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
import redis
from sqlalchemy.orm import Session
from loguru import logger

# Configuration
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Redis for session management
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)


class TokenData(BaseModel):
    """Token data model"""
    username: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None


class UserLogin(BaseModel):
    """User login model"""
    username: str
    password: str


class UserRegister(BaseModel):
    """User registration model"""
    username: str
    email: EmailStr
    password: str
    role: str = "user"


class Token(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class User(BaseModel):
    """User model"""
    id: str
    username: str
    email: str
    role: str
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool = True


class AuthService:
    """Authentication service"""

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)

    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

        return encoded_jwt

    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

        return encoded_jwt

    @staticmethod
    def decode_token(token: str) -> TokenData:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            role: str = payload.get("role")

            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            return TokenData(username=username, user_id=user_id, role=role)

        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @staticmethod
    def create_api_key(user_id: str, name: str, permissions: list = None) -> str:
        """Create API key for user"""
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Store in database (implementation depends on your DB setup)
        # For now, store in Redis with expiration
        key_data = {
            "user_id": user_id,
            "name": name,
            "permissions": permissions or [],
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None
        }

        redis_client.setex(
            f"api_key:{key_hash}",
            timedelta(days=365),
            str(key_data)
        )

        return api_key

    @staticmethod
    def validate_api_key(api_key: str) -> Optional[Dict]:
        """Validate API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_data = redis_client.get(f"api_key:{key_hash}")

        if key_data:
            # Update last used
            data = eval(key_data)
            data["last_used"] = datetime.utcnow().isoformat()
            redis_client.setex(
                f"api_key:{key_hash}",
                timedelta(days=365),
                str(data)
            )
            return data

        return None


class SessionManager:
    """Session management"""

    @staticmethod
    def create_session(user_id: str, token: str, ip_address: str = None, user_agent: str = None) -> str:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)

        session_data = {
            "user_id": user_id,
            "token": token,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "ip_address": ip_address,
            "user_agent": user_agent
        }

        # Store session in Redis with expiration
        redis_client.setex(
            f"session:{session_id}",
            timedelta(hours=24),
            str(session_data)
        )

        # Track active sessions for user
        redis_client.sadd(f"user_sessions:{user_id}", session_id)

        return session_id

    @staticmethod
    def get_session(session_id: str) -> Optional[Dict]:
        """Get session data"""
        session_data = redis_client.get(f"session:{session_id}")

        if session_data:
            data = eval(session_data)
            # Update last activity
            data["last_activity"] = datetime.utcnow().isoformat()
            redis_client.setex(
                f"session:{session_id}",
                timedelta(hours=24),
                str(data)
            )
            return data

        return None

    @staticmethod
    def invalidate_session(session_id: str):
        """Invalidate session"""
        session_data = redis_client.get(f"session:{session_id}")

        if session_data:
            data = eval(session_data)
            user_id = data.get("user_id")

            # Remove session
            redis_client.delete(f"session:{session_id}")

            # Remove from user's active sessions
            if user_id:
                redis_client.srem(f"user_sessions:{user_id}", session_id)

    @staticmethod
    def invalidate_all_user_sessions(user_id: str):
        """Invalidate all sessions for a user"""
        session_ids = redis_client.smembers(f"user_sessions:{user_id}")

        for session_id in session_ids:
            redis_client.delete(f"session:{session_id}")

        redis_client.delete(f"user_sessions:{user_id}")


class RateLimiter:
    """Rate limiting for API requests"""

    @staticmethod
    def check_rate_limit(identifier: str, limit: int = 100, window: int = 60) -> bool:
        """Check if rate limit is exceeded"""
        key = f"rate_limit:{identifier}"
        current = redis_client.get(key)

        if current is None:
            redis_client.setex(key, window, 1)
            return True

        if int(current) >= limit:
            return False

        redis_client.incr(key)
        return True


class PermissionChecker:
    """Permission checking"""

    ROLE_PERMISSIONS = {
        "admin": ["*"],
        "user": ["read", "write", "execute"],
        "viewer": ["read"],
        "agent": ["execute", "report"]
    }

    @staticmethod
    def has_permission(role: str, permission: str) -> bool:
        """Check if role has permission"""
        if role == "admin":
            return True

        permissions = PermissionChecker.ROLE_PERMISSIONS.get(role, [])
        return permission in permissions or "*" in permissions

    @staticmethod
    def require_permission(permission: str):
        """Decorator to require specific permission"""
        def decorator(func):
            async def wrapper(*args, current_user: TokenData = None, **kwargs):
                if not current_user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Not authenticated"
                    )

                if not PermissionChecker.has_permission(current_user.role, permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )

                return await func(*args, current_user=current_user, **kwargs)
            return wrapper
        return decorator


# Dependency for getting current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Get current authenticated user"""
    token = credentials.credentials

    # Check if token is blacklisted
    if redis_client.sismember("blacklisted_tokens", token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return AuthService.decode_token(token)


async def get_current_active_user(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """Get current active user"""
    # Here you would typically check if the user is active in the database
    # For now, just return the user
    return current_user


async def require_admin(current_user: TokenData = Depends(get_current_active_user)) -> TokenData:
    """Require admin role"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


class OAuth2Service:
    """OAuth2 integration for third-party authentication"""

    @staticmethod
    async def authenticate_github(code: str) -> Optional[Dict]:
        """Authenticate with GitHub OAuth"""
        # Implementation for GitHub OAuth
        pass

    @staticmethod
    async def authenticate_google(token: str) -> Optional[Dict]:
        """Authenticate with Google OAuth"""
        # Implementation for Google OAuth
        pass


class TwoFactorAuth:
    """Two-factor authentication"""

    @staticmethod
    def generate_totp_secret() -> str:
        """Generate TOTP secret"""
        return secrets.token_urlsafe(32)

    @staticmethod
    def generate_backup_codes(count: int = 10) -> list:
        """Generate backup codes"""
        return [secrets.token_urlsafe(8) for _ in range(count)]

    @staticmethod
    def verify_totp(secret: str, token: str) -> bool:
        """Verify TOTP token"""
        # Implementation for TOTP verification
        pass


# Audit logging
class AuditLogger:
    """Audit logging for security events"""

    @staticmethod
    def log_login(user_id: str, ip_address: str, success: bool):
        """Log login attempt"""
        event = {
            "type": "login",
            "user_id": user_id,
            "ip_address": ip_address,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(f"Login {'successful' if success else 'failed'} for user {user_id}")
        # Store in database for audit trail

    @staticmethod
    def log_permission_denied(user_id: str, resource: str, action: str):
        """Log permission denied event"""
        event = {
            "type": "permission_denied",
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.warning(f"Permission denied for user {user_id}: {action} on {resource}")
        # Store in database for audit trail