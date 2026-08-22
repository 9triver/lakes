"""Authenticate requests using Cloudflare Access JWT assertions."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from http.cookies import CookieError, SimpleCookie
from ipaddress import IPv4Network, IPv6Network, ip_address, ip_network

import jwt
from jwt import PyJWKClient


class AuthError(ValueError):
    """Raised when a request has no valid authenticated identity."""


@dataclass(frozen=True)
class AuthIdentity:
    provider: str
    subject: str
    email: str
    name: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


class AuthenticationService:
    """Resolve a trusted external identity for each API request."""

    def __init__(
        self,
        mode: str = "development",
        *,
        team_domain: str = "",
        audience: str = "",
        dev_identity: AuthIdentity | None = None,
        local_auth_bypass: bool = False,
        local_network: str = "",
        local_user_id: str = "default",
    ) -> None:
        if mode not in {"development", "cloudflare"}:
            raise ValueError("LAKES_AUTH_MODE must be development or cloudflare")
        self.mode = mode
        self.team_domain = self._normalize_team_domain(team_domain)
        self.audience = audience.strip()
        self.local_auth_bypass = bool(local_auth_bypass)
        self.local_user_id = local_user_id.strip()
        self.local_network: IPv4Network | IPv6Network | None = None
        self.dev_identity = dev_identity or AuthIdentity(
            provider="development",
            subject="default",
            email="developer@localhost",
            name="本地开发用户",
        )
        if mode == "cloudflare" and (not self.team_domain or not self.audience):
            raise ValueError(
                "Cloudflare authentication requires LAKES_CF_TEAM_DOMAIN and LAKES_CF_AUD"
            )
        if self.local_auth_bypass:
            if mode != "cloudflare":
                raise ValueError("Local authentication bypass requires cloudflare auth mode")
            if not self.local_user_id or not local_network.strip():
                raise ValueError("Local authentication bypass requires network and user id")
            try:
                self.local_network = ip_network(local_network.strip(), strict=False)
            except ValueError as exc:
                raise ValueError("LAKES_LOCAL_NETWORK must be a valid IP network") from exc
        self._jwks = (
            PyJWKClient(f"{self.team_domain}/cdn-cgi/access/certs", cache_jwk_set=True)
            if mode == "cloudflare"
            else None
        )

    @classmethod
    def from_env(cls) -> "AuthenticationService":
        return cls(
            os.environ.get("LAKES_AUTH_MODE", "development").strip().lower(),
            team_domain=os.environ.get("LAKES_CF_TEAM_DOMAIN", ""),
            audience=os.environ.get("LAKES_CF_AUD", ""),
            local_auth_bypass=cls._truthy(os.environ.get("LAKES_LOCAL_AUTH_BYPASS", "")),
            local_network=os.environ.get("LAKES_LOCAL_NETWORK", ""),
            local_user_id=os.environ.get("LAKES_LOCAL_USER_ID", "default"),
            dev_identity=AuthIdentity(
                provider="development",
                subject=os.environ.get("LAKES_DEV_USER_SUBJECT", "default"),
                email=os.environ.get("LAKES_DEV_USER_EMAIL", "developer@localhost"),
                name=os.environ.get("LAKES_DEV_USER_NAME", "本地开发用户"),
            ),
        )

    def local_user_id_for(self, client_host: str, headers) -> str | None:
        """Return the explicitly configured local User for a trusted LAN request."""
        if not self.local_auth_bypass or self.local_network is None:
            return None
        if self._header_token(headers) or self._cookie_token(headers.get("Cookie", "")):
            return None
        try:
            address = ip_address(str(client_host).strip())
        except ValueError:
            return None
        return self.local_user_id if address in self.local_network else None

    def authenticate(self, headers) -> AuthIdentity:
        if self.mode == "development":
            return self.dev_identity
        token = self._header_token(headers) or self._cookie_token(headers.get("Cookie", ""))
        if not token:
            raise AuthError("Cloudflare Access login required")
        try:
            signing_key = self._jwks.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=self.team_domain,
                options={"require": ["exp", "iat", "sub", "aud", "iss"]},
            )
        except jwt.PyJWTError as exc:
            raise AuthError("Invalid Cloudflare Access token") from exc
        subject = str(claims.get("sub") or "").strip()
        email = str(claims.get("email") or "").strip().lower()
        if not subject:
            raise AuthError("Cloudflare Access token has no subject")
        name = str(claims.get("name") or claims.get("common_name") or email or subject).strip()
        return AuthIdentity(provider="cloudflare-access", subject=subject, email=email, name=name)

    def session_payload(self, identity: AuthIdentity, user: dict) -> dict:
        return {
            "authenticated": True,
            "mode": self.mode,
            "identity": identity.as_dict(),
            "user": user,
            "logout_url": "/cdn-cgi/access/logout"
            if self.mode == "cloudflare" and identity.provider == "cloudflare-access"
            else "",
        }

    @staticmethod
    def _header_token(headers) -> str:
        return str(headers.get("Cf-Access-Jwt-Assertion") or "").strip()

    @staticmethod
    def _truthy(value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _cookie_token(cookie_header: str) -> str:
        cookie = SimpleCookie()
        try:
            cookie.load(cookie_header)
        except CookieError:
            return ""
        value = cookie.get("CF_Authorization")
        return value.value if value else ""

    @staticmethod
    def _normalize_team_domain(value: str) -> str:
        domain = value.strip().rstrip("/")
        if domain and not domain.startswith(("http://", "https://")):
            domain = f"https://{domain}"
        return domain
