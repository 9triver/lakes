from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import jwt

from lake_workbench.auth import AuthError, AuthIdentity, AuthenticationService


class AuthenticationServiceTests(unittest.TestCase):
    def test_development_mode_returns_fixed_identity(self) -> None:
        identity = AuthIdentity("development", "local", "dev@example.com", "Developer")
        service = AuthenticationService(dev_identity=identity)

        self.assertEqual(service.authenticate({}), identity)

    def test_cloudflare_mode_requires_team_domain_and_audience(self) -> None:
        with self.assertRaises(ValueError):
            AuthenticationService("cloudflare")

    def test_cloudflare_mode_requires_access_token(self) -> None:
        service = AuthenticationService(
            "cloudflare",
            team_domain="team.cloudflareaccess.com",
            audience="audience",
        )

        with self.assertRaisesRegex(AuthError, "login required"):
            service.authenticate({})

    def test_cloudflare_mode_verifies_header_token_and_extracts_identity(self) -> None:
        service = AuthenticationService(
            "cloudflare",
            team_domain="team.cloudflareaccess.com/",
            audience="audience",
        )
        service._jwks = Mock()
        service._jwks.get_signing_key_from_jwt.return_value.key = "public-key"
        claims = {
            "sub": "access-subject",
            "email": "Person@Example.com",
            "name": "Person",
        }

        with patch("lake_workbench.auth.service.jwt.decode", return_value=claims) as decode:
            identity = service.authenticate({"Cf-Access-Jwt-Assertion": "signed-token"})

        service._jwks.get_signing_key_from_jwt.assert_called_once_with("signed-token")
        decode.assert_called_once_with(
            "signed-token",
            "public-key",
            algorithms=["RS256"],
            audience="audience",
            issuer="https://team.cloudflareaccess.com",
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
        )
        self.assertEqual(
            identity,
            AuthIdentity("cloudflare-access", "access-subject", "person@example.com", "Person"),
        )

    def test_cloudflare_mode_accepts_authorization_cookie(self) -> None:
        service = AuthenticationService(
            "cloudflare",
            team_domain="https://team.cloudflareaccess.com",
            audience="audience",
        )
        service._jwks = Mock()
        service._jwks.get_signing_key_from_jwt.return_value.key = "public-key"
        claims = {"sub": "subject", "email": "person@example.com"}

        with patch("lake_workbench.auth.service.jwt.decode", return_value=claims):
            identity = service.authenticate({"Cookie": "other=x; CF_Authorization=cookie-token"})

        service._jwks.get_signing_key_from_jwt.assert_called_once_with("cookie-token")
        self.assertEqual(identity.name, "person@example.com")

    def test_token_errors_are_not_exposed(self) -> None:
        service = AuthenticationService(
            "cloudflare",
            team_domain="team.cloudflareaccess.com",
            audience="audience",
        )
        service._jwks = Mock()
        service._jwks.get_signing_key_from_jwt.side_effect = jwt.PyJWKClientConnectionError(
            "private network detail"
        )

        with self.assertRaisesRegex(AuthError, "^Invalid Cloudflare Access token$"):
            service.authenticate({"Cf-Access-Jwt-Assertion": "signed-token"})

    def test_local_bypass_is_limited_to_configured_network_without_access_token(self) -> None:
        service = AuthenticationService(
            "cloudflare",
            team_domain="team.cloudflareaccess.com",
            audience="audience",
            local_auth_bypass=True,
            local_network="192.168.30.0/24",
            local_user_id="default",
        )

        self.assertEqual(service.local_user_id_for("192.168.30.107", {}), "default")
        self.assertIsNone(service.local_user_id_for("192.168.31.107", {}))
        self.assertIsNone(
            service.local_user_id_for(
                "192.168.30.107",
                {"Cf-Access-Jwt-Assertion": "already-authenticated"},
            )
        )

    def test_local_bypass_requires_cloudflare_mode(self) -> None:
        with self.assertRaises(ValueError):
            AuthenticationService(
                local_auth_bypass=True,
                local_network="192.168.30.0/24",
            )

    def test_local_network_session_has_no_cloudflare_logout_url(self) -> None:
        service = AuthenticationService("cloudflare", team_domain="team.cloudflareaccess.com", audience="audience")

        payload = service.session_payload(
            AuthIdentity("local-network", "default", "owner@example.com", "Owner"),
            {"id": "default"},
        )

        self.assertEqual(payload["logout_url"], "")


if __name__ == "__main__":
    unittest.main()
