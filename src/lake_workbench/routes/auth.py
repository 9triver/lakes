"""Authenticated session routes."""


def handle_auth_get(handler, path: str) -> bool:
    if path != "/api/auth/session":
        return False
    handler._json(
        handler.__class__.auth_service.session_payload(
            handler.auth_identity,
            handler.current_user,
        )
    )
    return True
