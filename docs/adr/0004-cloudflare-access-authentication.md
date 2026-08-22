# Authenticate Users with Cloudflare Access

Status: accepted.

## Decision

Cloudflare Access is the production authentication boundary. Lakes verifies the Access JWT from the `Cf-Access-Jwt-Assertion` header or `CF_Authorization` cookie against Cloudflare's JWKS, audience, and issuer. It never trusts identity headers without a valid JWT.

The verified external identity maps to one Lakes User. A first-time User receives one empty default Workspace. `LAKES_BOOTSTRAP_EMAIL` may bind the first production identity to the existing default User, and `LAKES_ADMIN_EMAILS` controls administrative access.

Local execution defaults to a fixed development identity. Production must explicitly select Cloudflare mode.

## Consequences

- Cloudflare owns login UI, identity-provider integration, sessions, and access policies.
- Lakes owns User records, Workspace assignment, roles, and resource authorization.
- Regular Users can access only their default Workspace. Admins may manage Users and inspect models across Workspaces.
- A separately enabled local-network bypass may bind requests from a trusted CIDR directly to one configured User; it never applies when a Cloudflare token is present.
- The origin must not be publicly reachable around Cloudflare Access; use Cloudflare Tunnel or equivalent firewall restrictions.
- WeChat login can be added later as a Cloudflare-supported identity provider without changing the Lakes User/Workspace model.
