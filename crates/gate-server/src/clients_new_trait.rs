pub trait Identify: Send + Sync + std::fmt::Debug {
    /// Phase 1: Identify using only headers and peer information.
    /// Returns:
    /// - `Ok(Some(Identity))` if fully identified.
    /// - `Ok(None)` if potentially valid but needs body.
    /// - `Err(Denied)` if unauthorized.
    fn identify_headers(&self, headers: &HeaderMap, peer: IpAddr) -> Result<Option<Identity>, Denied>;

    /// Phase 2: Complete identification using the request body.
    fn identify_body(&self, identity: Option<Identity>, body: &Value) -> Result<Identity, Denied>;

    // ... existing methods (precheck, is_admin, owns_key, refresh) ...
}
