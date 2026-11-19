//! Who sent a request, whether it may, and which class it is scheduled in.
//!
//! [`Identify`] is the extension point: [`Rules`] implements it from
//! configuration (API keys, headers, networks, host names, the OpenAI
//! `user` field).
use crate::config::{AuthConfig, ClientRule, ClientsConfig, Secret};
use axum::http::HeaderMap;
use gate_core::{ClassId, ClientId};
use ipnet::IpNet;
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::net::IpAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// How long a host that stops resolving keeps its last addresses.
const KEEP_UNRESOLVED: Duration = Duration::from_secs(120);

/// Takes one lookup's result into what is known of `host`.
fn remember(
    known: &mut HashMap<String, Known>,
    host: String,
    found: Option<Vec<IpAddr>>,
    now: Instant,
) {
    match found {
        Some(addresses) => {
            known.insert(
                host,
                Known {
                    addresses,
                    failing_since: None,
                },
            );
        }
        None => {
            if let Some(entry) = known.get_mut(&host) {
                let since = *entry.failing_since.get_or_insert(now);
                if now.duration_since(since) > KEEP_UNRESOLVED {
                    known.remove(&host);
                }
            }
        }
    }
}

/// A rule host's addresses, and since when it has failed to resolve.
#[derive(Debug)]
struct Known {
    addresses: Vec<IpAddr>,
    failing_since: Option<Instant>,
}

/// A request's identity for scheduling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    pub client: ClientId,
    pub name: Arc<str>,
    pub class: ClassId,
}

/// What an identifier sees of a request. `body` is `Null` for requests the
/// gate does not schedule.
#[derive(Clone, Copy, Debug)]
pub struct Caller<'a> {
    pub peer: IpAddr,
    pub headers: &'a HeaderMap,
    pub body: &'a Value,
}

/// Why a request was turned away (HTTP 401).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Denied {
    MissingKey,
    UnknownKey,
}

impl Denied {
    pub fn message(self) -> &'static str {
        match self {
            Denied::MissingKey => "an API key is required (Authorization: Bearer <key>)",
            Denied::UnknownKey => "the API key is not recognised",
        }
    }
}

/// Decides who a request is from, and whether it may be served.
pub trait Identify: Send + Sync + std::fmt::Debug {
    fn identify(&self, caller: &Caller<'_>) -> Result<Identity, Denied>;

    /// Refuses a request from its headers alone, before its body is read
    /// (a missing or unknown required key). [`Identify::identify`] still
    /// decides once the body is in.
    fn precheck(&self, _headers: &HeaderMap) -> Result<(), Denied> {
        Ok(())
    }

    /// Whether the request may use the gate's own endpoints (status, events,
    /// metrics, reload).
    fn is_admin(&self, _headers: &HeaderMap) -> bool {
        true
    }

    /// Whether the request's credentials are the gate's own, to be removed
    /// before it is forwarded.
    fn owns_key(&self, _headers: &HeaderMap) -> bool {
        false
    }

    /// Upkeep run every `clients.resolve_interval`, such as re-resolving
    /// host names.
    fn refresh(&self) -> Option<Pin<Box<dyn Future<Output = ()> + Send + '_>>> {
        None
    }
}

/// The key a request presents: `Authorization: Bearer <key>` or
/// `x-api-key: <key>`.
pub fn presented_key(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| {
            value
                .strip_prefix("Bearer ")
                .or_else(|| value.strip_prefix("bearer "))
        })
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok())
        })
        .map(str::trim)
        .filter(|key| !key.is_empty())
}

/// Interns client names into small ids. Shared across reloads, so a client
/// keeps its id, and with it its fairness counter.
#[derive(Debug, Default)]
pub struct Names {
    ids: HashMap<Arc<str>, ClientId>,
}

/// Distinct client names the gate tracks. Names can come from headers, the
/// body's `user` field, or peer addresses; past this many, further ones
/// share one identity (and one fair share) instead of growing every table
/// keyed by client without bound.
pub const MAX_CLIENTS: usize = 4096;

/// The identity names beyond [`MAX_CLIENTS`] share.
pub const OVERFLOW_CLIENT: &str = "(other)";

impl Names {
    fn intern(&mut self, name: &str) -> (ClientId, Arc<str>) {
        if let Some((name, id)) = self.ids.get_key_value(name) {
            return (*id, name.clone());
        }
        if self.ids.len() >= MAX_CLIENTS && name != OVERFLOW_CLIENT {
            if !self.ids.contains_key(OVERFLOW_CLIENT) {
                tracing::warn!(
                    limit = MAX_CLIENTS,
                    "too many distinct clients; further ones share the identity \"(other)\""
                );
            }
            return self.intern(OVERFLOW_CLIENT);
        }
        let id = ClientId(self.ids.len() as u32);
        let name: Arc<str> = Arc::from(name);
        self.ids.insert(name.clone(), id);
        (id, name)
    }
}

#[derive(Debug)]
struct CompiledRule {
    name: String,
    class: ClassId,
    cidrs: Vec<IpNet>,
    hosts: Vec<String>,
    header: Option<(String, String)>,
}

#[derive(Debug)]
struct Key {
    secret: Secret,
    client: String,
    class: Option<ClassId>,
}

/// Configuration-driven identification.
#[derive(Debug)]
pub struct Rules {
    rules: Vec<CompiledRule>,
    default_class: ClassId,
    identity_header: Option<String>,
    class_header: Option<String>,
    classes: HashMap<String, ClassId>,
    use_openai_user: bool,
    names: Arc<Mutex<Names>>,
    /// Addresses of rule host names, refreshed by [`Rules::refresh`].
    resolved: RwLock<HashMap<IpAddr, usize>>,
    /// Each host's last successful lookup, kept through brief failures.
    known: Mutex<HashMap<String, Known>>,
    keys: Vec<Key>,
    required: bool,
    admin: Vec<Secret>,
}

impl Rules {
    pub fn new(
        clients: &ClientsConfig,
        auth: &AuthConfig,
        class_names: &[String],
        names: Arc<Mutex<Names>>,
    ) -> Self {
        let classes: HashMap<String, ClassId> = class_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), ClassId(index as u16)))
            .collect();
        let class_of = |name: Option<&String>| {
            name.and_then(|name| classes.get(name).copied())
                .or_else(|| classes.get(&clients.default_class).copied())
                .unwrap_or(ClassId(0))
        };
        let rules = clients
            .rules
            .iter()
            .map(|rule: &ClientRule| CompiledRule {
                name: rule.name.clone(),
                class: class_of(rule.class.as_ref()),
                cidrs: rule.cidrs.clone(),
                hosts: rule.hosts.clone(),
                header: rule
                    .header
                    .as_ref()
                    .map(|header| (header.name.to_ascii_lowercase(), header.value.clone())),
            })
            .collect();
        let keys = auth
            .keys
            .iter()
            .map(|key| Key {
                secret: key.key.clone(),
                client: key.client.clone(),
                class: key
                    .class
                    .as_ref()
                    .and_then(|name| classes.get(name).copied()),
            })
            .collect();
        Self {
            rules,
            default_class: class_of(None),
            identity_header: clients
                .identity_header
                .as_ref()
                .map(|name| name.to_ascii_lowercase()),
            class_header: clients
                .class_header
                .as_ref()
                .filter(|_| clients.trust_class_header)
                .map(|name| name.to_ascii_lowercase()),
            classes,
            use_openai_user: clients.use_openai_user,
            names,
            resolved: RwLock::new(HashMap::new()),
            known: Mutex::new(HashMap::new()),
            keys,
            required: auth.required,
            admin: auth.admin_keys.clone(),
        }
    }

    /// Re-resolves rule host names, so a recreated container's new address
    /// is recognized. A host whose lookup fails keeps its last addresses for
    /// a while: a container restarting is briefly unresolvable, and its
    /// clients should not change class meanwhile. Past that, the addresses
    /// are dropped, so a recycled address does not keep a stranger's identity.
    pub async fn resolve(&self) {
        let mut lookups = Vec::new();
        for host in self.rules.iter().flat_map(|rule| &rule.hosts) {
            let found = tokio::net::lookup_host((host.as_str(), 0))
                .await
                .ok()
                .map(|addresses| addresses.map(|address| address.ip()).collect::<Vec<_>>())
                .filter(|addresses| !addresses.is_empty());
            lookups.push((host.clone(), found));
        }
        let now = Instant::now();
        let mut resolved = HashMap::new();
        {
            let mut known = self
                .known
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for (host, found) in lookups {
                remember(&mut known, host, found, now);
            }
            for (index, rule) in self.rules.iter().enumerate() {
                for host in &rule.hosts {
                    for address in known.get(host).map_or(&[][..], |entry| &entry.addresses) {
                        resolved.entry(*address).or_insert(index);
                    }
                }
            }
        }
        if let Ok(mut current) = self.resolved.write() {
            *current = resolved;
        }
    }

    pub fn has_hosts(&self) -> bool {
        self.rules.iter().any(|rule| !rule.hosts.is_empty())
    }

    fn rule_for(&self, peer: IpAddr, headers: &HeaderMap) -> Option<&CompiledRule> {
        let peer = canonical(peer);
        let resolved = self.resolved.read().ok();
        self.rules.iter().enumerate().find_map(|(index, rule)| {
            let by_header = rule.header.as_ref().is_some_and(|(name, value)| {
                headers
                    .get(name.as_str())
                    .and_then(|header| header.to_str().ok())
                    == Some(value.as_str())
            });
            let by_network = rule.cidrs.iter().any(|net| net.contains(&peer));
            let by_host = resolved
                .as_ref()
                .and_then(|resolved| resolved.get(&peer))
                .is_some_and(|matched| *matched == index);
            (by_header || by_network || by_host).then_some(rule)
        })
    }

    fn header<'a>(headers: &'a HeaderMap, name: Option<&String>) -> Option<&'a str> {
        let value = headers.get(name?.as_str())?.to_str().ok()?.trim();
        (!value.is_empty() && value.len() <= 128).then_some(value)
    }

    fn key_for(&self, headers: &HeaderMap) -> Option<&Key> {
        let presented = presented_key(headers)?;
        self.keys.iter().find(|key| key.secret.matches(presented))
    }

    fn intern(&self, name: &str, class: ClassId) -> Identity {
        let (client, name) = self
            .names
            .lock()
            .map(|mut names| names.intern(name))
            .unwrap_or_else(|_| (ClientId(u32::MAX), Arc::from(name)));
        Identity {
            client,
            name,
            class,
        }
    }
}

impl Identify for Rules {
    fn precheck(&self, headers: &HeaderMap) -> Result<(), Denied> {
        if self.required && self.key_for(headers).is_none() {
            return Err(match presented_key(headers) {
                Some(_) => Denied::UnknownKey,
                None => Denied::MissingKey,
            });
        }
        Ok(())
    }

    fn identify(&self, caller: &Caller<'_>) -> Result<Identity, Denied> {
        let headers = caller.headers;
        let key = self.key_for(headers);
        if self.required && key.is_none() {
            return Err(match presented_key(headers) {
                Some(_) => Denied::UnknownKey,
                None => Denied::MissingKey,
            });
        }
        let rule = self.rule_for(caller.peer, headers);
        let flow = Self::header(headers, self.identity_header.as_ref());
        let trusted_class = Self::header(headers, self.class_header.as_ref())
            .and_then(|name| self.classes.get(name).copied());
        if let Some(key) = key {
            let name = match flow {
                Some(flow) => format!("{}/{flow}", key.client),
                None => key.client.clone(),
            };
            // A class given to a key is the key holder's, whatever header
            // they send; the header only chooses for keys without one.
            let class = key
                .class
                .or(trusted_class)
                .or(rule.map(|rule| rule.class))
                .unwrap_or(self.default_class);
            return Ok(self.intern(&name, class));
        }
        let user = self
            .use_openai_user
            .then(|| caller.body.get("user").and_then(Value::as_str))
            .flatten()
            .map(str::trim)
            .filter(|user| !user.is_empty() && user.len() <= 128);
        let name = match (rule, flow, user) {
            (Some(rule), Some(flow), _) => format!("{}/{flow}", rule.name),
            (Some(rule), None, _) => rule.name.clone(),
            (None, Some(flow), _) => flow.to_owned(),
            (None, None, Some(user)) => format!("user:{user}"),
            (None, None, None) => canonical(caller.peer).to_string(),
        };
        let class = trusted_class
            .or(rule.map(|rule| rule.class))
            .unwrap_or(self.default_class);
        Ok(self.intern(&name, class))
    }

    /// With admin keys, only they open the gate's own endpoints. Without
    /// any, a gate that requires keys asks for one of its keys there too; an
    /// open gate leaves them open.
    fn is_admin(&self, headers: &HeaderMap) -> bool {
        if !self.admin.is_empty() {
            return presented_key(headers)
                .is_some_and(|key| self.admin.iter().any(|secret| secret.matches(key)));
        }
        !self.required || self.key_for(headers).is_some()
    }

    fn owns_key(&self, headers: &HeaderMap) -> bool {
        presented_key(headers).is_some_and(|presented| {
            self.keys.iter().any(|key| key.secret.matches(presented))
                || self.admin.iter().any(|secret| secret.matches(presented))
        })
    }

    fn refresh(&self) -> Option<Pin<Box<dyn Future<Output = ()> + Send + '_>>> {
        self.has_hosts()
            .then(|| Box::pin(self.resolve()) as Pin<Box<dyn Future<Output = ()> + Send + '_>>)
    }
}

/// IPv4-mapped IPv6 addresses compare as IPv4.
fn canonical(address: IpAddr) -> IpAddr {
    match address {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(address, IpAddr::V4),
        v4 => v4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HeaderMatch, KeyConfig};
    use serde_json::json;

    fn rules_with(auth: AuthConfig) -> Rules {
        let config = ClientsConfig {
            rules: vec![
                ClientRule {
                    name: "chat-ui".into(),
                    class: Some("interactive".into()),
                    cidrs: vec!["10.0.0.0/24".parse().unwrap()],
                    ..ClientRule::default()
                },
                ClientRule {
                    name: "batch".into(),
                    class: Some("batch".into()),
                    header: Some(HeaderMatch {
                        name: "X-Job".into(),
                        value: "nightly".into(),
                    }),
                    ..ClientRule::default()
                },
            ],
            ..ClientsConfig::default()
        };
        Rules::new(
            &config,
            &auth,
            &["batch".into(), "interactive".into(), "standard".into()],
            Arc::default(),
        )
    }

    fn rules() -> Rules {
        rules_with(AuthConfig::default())
    }

    fn identify(
        rules: &Rules,
        peer: &str,
        headers: &HeaderMap,
        body: &Value,
    ) -> Result<Identity, Denied> {
        rules.identify(&Caller {
            peer: peer.parse().unwrap(),
            headers,
            body,
        })
    }

    #[test]
    fn rules_match_networks_and_headers_and_name_sub_flows() {
        let rules = rules();
        let identity = identify(&rules, "10.0.0.7", &HeaderMap::new(), &json!({})).unwrap();
        assert_eq!((&*identity.name, identity.class), ("chat-ui", ClassId(1)));

        let mut headers = HeaderMap::new();
        headers.insert("x-gate-client", "alice".parse().unwrap());
        let alice = identify(&rules, "10.0.0.7", &headers, &json!({})).unwrap();
        assert_eq!(&*alice.name, "chat-ui/alice");
        assert_ne!(alice.client, identity.client);

        let mut job = HeaderMap::new();
        job.insert("x-job", "nightly".parse().unwrap());
        let batch = identify(&rules, "192.168.1.1", &job, &json!({})).unwrap();
        assert_eq!((&*batch.name, batch.class), ("batch", ClassId(0)));
    }

    #[test]
    fn unmatched_requests_fall_back_to_user_then_address() {
        let rules = rules();
        let by_user = identify(
            &rules,
            "::ffff:192.168.1.9",
            &HeaderMap::new(),
            &json!({"user": "bob"}),
        )
        .unwrap();
        assert_eq!((&*by_user.name, by_user.class), ("user:bob", ClassId(2)));
        let by_address =
            identify(&rules, "::ffff:192.168.1.9", &HeaderMap::new(), &json!({})).unwrap();
        assert_eq!(&*by_address.name, "192.168.1.9");
    }

    #[test]
    fn the_class_header_is_ignored_unless_trusted() {
        let rules = rules();
        let mut headers = HeaderMap::new();
        headers.insert("x-gate-class", "interactive".parse().unwrap());
        let identity = identify(&rules, "192.168.1.1", &headers, &json!({})).unwrap();
        assert_eq!(identity.class, ClassId(2));
    }

    #[test]
    fn keys_name_their_client_and_can_be_required() {
        let auth = AuthConfig {
            required: true,
            keys: vec![KeyConfig {
                key: Secret::new("sk-desk"),
                client: "desktop-app".into(),
                class: Some("interactive".into()),
            }],
            admin_keys: vec![Secret::new("sk-admin")],
            public_metrics: false,
        };
        let rules = rules_with(auth);
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer sk-desk".parse().unwrap());
        let desk = identify(&rules, "10.0.0.7", &headers, &json!({})).unwrap();
        assert_eq!((&*desk.name, desk.class), ("desktop-app", ClassId(1)));
        assert!(rules.owns_key(&headers));
        assert!(!rules.is_admin(&headers));

        assert_eq!(
            identify(&rules, "10.0.0.7", &HeaderMap::new(), &json!({})),
            Err(Denied::MissingKey)
        );
        let mut wrong = HeaderMap::new();
        wrong.insert("x-api-key", "sk-nope".parse().unwrap());
        assert_eq!(
            identify(&rules, "10.0.0.7", &wrong, &json!({})),
            Err(Denied::UnknownKey)
        );

        let mut admin = HeaderMap::new();
        admin.insert("authorization", "Bearer sk-admin".parse().unwrap());
        assert!(rules.is_admin(&admin));
    }

    #[test]
    fn without_required_keys_an_unknown_key_is_the_servers_business() {
        let auth = AuthConfig {
            keys: vec![KeyConfig {
                key: Secret::new("sk-desk"),
                client: "desktop-app".into(),
                class: None,
            }],
            ..AuthConfig::default()
        };
        let rules = rules_with(auth);
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer for-llama-server".parse().unwrap());
        let identity = identify(&rules, "10.0.0.7", &headers, &json!({})).unwrap();
        assert_eq!(&*identity.name, "chat-ui");
        assert!(!rules.owns_key(&headers), "it is forwarded untouched");
        assert!(
            rules.is_admin(&headers),
            "no admin keys: the gate's endpoints are open"
        );
    }

    #[test]
    fn a_host_that_stops_resolving_keeps_its_addresses_for_a_while() {
        let mut known = HashMap::new();
        let start = Instant::now();
        let address: IpAddr = "172.18.0.5".parse().unwrap();
        remember(&mut known, "openwebui".into(), Some(vec![address]), start);
        remember(
            &mut known,
            "openwebui".into(),
            None,
            start + Duration::from_secs(5),
        );
        assert_eq!(
            known["openwebui"].addresses,
            vec![address],
            "restarting, not gone"
        );
        remember(
            &mut known,
            "openwebui".into(),
            None,
            start + Duration::from_secs(200),
        );
        assert!(
            !known.contains_key("openwebui"),
            "gone long enough to be forgotten"
        );
        remember(
            &mut known,
            "openwebui".into(),
            Some(vec![address]),
            start + Duration::from_secs(201),
        );
        assert!(known["openwebui"].failing_since.is_none());
    }

    #[test]
    fn a_keys_class_cannot_be_raised_by_a_header() {
        let config = ClientsConfig {
            trust_class_header: true,
            ..ClientsConfig::default()
        };
        let auth = AuthConfig {
            keys: vec![KeyConfig {
                key: Secret::new("sk-batch"),
                client: "nightly".into(),
                class: Some("batch".into()),
            }],
            ..AuthConfig::default()
        };
        let rules = Rules::new(
            &config,
            &auth,
            &["batch".into(), "interactive".into(), "standard".into()],
            Arc::default(),
        );
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer sk-batch".parse().unwrap());
        headers.insert("x-gate-class", "interactive".parse().unwrap());
        let identity = identify(&rules, "10.9.9.9", &headers, &json!({})).unwrap();
        assert_eq!(identity.class, ClassId(0), "the key's class holds");
    }

    #[test]
    fn a_gate_that_requires_keys_closes_its_own_endpoints_without_admin_keys() {
        let auth = AuthConfig {
            required: true,
            keys: vec![KeyConfig {
                key: Secret::new("sk-desk"),
                client: "desktop-app".into(),
                class: None,
            }],
            ..AuthConfig::default()
        };
        let rules = rules_with(auth);
        assert!(
            !rules.is_admin(&HeaderMap::new()),
            "anonymous callers are kept out"
        );
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer sk-desk".parse().unwrap());
        assert!(
            rules.is_admin(&headers),
            "a holder of one of its keys may look"
        );
    }

    #[test]
    fn distinct_clients_are_bounded() {
        let mut names = Names::default();
        let first = names.intern("client-0");
        for n in 1..MAX_CLIENTS {
            names.intern(&format!("client-{n}"));
        }
        let (overflow, name) = names.intern("one-too-many");
        assert_eq!(&*name, OVERFLOW_CLIENT);
        assert_eq!(
            names.intern("another").0,
            overflow,
            "they share one identity"
        );
        assert_eq!(names.intern("client-0"), first, "known names keep theirs");
        assert_eq!(names.ids.len(), MAX_CLIENTS + 1);
    }
}
