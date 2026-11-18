//! A scheduled request from arrival to its end, across attempts.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) async fn generate(
    state: Arc<AppState>,
    config: Arc<Config>,
    parts: axum::http::request::Parts,
    peer: SocketAddr,
    dialect: Arc<dyn Dialect>,
    body: Value,
    size: usize,
    identity: Identity,
    model: ModelId,
    owns_key: bool,
) -> Response {
    let arrived = Instant::now();
    let Some(link) = state.backend_for(Some(model)) else {
        return error(
            StatusCode::BAD_GATEWAY,
            "server_error",
            "no backend serves this model".into(),
        );
    };
    let measurer = link.measurer.borrow().clone();
    let scope = config.clients.isolate.then(|| identity.name.to_string());
    let measurement = measurer
        .measure_in(dialect.as_ref(), &body, size, scope.as_deref())
        .await;
    let measured = arrived.elapsed();
    if measurement.mismatch {
        warn!(
            backend = link.name,
            "incremental tokenization disagreed with the server; prompts of this model are tokenized whole from now on"
        );
    }
    let class_name = config
        .class_names()
        .get(usize::from(identity.class.0))
        .cloned()
        .unwrap_or_default();
    let admission = &config.admission;
    let class = config.classes.get(&class_name);
    let (generation, min_generation) = if measurement.generates {
        let default = class
            .and_then(|class| class.default_generation)
            .unwrap_or(admission.default_generation);
        let ceiling = class
            .and_then(|class| class.max_generation)
            .or((admission.max_generation > 0).then_some(admission.max_generation));
        let mut generation = measurement.requested_generation.unwrap_or(default).max(1);
        if let Some(ceiling) = ceiling {
            generation = generation.min(ceiling.max(1));
        }
        (generation, admission.min_generation.min(generation))
    } else {
        (0, 0)
    };
    let model_name = state.models.name(model);
    let measured_label = match (measurement.exact, measurement.method) {
        (false, _) => "estimated",
        (true, Some(Tokenization::Incremental)) => "incremental",
        (true, _) => "exact",
    };

    if config.server.mode == Mode::Observe {
        info!(
            client = %identity.name,
            class = %class_name,
            model = model_name,
            dialect = dialect.name(),
            prompt = measurement.prompt,
            measured = measured_label,
            generation,
            "observe: forwarding without scheduling"
        );
        let bytes = Bytes::from(serde_json::to_vec(&body).unwrap_or_default());
        return relay(
            &state,
            &link,
            &parts.method,
            &parts.uri,
            &parts.headers,
            peer,
            bytes,
            owns_key,
        )
        .await;
    }

    let ticket = TicketId(state.tickets.fetch_add(1, Ordering::Relaxed) + 1);
    let control = Arc::new(Control::default());
    let (reply, grant) = oneshot::channel();
    let class_name: Arc<str> = Arc::from(class_name.as_str());
    state.handle.arrive(Arrival {
        ticket,
        demand: Demand {
            client: identity.client,
            class: identity.class,
            model,
            prompt: measurement.prompt,
            work: measurement.work,
            generation,
            min_generation,
            prefix: measurement.prefix.clone(),
            exclusive: measurement.exclusive,
            pinned: measurement.pinned,
        },
        client: identity.name.clone(),
        class: class_name.clone(),
        dialect: dialect.name(),
        measured: measured_label,
        control: control.clone(),
        reply,
    });
    let session = Session {
        state,
        config,
        ticket,
        control,
        method: parts.method,
        uri: parts.uri,
        headers: parts.headers,
        peer,
        streaming: measurement.streaming,
        requested: measurement.requested_generation,
        generates: measurement.generates,
        dialect,
        body,
        owns_key,
        arrived,
        waiting_since: Instant::now(),
        queued: Duration::ZERO,
        first_output: None,
        relayed: 0,
        ended: false,
        report: Report {
            ticket,
            at_ms: 0,
            client: identity.name,
            class: class_name,
            model: model_name,
            dialect: "",
            backend: None,
            slot: None,
            outcome: "cancelled",
            status: 499,
            prompt: measurement.prompt,
            cached: None,
            generated: None,
            measure_ms: measured.as_millis() as u64,
            queue_ms: 0,
            ttft_ms: None,
            total_ms: 0,
            tokens_per_second: None,
            preempted: 0,
            bounced: 0,
        },
    };
    session.run(grant).await
}

/// Times a request may be placed again because its server was not ready.
const MAX_BOUNCES: u32 = 8;

/// One attempt at running the request on a granted slot.
pub(super) enum Attempt {
    Response(reqwest::Response),
    /// Paused before its output began; the next grant arrives here.
    Paused(oneshot::Receiver<Grant>),
    /// The server refused the connection; the next grant arrives here.
    Bounced(oneshot::Receiver<Grant>),
    Failed(String),
}

/// A request from arrival to its end, across attempts.
pub(super) struct Session {
    state: Arc<AppState>,
    config: Arc<Config>,
    ticket: TicketId,
    control: Arc<Control>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    peer: SocketAddr,
    dialect: Arc<dyn Dialect>,
    body: Value,
    streaming: bool,
    requested: Option<Tokens>,
    generates: bool,
    owns_key: bool,
    arrived: Instant,
    /// When the current wait for a grant began.
    waiting_since: Instant,
    /// Time spent waiting for grants so far.
    queued: Duration,
    first_output: Option<Instant>,
    /// Output events sent to the client: what a request that ends early is
    /// charged for.
    relayed: u64,
    ended: bool,
    report: Report,
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.ended {
            // The client left, or the handler stopped: withdraw once.
            self.close("cancelled", 499, None);
        }
    }
}

impl Session {
    async fn run(mut self, mut grant: oneshot::Receiver<Grant>) -> Response {
        let keepalive = self.streaming && !self.config.queue.keepalive_interval.0.is_zero();
        loop {
            let granted = if keepalive {
                match tokio::time::timeout(self.config.queue.keepalive_after.0, &mut grant).await {
                    Ok(granted) => granted,
                    Err(_) => return self.commit_waiting(grant),
                }
            } else {
                (&mut grant).await
            };
            let (slot, generation, exclusive) = match granted {
                Ok(Grant::Run {
                    slot,
                    generation,
                    exclusive,
                    ..
                }) => (slot, generation, exclusive),
                Ok(Grant::Refused(rejection)) => return self.refuse(&rejection),
                Err(_) => {
                    return self.fail(StatusCode::SERVICE_UNAVAILABLE, "the scheduler stopped");
                }
            };
            match self.attempt(slot, generation, exclusive).await {
                Attempt::Response(response) => {
                    if self.streaming && response.status().is_success() {
                        return self.commit_stream(response);
                    }
                    // A whole answer (or an error): it is all the output.
                    if !self.control.begin() {
                        match self.control.take_regrant() {
                            Some(next) => {
                                self.report.preempted += 1;
                                grant = next;
                                continue;
                            }
                            None => {
                                return self.fail(
                                    StatusCode::SERVICE_UNAVAILABLE,
                                    "paused without a way to resume",
                                );
                            }
                        }
                    }
                    return self.relay_whole(response);
                }
                Attempt::Paused(next) => {
                    self.report.preempted += 1;
                    grant = next;
                }
                Attempt::Bounced(next) => {
                    self.report.bounced += 1;
                    grant = next;
                }
                Attempt::Failed(message) => return self.fail(StatusCode::BAD_GATEWAY, &message),
            }
        }
    }

    /// Sends the request to its granted slot, unless it is paused first.
    async fn attempt(&mut self, slot: SlotId, generation: Tokens, exclusive: bool) -> Attempt {
        self.queued += self.waiting_since.elapsed();
        let Some(link) = self
            .state
            .backends
            .get(usize::from(slot.backend.0))
            .cloned()
        else {
            return Attempt::Failed(format!("granted unknown backend {}", slot.backend));
        };
        self.report.backend = Some(link.name.clone());
        self.report.slot = Some(slot.index);
        let mut body = self.body.clone();
        if !exclusive {
            self.dialect.pin(&mut body, slot.index);
        }
        // The gate only ever lowers a limit: one the client set at or below
        // the granted generation stands as it is.
        if self.generates
            && self.config.admission.enforce_generation
            && self
                .requested
                .is_none_or(|requested| requested > generation)
        {
            self.dialect.limit_generation(&mut body, generation);
        }
        if let (Some(model), Some(object)) = (&link.router_model, body.as_object_mut()) {
            object.insert("model".into(), Value::String(model.clone()));
        }
        let bytes = Bytes::from(serde_json::to_vec(&body).unwrap_or_default());
        let request = upstream_request(
            &self.state,
            &link,
            &self.method,
            &self.uri,
            &self.headers,
            self.peer,
            bytes,
            self.owns_key,
        );
        let control = self.control.clone();
        let outcome = tokio::select! {
            biased;
            _ = control.paused() => None,
            result = request.send() => Some(result),
        };
        self.waiting_since = Instant::now();
        match outcome {
            None => match self.control.take_regrant() {
                Some(next) => Attempt::Paused(next),
                None => Attempt::Failed("paused without a way to resume".into()),
            },
            // A server still loading its model (or waking from sleep)
            // answers 503 before doing anything: like a refused connection,
            // the request is placed again rather than failed. Bounded, so a
            // server that only ever answers 503 still gets its answer through.
            Some(Ok(response))
                if response.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE
                    && self.report.bounced < MAX_BOUNCES =>
            {
                Attempt::Bounced(self.state.handle.bounce(self.ticket))
            }
            Some(Ok(response)) => Attempt::Response(response),
            Some(Err(error)) if error.is_connect() && self.report.bounced < MAX_BOUNCES => {
                Attempt::Bounced(self.state.handle.bounce(self.ticket))
            }
            Some(Err(error)) => Attempt::Failed(format!("upstream request failed: {error}")),
        }
    }

    /// Relays a whole (non-streaming or error) response, reporting when its
    /// body is through.
    fn relay_whole(mut self, response: reqwest::Response) -> Response {
        self.first_output = Some(Instant::now());
        self.state.handle.started(self.ticket);
        let status = response.status();
        let mut builder = Response::builder().status(status.as_u16());
        for (name, value) in response.headers() {
            if !is_hop_by_hop(name.as_str()) {
                builder = builder.header(name.as_str(), value.as_bytes());
            }
        }
        let stream = Leased {
            inner: response.bytes_stream(),
            reader: WholeReader::new(self.dialect.wire()),
            session: Some(self),
            status: status.as_u16(),
        };
        builder
            .body(Body::from_stream(stream))
            .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
    }

    /// Answers a streaming request whose server has answered: mirrors the
    /// server's status and headers, then streams through a channel that
    /// outlives this attempt, should the request be paused before it
    /// produces anything.
    fn commit_stream(self, response: reqwest::Response) -> Response {
        let mut builder = Response::builder().status(response.status().as_u16());
        for (name, value) in response.headers() {
            if !is_hop_by_hop(name.as_str()) {
                builder = builder.header(name.as_str(), value.as_bytes());
            }
        }
        let (sender, receiver) = mpsc::channel(64);
        tokio::spawn(self.pump(sender, Some(response), None));
        builder
            .body(channel_body(receiver))
            .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
    }

    /// Answers a streaming request that is still waiting: commits to a 200
    /// event stream now and pings it until the request runs. Later refusals
    /// and errors are reported in-band as `error` events, as llama.cpp
    /// reports mid-stream errors.
    fn commit_waiting(self, grant: oneshot::Receiver<Grant>) -> Response {
        let (sender, receiver) = mpsc::channel(64);
        tokio::spawn(self.pump(sender, None, Some(grant)));
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header("x-accel-buffering", "no")
            .body(channel_body(receiver))
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
    }

    /// Streams a committed response to its end: relays the server's events,
    /// and between attempts waits for the next grant with pings.
    async fn pump(
        mut self,
        sender: mpsc::Sender<Result<Bytes, std::io::Error>>,
        mut current: Option<reqwest::Response>,
        mut grant: Option<oneshot::Receiver<Grant>>,
    ) {
        let interval = self
            .config
            .queue
            .keepalive_interval
            .0
            .max(Duration::from_millis(100));
        loop {
            if let Some(response) = current.take() {
                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    let message = serde_json::from_str::<Value>(&text)
                        .ok()
                        .and_then(|value| {
                            value
                                .pointer("/error/message")
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                        })
                        .unwrap_or(text);
                    let _ = sender
                        .send(Ok(sse_error(
                            "server_error",
                            &format!("HTTP {status}: {message}"),
                        )))
                        .await;
                    self.close("error", status, None);
                    return;
                }
                let mut stream = response.bytes_stream();
                // Events are relayed whole, and output is told by meaning:
                // progress and opening events before the first token leave
                // the request pausable.
                let mut reader = Reader::new(self.dialect.wire());
                'attempt: loop {
                    let step = if self.control.started() {
                        Some(stream.next().await)
                    } else {
                        let control = self.control.clone();
                        tokio::select! {
                            biased;
                            _ = control.paused() => None,
                            chunk = stream.next() => Some(chunk),
                        }
                    };
                    match step {
                        None => {
                            // Paused before any output: drop this attempt.
                            self.report.preempted += 1;
                            grant = self.control.take_regrant();
                            break;
                        }
                        Some(Some(Ok(chunk))) => {
                            for piece in reader.read(&chunk) {
                                if self.first_output.is_none() && piece.kind == Kind::Output {
                                    if !self.control.begin() {
                                        self.report.preempted += 1;
                                        grant = self.control.take_regrant();
                                        break 'attempt;
                                    }
                                    self.first_output = Some(Instant::now());
                                    self.state.handle.started(self.ticket);
                                }
                                if piece.kind == Kind::Output {
                                    self.relayed += 1;
                                }
                                if sender.send(Ok(piece.bytes)).await.is_err() {
                                    // The client left; dropping the stream
                                    // cancels the request upstream.
                                    self.close("cancelled", 499, None);
                                    return;
                                }
                            }
                        }
                        Some(Some(Err(error))) => {
                            let _ = sender
                                .send(Ok(sse_error("server_error", &error.to_string())))
                                .await;
                            self.close("error", 502, None);
                            return;
                        }
                        Some(None) => {
                            if let Some(rest) = reader.finish() {
                                let _ = sender.send(Ok(rest)).await;
                            }
                            // llama.cpp ends a failed stream like any other:
                            // only its error event says it failed.
                            match reader.failure() {
                                Some(failure) => {
                                    warn!(ticket = %self.ticket, %failure, "the server reported an error mid-stream");
                                    self.close("error", 500, None);
                                }
                                None => self.close("ok", 200, Some(reader.usage())),
                            }
                            return;
                        }
                    }
                }
            }
            // Waiting for a grant (the first, or the next after a pause or
            // a bounce), pinging the client meanwhile.
            let Some(mut receiver) = grant.take() else {
                let _ = sender
                    .send(Ok(sse_error("server_error", "the request lost its place")))
                    .await;
                self.close("error", 500, None);
                return;
            };
            let mut ping = tokio::time::interval(interval);
            ping.tick().await;
            let granted = loop {
                tokio::select! {
                    granted = &mut receiver => break granted,
                    _ = ping.tick() => {
                        if sender.send(Ok(Bytes::from_static(b": queued\n\n"))).await.is_err() {
                            self.close("cancelled", 499, None);
                            return;
                        }
                    }
                }
            };
            match granted {
                Ok(Grant::Run {
                    slot,
                    generation,
                    exclusive,
                    ..
                }) => match self.attempt(slot, generation, exclusive).await {
                    Attempt::Response(response) => current = Some(response),
                    Attempt::Paused(next) => {
                        self.report.preempted += 1;
                        grant = Some(next);
                    }
                    Attempt::Bounced(next) => {
                        self.report.bounced += 1;
                        grant = Some(next);
                    }
                    Attempt::Failed(message) => {
                        let _ = sender.send(Ok(sse_error("server_error", &message))).await;
                        self.close("error", 502, None);
                        return;
                    }
                },
                Ok(Grant::Refused(rejection)) => {
                    let (status, kind, message) = describe(&rejection);
                    let _ = sender.send(Ok(sse_error(kind, &message))).await;
                    self.close("refused", status.as_u16(), None);
                    return;
                }
                Err(_) => {
                    let _ = sender
                        .send(Ok(sse_error("server_error", "the scheduler stopped")))
                        .await;
                    self.close("error", 503, None);
                    return;
                }
            }
        }
    }

    fn refuse(mut self, rejection: &Rejection) -> Response {
        let (status, ..) = describe(rejection);
        self.close("refused", status.as_u16(), None);
        refusal(rejection, self.config.queue.retry_after.0)
    }

    fn fail(mut self, status: StatusCode, message: &str) -> Response {
        self.close("error", status.as_u16(), None);
        error(status, "server_error", message.to_owned())
    }

    /// Ends the session: reports a completed request to the engine with
    /// its usage, anything else as withdrawn.
    fn close(&mut self, outcome: &'static str, status: u16, usage: Option<Usage>) {
        if self.ended {
            return;
        }
        self.ended = true;
        let now = Instant::now();
        let report = &mut self.report;
        report.dialect = self.dialect.name();
        report.outcome = outcome;
        report.status = status;
        report.queue_ms = self.queued.as_millis() as u64;
        report.total_ms = now.duration_since(self.arrived).as_millis() as u64;
        report.ttft_ms = self
            .first_output
            .map(|first| first.duration_since(self.arrived).as_millis() as u64);
        if usage.is_none() {
            // Ended early: charged what it was sent. An unstreamed request the
            // client abandoned may have generated anything up to its grant,
            // which is left to the engine to charge.
            report.generated = (self.streaming || outcome != "cancelled").then_some(self.relayed);
        }
        if let Some(usage) = usage {
            report.cached = usage.cached;
            report.generated = usage.generated;
            // The server's own measure; for a stream, failing that, the
            // rate the client saw.
            report.tokens_per_second = usage.tokens_per_second().or_else(|| {
                let generated = usage.generated.filter(|generated| *generated > 1)?;
                let seconds = now.duration_since(self.first_output?).as_secs_f64();
                (self.streaming && seconds > 0.0).then(|| generated as f64 / seconds)
            });
        }
        let report = report.clone();
        match usage {
            Some(usage) if outcome == "ok" => self.state.handle.finish(self.ticket, usage, report),
            _ => self.state.handle.withdraw(self.ticket, Some(report)),
        }
    }
}

pub(super) fn channel_body(receiver: mpsc::Receiver<Result<Bytes, std::io::Error>>) -> Body {
    Body::from_stream(futures_util::stream::unfold(
        receiver,
        |mut receiver| async move { receiver.recv().await.map(|item| (item, receiver)) },
    ))
}

/// A whole response's body, reporting its session's end when it is through
/// (or, if the client leaves first, its withdrawal).
pub(super) struct Leased<S> {
    inner: S,
    session: Option<Session>,
    reader: WholeReader,
    status: u16,
}

impl<S> Stream for Leased<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<Bytes, reqwest::Error>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let polled = self.inner.poll_next_unpin(context);
        match &polled {
            Poll::Ready(Some(Ok(chunk))) => {
                let chunk = chunk.clone();
                self.reader.read(&chunk);
            }
            Poll::Ready(None) => {
                let usage = self.reader.usage();
                let status = self.status;
                if let Some(mut session) = self.session.take() {
                    let outcome = if (200..300).contains(&status) {
                        "ok"
                    } else {
                        "error"
                    };
                    session.close(outcome, status, Some(usage));
                }
            }
            Poll::Ready(Some(Err(_))) => {
                if let Some(mut session) = self.session.take() {
                    session.close("error", 502, None);
                }
            }
            Poll::Pending => {}
        }
        polled
    }
}
