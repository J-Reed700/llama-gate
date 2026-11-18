# syntax=docker/dockerfile:1.7
# A static llama-gate binary on Alpine.
FROM rust:1.95-alpine AS build
RUN apk add --no-cache musl-dev
WORKDIR /source
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/source/target \
    cargo build --release --locked -p llama-gate && \
    cp target/release/llama-gate /llama-gate

FROM alpine:3.22
LABEL org.opencontainers.image.title="llama-gate" \
      org.opencontainers.image.description="KV-aware admission, fair scheduling, and pausing in front of llama.cpp servers" \
      org.opencontainers.image.source="https://github.com/J-Reed700/llama-gate" \
      org.opencontainers.image.licenses="MIT"
RUN addgroup -S gate && adduser -S -G gate -h /var/lib/llama-gate gate
COPY --from=build /llama-gate /usr/local/bin/llama-gate
USER gate
# The listen address comes from the configuration (default 0.0.0.0:8080).
# The probe reads the same configuration, so it always checks the right port.
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD ["llama-gate", "--probe"]
ENTRYPOINT ["llama-gate"]
