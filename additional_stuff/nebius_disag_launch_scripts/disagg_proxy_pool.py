#!/usr/bin/env python3
"""Round-robin proxy for multi-instance disaggregated prefill/decode serving."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int

    @property
    def key(self) -> str:
        return f"{self.host}:{self.port}"


class BackendPool:
    def __init__(self, endpoints: List[Endpoint]) -> None:
        if not endpoints:
            raise ValueError("BackendPool requires at least one endpoint")
        self._endpoints = endpoints
        self._next_index = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> Endpoint:
        async with self._lock:
            endpoint = self._endpoints[self._next_index]
            self._next_index = (self._next_index + 1) % len(self._endpoints)
            return endpoint

    @property
    def keys(self) -> List[str]:
        return [ep.key for ep in self._endpoints]


class ProxyAppState:
    def __init__(
        self,
        prefill_endpoints: List[Endpoint],
        decode_endpoints: List[Endpoint],
        request_timeout_sec: int,
    ) -> None:
        self.prefill_pool = BackendPool(prefill_endpoints)
        self.decode_pool = BackendPool(decode_endpoints)
        self.request_timeout_sec = request_timeout_sec
        self.clients: Dict[str, httpx.AsyncClient] = {}

    async def open(self) -> None:
        timeout = httpx.Timeout(
            connect=10.0,
            read=self.request_timeout_sec,
            write=self.request_timeout_sec,
            pool=self.request_timeout_sec,
        )
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)

        for endpoint in self.prefill_pool.keys + self.decode_pool.keys:
            self.clients[endpoint] = httpx.AsyncClient(
                base_url=f"http://{endpoint}/v1",
                timeout=timeout,
                limits=limits,
            )

    async def close(self) -> None:
        for client in self.clients.values():
            await client.aclose()

    def client_for(self, endpoint: Endpoint) -> httpx.AsyncClient:
        return self.clients[endpoint.key]


def parse_endpoint_csv(value: str) -> List[Endpoint]:
    endpoints: List[Endpoint] = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise ValueError(
                f"Invalid endpoint '{stripped}'. Expected format host:port"
            )
        host, port_text = stripped.rsplit(":", 1)
        endpoints.append(Endpoint(host=host, port=int(port_text)))
    if not endpoints:
        raise ValueError("Endpoint list cannot be empty")
    return endpoints


def request_headers(req: Request) -> Dict[str, str]:
    forwarded = {}
    auth = req.headers.get("authorization")
    if auth:
        forwarded["authorization"] = auth
    req_id = req.headers.get("x-request-id")
    if req_id:
        forwarded["x-request-id"] = req_id
    return forwarded


def prefill_payload(payload: Dict) -> Dict:
    prefill = dict(payload)
    prefill["max_tokens"] = 1
    if "max_completion_tokens" in prefill:
        prefill["max_completion_tokens"] = 1
    prefill["stream"] = False
    return prefill


def create_app(
    prefill_endpoints: List[Endpoint],
    decode_endpoints: List[Endpoint],
    request_timeout_sec: int,
) -> FastAPI:
    state = ProxyAppState(
        prefill_endpoints=prefill_endpoints,
        decode_endpoints=decode_endpoints,
        request_timeout_sec=request_timeout_sec,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await state.open()
        app.state.proxy_state = state
        try:
            yield
        finally:
            await state.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> Dict[str, List[str] | str]:
        return {
            "status": "ok",
            "prefill_backends": state.prefill_pool.keys,
            "decode_backends": state.decode_pool.keys,
        }

    async def run_prefill_then_decode(
        endpoint_suffix: str,
        payload: Dict,
        headers: Dict[str, str],
    ):
        prefill_ep = await state.prefill_pool.acquire()
        decode_ep = await state.decode_pool.acquire()

        prefill_client = state.client_for(prefill_ep)
        decode_client = state.client_for(decode_ep)

        try:
            prefill_resp = await prefill_client.post(
                endpoint_suffix,
                json=prefill_payload(payload),
                headers=headers,
            )
            prefill_resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Prefill request failed for backend {prefill_ep.key}: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc

        if payload.get("stream", False):

            async def stream_decode() -> AsyncGenerator[bytes, None]:
                async with decode_client.stream(
                    "POST",
                    endpoint_suffix,
                    json=payload,
                    headers=headers,
                ) as response:
                    try:
                        response.raise_for_status()
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    except httpx.HTTPError as exc:
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                f"Decode stream failed for backend {decode_ep.key}: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        ) from exc

            return StreamingResponse(stream_decode(), media_type="text/event-stream")

        try:
            decode_resp = await decode_client.post(
                endpoint_suffix,
                json=payload,
                headers=headers,
            )
            decode_resp.raise_for_status()
            return JSONResponse(status_code=decode_resp.status_code, content=decode_resp.json())
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Decode request failed for backend {decode_ep.key}: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc

    @app.post("/v1/completions")
    async def completions(request: Request):
        payload = await request.json()
        headers = request_headers(request)
        return await run_prefill_then_decode("/completions", payload, headers)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        headers = request_headers(request)
        return await run_prefill_then_decode("/chat/completions", payload, headers)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--prefill-endpoints",
        required=True,
        help="Comma-separated host:port list",
    )
    parser.add_argument(
        "--decode-endpoints",
        required=True,
        help="Comma-separated host:port list",
    )
    parser.add_argument("--request-timeout-sec", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prefill_endpoints = parse_endpoint_csv(args.prefill_endpoints)
    decode_endpoints = parse_endpoint_csv(args.decode_endpoints)

    app = create_app(
        prefill_endpoints=prefill_endpoints,
        decode_endpoints=decode_endpoints,
        request_timeout_sec=args.request_timeout_sec,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
