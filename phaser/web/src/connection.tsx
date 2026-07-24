import { PrimitiveAtom, createStore } from 'jotai';
import { backOff } from 'exponential-backoff';

type Store = ReturnType<typeof createStore>;

// Send a heartbeat ping after this long without any message (of any kind -- a topic
// update counts just as much as a pong). Force-reconnect if nothing at all arrives
// within `DEAD_TIMEOUT_MS`, rather than waiting on the browser's own (much slower, if it
// happens at all) detection of a dead connection.
const IDLE_PING_MS = 30_000;
const DEAD_TIMEOUT_MS = 60_000;

const BACKOFF_STARTING_MS = 500;
const BACKOFF_MAX_MS = 5 * 60_000;

export type ConnectionStatus =
    | { type: 'connecting' }
    | { type: 'connected' }
    | { type: 'reconnecting'; attempt: number; retryAt: Date }
    | { type: 'disconnected' }
    | { type: 'stopped' };

export class WebsocketConnection {
    socket: WebSocket | null = null;

    private idleTimer: ReturnType<typeof setTimeout> | null = null;
    private deadTimer: ReturnType<typeof setTimeout> | null = null;
    // Set by `disconnect()` only. Distinguishes an intentional teardown (component
    // unmounted) from every other closure (heartbeat-forced, server-initiated, network
    // drop), which all fall through to `_reconnectLoop`.
    private shouldReconnect = false;

    public constructor(
        public readonly address: string,
        public readonly store: Store,
        public readonly lastSeen: PrimitiveAtom<Date | null>,
        public readonly status: PrimitiveAtom<ConnectionStatus>,
        public readonly onMessage: ((_: MessageEvent<any>) => void) | null,
        // called after the socket opens (including on reconnect) -- lets a caller (e.g.
        // `PubSubConnection`) replay state that only makes sense once connected, such as
        // re-sending active subscriptions.
        public readonly onOpen: (() => void) | null = null,
    ) { }

    connect() {
        this.shouldReconnect = true;
        this._reconnectLoop();
    }

    disconnect() {
        this.shouldReconnect = false;
        this._clearHeartbeat();
        if (this.socket) {
            console.log(`disconnecting from '${this.address}'...`);
            this.socket.close();
        }
        this.store.set(this.status, { type: 'disconnected' });
    }

    // Called when a `ServerShutdownMessage` is seen (see `pubsub.tsx`): the server is
    // going away on purpose, so there's no point auto-reconnecting -- the close that
    // immediately follows is expected, not a failure to retry past.
    notifyServerShutdown() {
        this.shouldReconnect = false;
        this.store.set(this.status, { type: 'stopped' });
    }

    send(data: unknown) {
        // Only transmit on an OPEN socket. Sends attempted while CONNECTING (e.g. a
        // component subscribing on mount, before `onopen`) would throw an
        // InvalidStateError DOMException; those are safely dropped here because `_open`
        // replays all active subscriptions via `onOpen` once the socket is usable.
        if (this.socket?.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(data));
        }
    }

    // Retries `_attemptOnce` with exponential backoff until it succeeds or
    // `shouldReconnect` goes false. Called once from `connect()`, and again (fresh)
    // every time an established connection subsequently drops.
    private async _reconnectLoop() {
        this.store.set(this.status, { type: 'connecting' });
        try {
            await backOff(() => this._attemptOnce(), {
                numOfAttempts: Number.POSITIVE_INFINITY,
                startingDelay: BACKOFF_STARTING_MS,
                maxDelay: BACKOFF_MAX_MS,
                jitter: 'full',
                retry: (_error, attemptNumber) => {
                    if (!this.shouldReconnect) return false;
                    const delay = Math.min(BACKOFF_STARTING_MS * 2 ** (attemptNumber - 1), BACKOFF_MAX_MS);
                    this.store.set(this.status, {
                        type: 'reconnecting', attempt: attemptNumber, retryAt: new Date(Date.now() + delay),
                    });
                    return true;
                },
            });
        } catch {
            // only reached once `shouldReconnect` went false mid-backoff (`retry` above)
        }
    }

    // One connection attempt: resolves once OPEN, rejects if it closes/errors first.
    // Once resolved, a later drop of this same socket no longer touches this promise --
    // `onclose` instead starts a brand new `_reconnectLoop`.
    private _attemptOnce(): Promise<void> {
        if (!this.shouldReconnect) return Promise.reject(new Error('disconnected'));

        return new Promise((resolve, reject) => {
            console.log(`connecting to '${this.address}'...`);
            const socket = new WebSocket(this.address);
            socket.binaryType = "arraybuffer";
            this.socket = socket;
            let opened = false;

            socket.onopen = (event) => {
                opened = true;
                this._open(event);
                resolve();
            };
            socket.onerror = () => { };
            socket.onclose = (event) => {
                this._clearHeartbeat();
                if (!opened) {
                    reject(new Error(`connection closed before opening (code ${event.code})`));
                    return;
                }
                if (this.shouldReconnect) {
                    this.store.set(this.status, { type: 'disconnected' });
                    this._reconnectLoop();
                }
                // else: `disconnect()`/`notifyServerShutdown()` already set a terminal
                // status ('disconnected'/'stopped') -- don't overwrite it.
            };
            socket.onmessage = this._message.bind(this);
        });
    }

    private _open(event: Event) {
        this.store.set(this.status, { type: 'connected' });
        this.store.set(this.lastSeen, new Date(event.timeStamp));
        this._scheduleHeartbeat();
        if (this.onOpen) {
            this.onOpen();
        }
    }

    private _message(event: MessageEvent<any>) {
        this.store.set(this.lastSeen, new Date(event.timeStamp));
        this._scheduleHeartbeat();

        if (this.onMessage) {
            this.onMessage(event);
        }
    }

    // (Re)schedules both heartbeat timers from now. Called on every message (any
    // message counts as proof of life, not just a pong) and on open.
    private _scheduleHeartbeat() {
        this._clearHeartbeat();
        this.idleTimer = setTimeout(() => this.send({ msg: 'ping' }), IDLE_PING_MS);
        this.deadTimer = setTimeout(() => {
            console.warn(`no message from '${this.address}' in ${DEAD_TIMEOUT_MS}ms, reconnecting`);
            this.socket?.close();
        }, DEAD_TIMEOUT_MS);
    }

    private _clearHeartbeat() {
        if (this.idleTimer !== null) clearTimeout(this.idleTimer);
        if (this.deadTimer !== null) clearTimeout(this.deadTimer);
        this.idleTimer = this.deadTimer = null;
    }
}
