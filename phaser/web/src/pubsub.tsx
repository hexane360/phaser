// Topic-based pub/sub client (`WEB_FIXES` #10). `PubSubConnection` wraps a single
// `/listen` websocket, ref-counting subscriptions per canonical topic so multiple
// components can independently subscribe/unsubscribe to the same topic without sending
// duplicate `sub`/`unsub` messages. It is deliberately generic: it has no notion of the
// `?job=` default-topic scoping the server applies -- it sends/receives topics exactly as
// given, and the server echoes updates back keyed by that same client-sent form.

import React from 'react';
import { atom, PrimitiveAtom, useStore, createStore } from 'jotai';

import { Topic, ClientMessage, ServerMessage, canonicalTopic } from './types';
import { decodeState } from './array';
import { WebsocketConnection, ConnectionStatus } from './connection';
import { ConnectionModal } from './notify';

type Store = ReturnType<typeof createStore>;

// A subscriber sees either a new (decoded) value + optional cause, or a subscription error.
export type TopicMessage = { data: any; cause: any } | { error: string };
export type TopicListener = (msg: TopicMessage) => void;

interface Subscription {
    topic: Topic;
    listeners: Set<TopicListener>;
}

// What a view's atom holds. `pending` is distinct from an empty `ok` -- "not fetched yet"
// and "this job has no object" are different things to render.
export type ViewState<T> =
    | { status: 'pending' }
    | { status: 'ok'; data: T }
    | { status: 'error'; reason: string };

export interface ViewOptions<T> {
    // An append-conflated topic (`logs`) publishes deltas rather than a whole value; `reduce`
    // folds each one into the accumulator, starting from `initial`.
    initial?: T;
    reduce?: (prev: T, delta: any) => T;
    // History the socket never replays. Runs on first subscribe and again on every
    // reconnect; `merge` folds the result into whatever has accumulated since, so updates
    // that land mid-fetch aren't overwritten. `prev` is what's held at the time of the
    // call, so a reconnecting view can fetch exactly what it missed.
    hydrate?: (prev: T | undefined) => Promise<T>;
    merge?: (prev: T | undefined, fetched: T) => T;
}

// One atom per canonical topic, shared by every component that asks for it, so N widgets
// on the same view are one subscription over one piece of state -- and widget N+1 sees the
// value that's already there instead of waiting for the next publish.
interface ViewEntry {
    topic: Topic;
    options: ViewOptions<any>;
    atom: PrimitiveAtom<ViewState<any>>;
    refs: number;
    stop: (() => void) | null;
}

export class PubSubConnection {
    public readonly status: PrimitiveAtom<ConnectionStatus>;
    public readonly lastSeen: PrimitiveAtom<Date | null>;

    private conn: WebsocketConnection;
    private store: Store;
    private subscriptions: Map<string, Subscription> = new Map();
    private views: Map<string, ViewEntry> = new Map();
    private reconnectListeners: Set<() => void> = new Set();
    private everConnected = false;

    public constructor(address: string, store: Store) {
        this.store = store;
        this.status = atom<ConnectionStatus>({ type: 'connecting' });
        this.lastSeen = atom<Date | null>(null);
        this.conn = new WebsocketConnection(
            address, store, this.lastSeen, this.status,
            this._onMessage.bind(this), this._onOpen.bind(this),
        );
        this.conn.connect();
    }

    disconnect() {
        this.conn.disconnect();
    }

    // Reconnects after a `disconnect()`; the socket's own `onclose` handles every
    // unintentional drop, so this is only for a deliberate teardown (see `PubSubProvider`).
    connect() {
        this.conn.connect();
    }

    // Resolves once the socket is live, retrying immediately rather than waiting out a
    // backoff. Rejects with `NotConnectedError` on timeout. See `requests.tsx`.
    ensureConnected(timeoutMs?: number): Promise<void> {
        return this.conn.ensureConnected(timeoutMs);
    }

    // Subscribes `listener` to `topic`, sending a `sub` message for the first listener on
    // a given (canonical) topic. Returns an unsubscribe function; the last listener
    // removed for a topic sends `unsub`.
    subscribe(topic: Topic, listener: TopicListener): () => void {
        const key = canonicalTopic(topic);
        let sub = this.subscriptions.get(key);
        if (!sub) {
            sub = { topic, listeners: new Set() };
            this.subscriptions.set(key, sub);
            this._send({ msg: 'sub', topics: [topic] });
        }
        sub.listeners.add(listener);

        let unsubscribed = false;
        return () => {
            if (unsubscribed) return;
            unsubscribed = true;
            const current = this.subscriptions.get(key);
            if (!current) return;
            current.listeners.delete(listener);
            if (current.listeners.size === 0) {
                this.subscriptions.delete(key);
                this._send({ msg: 'unsub', topics: [topic] });
            }
        };
    }

    // Idempotent and keyed, so it's safe to call during render. Creating the atom is
    // separate from subscribing (`retainView`): the atom has to exist before the effect
    // that starts the subscription runs, and its identity must survive the ref count
    // hitting zero.
    viewAtom<T>(key: string, topic: Topic, options: ViewOptions<T> = {}): PrimitiveAtom<ViewState<T>> {
        let entry = this.views.get(key);
        if (!entry) {
            entry = { topic, options, atom: atom<ViewState<any>>({ status: 'pending' }), refs: 0, stop: null };
            this.views.set(key, entry);
        }
        return entry.atom;
    }

    // Subscribes on the first consumer, unsubscribes after the last one leaves. Returns the
    // release function for an effect cleanup.
    retainView(key: string): () => void {
        const entry = this.views.get(key);
        if (!entry) return () => {};

        entry.refs += 1;
        if (entry.refs === 1) this._startView(entry);

        let released = false;
        return () => {
            if (released) return;
            released = true;
            entry.refs -= 1;
            if (entry.refs === 0) this._stopView(entry);
        };
    }

    private _startView(entry: ViewEntry) {
        const { initial, reduce, hydrate, merge } = entry.options;
        const set = (fn: (prev: ViewState<any>) => ViewState<any>) => this.store.set(entry.atom, fn);

        const unsubscribe = this.subscribe(entry.topic, (msg) => {
            if ('error' in msg) {
                set(() => ({ status: 'error', reason: msg.error }));
                return;
            }
            set((prev) => {
                if (!reduce) return { status: 'ok', data: msg.data };
                return { status: 'ok', data: reduce(prev.status === 'ok' ? prev.data : initial, msg.data) };
            });
        });

        const runHydrate = () => {
            if (!hydrate) return;
            const held = this.store.get(entry.atom);
            hydrate(held.status === 'ok' ? held.data : undefined)
                .then((fetched) => set((cur) => ({
                    status: 'ok',
                    data: merge ? merge(cur.status === 'ok' ? cur.data : undefined, fetched) : fetched,
                })))
                .catch((e) => set((cur) => cur.status === 'ok' ? cur : { status: 'error', reason: String(e) }));
        };
        runHydrate();
        const offReconnect = this.onReconnect(runHydrate);

        entry.stop = () => { unsubscribe(); offReconnect(); };
    }

    // Drops the accumulated value along with the subscription: for `object`/`probe` that's
    // a multi-megabyte array nobody is rendering any more. The (empty) entry stays, so a
    // remounting consumer keeps the same atom and re-subscribes for a fresh snapshot.
    private _stopView(entry: ViewEntry) {
        entry.stop?.();
        entry.stop = null;
        this.store.set(entry.atom, () => ({ status: 'pending' }));
    }

    private _send(msg: ClientMessage) {
        this.conn.send(msg);
    }

    // Notifies `listener` on every *re*connect (not the initial connect -- callers
    // typically fetch their own initial state separately, e.g. `Logs`'s mount effect).
    // Used to re-fetch anything that could have been missed while disconnected.
    onReconnect(listener: () => void): () => void {
        this.reconnectListeners.add(listener);
        return () => this.reconnectListeners.delete(listener);
    }

    // Replays every currently-active subscription. Called on (re)connect: the server has
    // no memory of a dropped connection's subscriptions, so a reconnecting session must
    // re-`sub` everything to keep receiving updates (and to get fresh retained snapshots).
    private _onOpen() {
        const topics = Array.from(this.subscriptions.values(), (sub) => sub.topic);
        if (topics.length) {
            this._send({ msg: 'sub', topics });
        }

        if (this.everConnected) {
            for (const listener of this.reconnectListeners) listener();
        }
        this.everConnected = true;
    }

    private _onMessage(event: MessageEvent<any>) {
        let text: string;
        if (event.data instanceof ArrayBuffer) {
            text = new TextDecoder().decode(event.data);
        } else {
            text = event.data;
        }

        const msg: ServerMessage = JSON.parse(text);

        if (msg.msg === 'update') {
            for (const update of msg.updates) {
                const sub = this.subscriptions.get(canonicalTopic(update.topic));
                if (!sub) continue;
                const data = decodeState(update.data);
                for (const listener of sub.listeners) listener({ data, cause: update.cause ?? null });
            }
        } else if (msg.msg === 'error') {
            console.error(`pub/sub error on topic ${JSON.stringify(msg.topic)}: ${msg.reason}`);
            const sub = this.subscriptions.get(canonicalTopic(msg.topic));
            if (!sub) return;
            for (const listener of sub.listeners) listener({ error: msg.reason });
        } else if (msg.msg === 'pong') {
            // no-op: any message already counts as proof of life (`connection.tsx`)
        } else if (msg.msg === 'shutdown') {
            console.info('server is shutting down');
            this.conn.notifyServerShutdown();
        } else {
            console.warn(`Unknown pub/sub message: ${text}`);
        }
    }
}

interface PubSubProps {
    address: string;
}

const PubSubContext = React.createContext<PubSubConnection | null>(null);

// Creates and tears down a `PubSubConnection` for `address`, exposing it to descendants
// via context. Mirrors the lifecycle of the old `websocket()` hook (connect inside
// `useEffect`, disconnect on cleanup), but keyed off a stable connection object instead of
// bare status/lastSeen atoms.
export function PubSubProvider({ address, children }: React.PropsWithChildren<PubSubProps>) {
    const store = useStore();
    const [conn, setConn] = React.useState<PubSubConnection | null>(null);

    React.useEffect(() => {
        const connection = new PubSubConnection(address, store);
        setConn(connection);
        // devtools handle for testing reconnect/gap-fill behaviour by hand, the client-side
        // counterpart of the server's `POST /debug/disconnect`
        (window as any).phaser = {
            disconnect: () => connection.disconnect(),
            connect: () => connection.connect(),
        };
        return () => {
            connection.disconnect();
        };
    }, [address, store]);

    // rendered here so every page with a connection reports a lost one, without opting in
    return <PubSubContext.Provider value={conn}>
        {conn && <ConnectionModal status={conn.status}/>}
        {children}
    </PubSubContext.Provider>;
}

export function usePubSubConnection(): PubSubConnection | null {
    return React.useContext(PubSubContext);
}

// The atom holding `topic`'s state, shared with every other consumer of the same topic and
// subscribed for as long as at least one of them is mounted. Re-points at a different atom
// when `topic`'s canonical form changes (e.g. a `slice` param) or the connection is
// replaced, so anything derived from it belongs in a `useMemo` keyed on the atom.
//
// `options` is read only when the topic's entry is first created -- the first consumer of a
// topic decides its semantics, and later ones inherit them.
export function usePubSubView<T>(topic: Topic, options?: ViewOptions<T>): PrimitiveAtom<ViewState<T>> {
    const conn = usePubSubConnection();
    const key = canonicalTopic(topic);

    // stands in until the socket exists, so callers always get a real atom to read
    const [pending] = React.useState<PrimitiveAtom<ViewState<T>>>(() => atom({ status: 'pending' } as ViewState<T>));
    const target = conn ? conn.viewAtom<T>(key, topic, options) : pending;

    React.useEffect(() => conn?.retainView(key), [conn, key]);

    return target;
}
