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
import { WebsocketConnection } from './websocket';

type Store = ReturnType<typeof createStore>;

// A subscriber sees either a new (decoded) value + optional cause, or a subscription error.
export type TopicMessage = { data: any; cause: any } | { error: string };
export type TopicListener = (msg: TopicMessage) => void;

interface Subscription {
    topic: Topic;
    listeners: Set<TopicListener>;
}

export class PubSubConnection {
    public readonly status: PrimitiveAtom<string>;
    public readonly lastSeen: PrimitiveAtom<Date | null>;

    private conn: WebsocketConnection;
    private subscriptions: Map<string, Subscription> = new Map();

    public constructor(address: string, store: Store) {
        this.status = atom('status');
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

    private _send(msg: ClientMessage) {
        this.conn.send(msg);
    }

    // Replays every currently-active subscription. Called on (re)connect: the server has
    // no memory of a dropped connection's subscriptions, so a reconnecting session must
    // re-`sub` everything to keep receiving updates (and to get fresh retained snapshots).
    private _onOpen() {
        const topics = Array.from(this.subscriptions.values(), (sub) => sub.topic);
        if (topics.length) {
            this._send({ msg: 'sub', topics });
        }
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
        return () => {
            connection.disconnect();
        };
    }, [address, store]);

    return <PubSubContext.Provider value={conn}>{children}</PubSubContext.Provider>;
}

export function usePubSubConnection(): PubSubConnection | null {
    return React.useContext(PubSubContext);
}

// Subscribes to `topic` for the component's lifetime, writing decoded updates into a
// (stable) jotai atom. Re-subscribes if `topic`'s canonical form changes (e.g. a `slice`
// param). Subscription errors are logged; the atom is left at its last-known value.
export function usePublishedAtom<T>(topic: Topic): PrimitiveAtom<T | null> {
    const conn = usePubSubConnection();
    const store = useStore();
    const [target] = React.useState<PrimitiveAtom<T | null>>(() => atom(null as T | null));
    const key = canonicalTopic(topic);

    React.useEffect(() => {
        if (!conn) return;
        return conn.subscribe(topic, (msg) => {
            if ('error' in msg) {
                console.error(`usePublishedAtom(${key}): ${msg.error}`);
                return;
            }
            store.set(target, (_) => msg.data as T);
        });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [conn, key, store, target]);

    return target;
}
