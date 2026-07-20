import { PrimitiveAtom, createStore } from 'jotai';

type Store = ReturnType<typeof createStore>;


export class WebsocketConnection {
    socket: WebSocket | null = null;

    public constructor(
        public readonly address: string,
        public readonly store: Store,
        public readonly lastSeen: PrimitiveAtom<Date | null>,
        public readonly status: PrimitiveAtom<string>,
        public readonly onMessage: ((_: MessageEvent<any>) => void) | null,
        // called after the socket opens (including on reconnect) -- lets a caller (e.g.
        // `PubSubConnection`) replay state that only makes sense once connected, such as
        // re-sending active subscriptions.
        public readonly onOpen: (() => void) | null = null,
    ) { }

    connect() {
        this.disconnect();

        console.log(`connecting to '${this.address}'...`);
        this.socket = new WebSocket(this.address);
        this.socket.binaryType = "arraybuffer";

        this.socket.onopen = this._open.bind(this);
        this.socket.onerror = this._error.bind(this);
        this.socket.onclose = this._close.bind(this);
        this.socket.onmessage = this._message.bind(this);
    }

    disconnect() {
        if (this.socket) {
            console.log(`disconnecting from '${this.address}...`);
            this.socket.close();
        }
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

    private _open(event: Event) {
        this.store.set(this.status, 'Connected');
        this.store.set(this.lastSeen, new Date(event.timeStamp));
        if (this.onOpen) {
            this.onOpen();
        }
    }

    private _error(event: Event) {

    }

    private _close(event: Event) {
        this.store.set(this.status, 'Disconnected');
    }

    private _message(event: MessageEvent<any>) {
        this.store.set(this.lastSeen, new Date(event.timeStamp));

        if (this.onMessage) {
            this.onMessage(event);
        }
    }
}